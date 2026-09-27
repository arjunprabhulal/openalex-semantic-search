import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import openalex_semantic_search.backfill as backfill
from openalex_semantic_search.embeddings import HashingEmbedder
from openalex_semantic_search.quantization import fit_int8_scales
from openalex_semantic_search.records import parse_work


def _work(work_id: int, title: str | None = None) -> dict:
    row = {"id": f"https://openalex.org/W{work_id}"}
    if title is not None:
        row["title"] = title
    return row


class _OverlapProbeEmbedder(HashingEmbedder):
    def __init__(
        self,
        two_shards_requested: threading.Event,
        third_shard_requested: threading.Event,
        requested: list[int],
    ):
        super().__init__()
        self._two_shards_requested = two_shards_requested
        self._third_shard_requested = third_shard_requested
        self._requested = requested
        self._first_call = True
        self.requested_during_first_embed: int | None = None

    def encode_documents(self, texts):
        if self._first_call:
            self._first_call = False
            assert self._two_shards_requested.wait(timeout=2), (
                "the producer did not prepare one shard while shard 0 embedded"
            )
            self.requested_during_first_embed = len(self._requested)
            assert not self._third_shard_requested.wait(timeout=0.2), (
                "the producer advanced into a second ahead shard"
            )
        return super().encode_documents(texts)


class _ConfiguredHashingEmbedder(HashingEmbedder):
    def __init__(
        self,
        *,
        name: str = "config-test-embedder",
        dimension: int = 32,
        dtype: str = "float32",
        max_tokens: int = 128,
        batch_size: int = 8,
    ):
        super().__init__(dimension=dimension)
        self.name = name
        self.dtype = dtype
        self.batch_size = batch_size
        self.model = SimpleNamespace(max_seq_length=max_tokens)


def test_backfill_ingestion_overlaps_exactly_one_shard_ahead(tmp_path: Path):
    two_shards_requested = threading.Event()
    third_shard_requested = threading.Event()
    requested: list[int] = []

    def works_source():
        for index in range(12):
            requested.append(index)
            if len(requested) == 8:
                two_shards_requested.set()
            elif len(requested) == 9:
                third_shard_requested.set()
            yield 0, index, _work(index + 1, f"paper {index + 1}")

    embedder = _OverlapProbeEmbedder(
        two_shards_requested,
        third_shard_requested,
        requested,
    )

    manifest = backfill.run_backfill(
        tmp_path / "artifacts",
        embedder,
        shard_size=4,
        limit=12,
        truth_stride=2,
        works_source=works_source(),
    )

    assert embedder.requested_during_first_embed == 8
    assert third_shard_requested.is_set()
    assert requested == list(range(12))
    assert manifest["records"] == 12
    assert manifest["shards"] == 3


def test_cancel_racing_with_slot_release_does_not_advance_source():
    stop = threading.Event()

    class RacingSemaphore:
        def __init__(self):
            self.releases = 0

        def acquire(self, timeout: float) -> bool:
            assert timeout == 0.1
            stop.set()
            return True

        def release(self) -> None:
            self.releases += 1

    slots = RacingSemaphore()

    assert not backfill._acquire_shard_slot(slots, stop)
    assert slots.releases == 1


def test_backfill_refuses_empty_finite_limit(tmp_path: Path):
    with pytest.raises(ValueError, match="limit must be positive"):
        backfill.run_backfill(
            tmp_path / "artifacts",
            HashingEmbedder(),
            limit=0,
            works_source=iter(()),
        )


def test_pipeline_preserves_id_order_and_exact_checkpoint_positions(tmp_path: Path):
    source = iter(
        [
            (2, 4, _work(10, "newest ten")),
            (2, 5, _work(11, "eleven")),
            (2, 6, _work(10, "stale ten")),
            (3, 0, _work(99)),  # rejected: no title
            (3, 1, _work(12, "twelve")),
            (3, 2, _work(13, "thirteen")),
            (4, 0, _work(14, "fourteen")),
        ]
    )
    artifacts = tmp_path / "artifacts"

    backfill.run_backfill(
        artifacts,
        HashingEmbedder(),
        shard_size=2,
        limit=5,
        truth_stride=1,
        works_source=source,
    )

    shard_ids = [
        np.load(artifacts / f"shard-{index:05d}.ids.npy").tolist()
        for index in range(3)
    ]
    markers = [
        json.loads((artifacts / f"shard-{index:05d}.done").read_text(encoding="utf-8"))
        for index in range(3)
    ]

    assert shard_ids == [[10, 11], [12, 13], [14]]
    assert [
        (marker["file_index"], marker["line_index"], marker["records_done"])
        for marker in markers
    ] == [
        (2, 6, 2),
        (3, 3, 4),
        (4, 1, 5),
    ]


def test_pipeline_accepts_worker_parsed_papers_and_rejected_rows(tmp_path: Path):
    first = parse_work(_work(1, "first"))
    second = parse_work(_work(2, "second"))
    assert first is not None and second is not None
    artifacts = tmp_path / "artifacts"

    backfill.run_backfill(
        artifacts,
        HashingEmbedder(),
        shard_size=2,
        limit=2,
        truth_stride=1,
        works_source=iter(
            [
                (7, 10, first),
                (7, 11, None),
                (7, 12, second),
            ]
        ),
    )

    marker = json.loads((artifacts / "shard-00000.done").read_text())
    assert (marker["file_index"], marker["line_index"]) == (7, 13)
    assert np.load(artifacts / "shard-00000.ids.npy").tolist() == [1, 2]


def test_dedupe_merges_at_interval_instead_of_after_each_shard(
    tmp_path: Path, monkeypatch
):
    merge_sizes: list[int] = []
    original_merge = backfill.SeenIds.merge_pending

    def tracked_merge(seen: backfill.SeenIds) -> None:
        merge_sizes.append(len(seen.pending))
        original_merge(seen)

    monkeypatch.setattr(backfill.SeenIds, "merge_pending", tracked_merge)
    backfill.run_backfill(
        tmp_path / "artifacts",
        HashingEmbedder(),
        shard_size=2,
        limit=8,
        works_source=(
            (0, index, _work(index + 1, f"paper {index + 1}")) for index in range(8)
        ),
    )

    assert merge_sizes == []

    seen = backfill.SeenIds(np.array([1, 5], dtype=np.int64))
    seen.add(2)
    seen.add(3)
    seen.merge_if_due(interval=3)
    assert seen.pending == {2, 3}
    seen.add(4)
    seen.merge_if_due(interval=3)
    assert seen.pending == set()
    assert seen.merged.tolist() == [1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "changed_setting",
    [
        "name",
        "dimension",
        "dtype",
        "max_tokens",
        "batch_size",
        "truth_stride",
        "include_expansion",
    ],
)
def test_resume_refuses_changes_to_pinned_corpus_and_quality_settings(
    tmp_path: Path, changed_setting: str
):
    artifacts = tmp_path / "artifacts"
    backfill.run_backfill(
        artifacts,
        _ConfiguredHashingEmbedder(),
        shard_size=2,
        limit=2,
        truth_stride=1,
        works_source=(
            (0, index, _work(index + 1, f"paper {index + 1}")) for index in range(4)
        ),
    )

    embedder_settings = {}
    resume_settings = {"truth_stride": 1, "include_expansion": False}
    if changed_setting == "name":
        embedder_settings["name"] = "different-embedder"
    elif changed_setting == "dimension":
        embedder_settings["dimension"] = 16
    elif changed_setting == "dtype":
        embedder_settings["dtype"] = "bfloat16"
    elif changed_setting == "max_tokens":
        embedder_settings["max_tokens"] = 256
    elif changed_setting == "batch_size":
        embedder_settings["batch_size"] = 16
    elif changed_setting == "truth_stride":
        resume_settings["truth_stride"] = 2
    elif changed_setting == "include_expansion":
        resume_settings["include_expansion"] = True

    with pytest.raises(RuntimeError, match="Backfill configuration changed"):
        backfill.run_backfill(
            artifacts,
            _ConfiguredHashingEmbedder(**embedder_settings),
            shard_size=2,
            limit=4,
            works_source=(
                (0, index, _work(index + 1, f"paper {index + 1}"))
                for index in range(6)
            ),
            **resume_settings,
        )


def test_resume_refuses_to_guess_missing_config_pin(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    embedder = _ConfiguredHashingEmbedder()
    backfill.run_backfill(
        artifacts,
        embedder,
        shard_size=2,
        limit=2,
        truth_stride=1,
        works_source=(
            (0, index, _work(index + 1, f"paper {index + 1}")) for index in range(4)
        ),
    )
    (artifacts / "backfill-config-pin.json").unlink()

    with pytest.raises(RuntimeError, match="no backfill-config-pin"):
        backfill.run_backfill(
            artifacts,
            embedder,
            shard_size=2,
            limit=4,
            truth_stride=1,
            works_source=(
                (0, index, _work(index + 1, f"paper {index + 1}")) for index in range(6)
            ),
        )


def test_resume_allows_larger_limit_and_records_compressed_shard_policy_history(
    tmp_path: Path,
):
    artifacts = tmp_path / "artifacts"
    embedder = _ConfiguredHashingEmbedder()
    backfill.run_backfill(
        artifacts,
        embedder,
        shard_size=2,
        limit=4,
        truth_stride=1,
        embed_subchunk=1,
        clip_fail_rate=1.0,
        works_source=(
            (0, index, _work(index + 1, f"paper {index + 1}")) for index in range(10)
        ),
    )

    manifest = backfill.run_backfill(
        artifacts,
        embedder,
        shard_size=3,
        limit=8,
        truth_stride=1,
        embed_subchunk=2,
        clip_fail_rate=1.0,
        works_source=(
            (0, index, _work(index + 1, f"paper {index + 1}")) for index in range(10)
        ),
    )

    marker_states = [
        json.loads(marker.read_text(encoding="utf-8"))
        for marker in sorted(artifacts.glob("shard-*.done"))
    ]
    marker_policies = [
        (marker["shard_size"], marker["embed_subchunk"]) for marker in marker_states
    ]
    assert manifest["records"] == 8
    assert manifest["shards"] == 4
    assert marker_policies == [(2, 1), (2, 1), (3, 2), (3, 2)]
    assert manifest["shard_config_history"] == [
        {
            "first_shard": 0,
            "last_shard": 1,
            "shard_size": 2,
            "embed_subchunk": 1,
        },
        {
            "first_shard": 2,
            "last_shard": 3,
            "shard_size": 3,
            "embed_subchunk": 2,
        },
    ]


def test_verified_config_pin_can_adopt_legacy_markers_without_rewriting_them(
    tmp_path: Path,
):
    artifacts = tmp_path / "artifacts"
    embedder = _ConfiguredHashingEmbedder()
    backfill.run_backfill(
        artifacts,
        embedder,
        shard_size=2,
        limit=2,
        truth_stride=1,
        works_source=(
            (0, index, _work(index + 1, f"paper {index + 1}")) for index in range(4)
        ),
    )
    marker_path = artifacts / "shard-00000.done"
    legacy_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    legacy_marker.pop("shard_size")
    legacy_marker.pop("embed_subchunk")
    marker_path.write_text(json.dumps(legacy_marker) + "\n", encoding="utf-8")

    manifest = backfill.run_backfill(
        artifacts,
        embedder,
        shard_size=3,
        limit=4,
        truth_stride=1,
        clip_fail_rate=1.0,
        works_source=(
            (0, index, _work(index + 1, f"paper {index + 1}")) for index in range(6)
        ),
    )

    assert manifest["legacy_marker_defaults_applied"] == 1
    assert manifest["shard_config_history"] == [
        {
            "first_shard": 0,
            "last_shard": 0,
            "shard_size": 2,
            "embed_subchunk": backfill.EMBED_SUBCHUNK_DEFAULT,
        },
        {
            "first_shard": 1,
            "last_shard": 1,
            "shard_size": 3,
            "embed_subchunk": backfill.EMBED_SUBCHUNK_DEFAULT,
        },
    ]


def test_zero_marker_start_removes_stale_scales_and_temporary_files(
    tmp_path: Path,
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "int8-scales.npy").write_bytes(b"corrupt uncommitted scales")
    stale_temporaries = [
        artifacts / ".tmp-shard-00000.int8.npy",
        artifacts / ".tmp-int8-scales.npy",
    ]
    for temporary in stale_temporaries:
        temporary.write_bytes(b"interrupted")

    embedder = _ConfiguredHashingEmbedder()
    backfill.run_backfill(
        artifacts,
        embedder,
        shard_size=2,
        limit=2,
        truth_stride=1,
        works_source=(
            (0, index, _work(index + 1, f"paper {index + 1}")) for index in range(2)
        ),
    )

    expected_scales = fit_int8_scales(embedder.encode_documents(["paper 1", "paper 2"]))
    np.testing.assert_array_equal(np.load(artifacts / "int8-scales.npy"), expected_scales)
    assert all(not temporary.exists() for temporary in stale_temporaries)
