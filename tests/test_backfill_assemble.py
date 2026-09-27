import gzip
import json
from pathlib import Path
import threading

import numpy as np
import pytest

import openalex_semantic_search.assemble as assemble_module
import openalex_semantic_search.backfill as backfill_module
from openalex_semantic_search.assemble import assemble_generation
from openalex_semantic_search.backfill import run_backfill
from openalex_semantic_search.benchmark import run_benchmark
from openalex_semantic_search.config import Stage
from openalex_semantic_search.embeddings import HashingEmbedder
from openalex_semantic_search.engine import SearchEngine
from openalex_semantic_search.quantization import fit_int8_scales, quantize_normalized
from openalex_semantic_search.records import parse_work
from openalex_semantic_search.store import Filters


def fake_work(index: int) -> dict:
    title = (
        "Attention Is All You Need"
        if index == 0
        else f"Measured research paper {index} on topic {index % 7}"
    )
    return {
        "id": f"https://openalex.org/W{index + 1}",
        "title": title,
        "publication_year": 2000 + index % 25,
        "cited_by_count": index % 300,
        "abstract_inverted_index": {f"word{index % 11}": [0], "study": [1], "method": [2]},
        "authorships": [{"author": {"display_name": f"Researcher {index}"}}],
        "open_access": {"is_oa": index % 2 == 0, "oa_url": f"https://example.test/{index}"},
        "type": "article",
    }


def works_stream(count: int, duplicates: int = 10):
    line = 0
    for index in range(count + duplicates):
        # Re-emit some earlier ids to prove dedupe holds across shards.
        actual = index % count if index >= count else index
        yield 0, line, fake_work(actual)
        line += 1


def test_backfill_assemble_search_roundtrip(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    embedder = HashingEmbedder()
    manifest = run_backfill(
        artifacts,
        embedder,
        shard_size=100,
        limit=320,
        truth_stride=32,
        works_source=works_stream(400),
    )
    assert manifest["records"] == 320
    assert manifest["shards"] == 4  # 100+100+100+20
    assert set(manifest["code_provenance"]) == {"git_commit", "git_dirty"}
    assert (artifacts / "int8-scales.npy").exists()
    assert (artifacts / "float32-truth-sample.npy").exists()
    assert manifest["checksums"]

    generation = tmp_path / "generation"
    assembled = assemble_generation(
        artifacts,
        stage=Stage("test", 320, 8, 2, candidate_count=320),
        output=generation,
        backend="numpy",
        embedder_name=embedder.name,
    )
    assert assembled["records"] == 320
    assert assembled["truth"] == "sampled"

    engine = SearchEngine(generation, embedder)
    try:
        response = engine.search("Attention is all you need", limit=5)
        assert response.results[0].title == "Attention Is All You Need"
        assert response.results[0].title_match is True
        assert response.corpus_records == 320
    finally:
        engine.close()

    # Benchmark must work without a full float32 truth matrix.
    report = run_benchmark(
        generation,
        embedder,
        ["research paper method"],
        title_queries=["Attention Is All You Need"],
        repeats=1,
        concurrency_levels=(1,),
    )
    assert report["truth_mode"] == "int8_exact_scan"
    assert report["exact_title_success_rate"] == 1.0
    assert report["int8_fidelity"] is not None
    assert report["int8_fidelity"]["cos_mean"] > 0.98


def test_backfill_resumes_from_shard_markers(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    embedder = HashingEmbedder()
    run_backfill(
        artifacts, embedder, shard_size=50, limit=100, works_source=works_stream(400)
    )
    assert len(list(artifacts.glob("shard-*.done"))) == 2
    # Resume with a higher limit: earlier shards must be kept, not recomputed,
    # and previously seen ids must not be re-emitted.
    manifest = run_backfill(
        artifacts, embedder, shard_size=50, limit=200, works_source=works_stream(400)
    )
    assert manifest["records"] == 200
    assert manifest["shards"] == 4
    all_ids = []
    for index in range(4):
        all_ids.extend(int(v) for v in __import__("numpy").load(artifacts / f"shard-{index:05d}.ids.npy"))
    assert len(all_ids) == 200 and len(set(all_ids)) == 200


def test_backfill_rerun_at_same_limit_is_idempotent(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    embedder = HashingEmbedder()
    first = run_backfill(
        artifacts, embedder, shard_size=50, limit=100, works_source=works_stream(400)
    )
    second = run_backfill(
        artifacts, embedder, shard_size=50, limit=100, works_source=works_stream(400)
    )
    assert first["records"] == second["records"] == 100
    assert first["shards"] == second["shards"] == 2


def test_finite_backfill_pins_and_deduplicates_benchmark_seed_overlay(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    embedder = HashingEmbedder()
    seed = fake_work(9_000)
    source = [
        (0, 0, seed),
        *((0, index + 1, fake_work(index)) for index in range(20)),
    ]

    manifest = run_backfill(
        artifacts,
        embedder,
        shard_size=10,
        limit=20,
        works_source=iter(source),
        seed_works=[seed],
    )

    import numpy as np

    ids = np.concatenate(
        [np.load(artifacts / f"shard-{index:05d}.ids.npy") for index in range(2)]
    )
    assert int(ids[0]) == 9_001
    assert len(ids) == len(set(int(value) for value in ids)) == 20
    assert manifest["benchmark_seed_overlay"]["openalex_ids"] == [
        "https://openalex.org/W9001"
    ]

    # Resume safety includes the overlay contents, not only the snapshot ETag.
    import pytest

    with pytest.raises(RuntimeError, match="seed overlay changed"):
        run_backfill(
            artifacts,
            embedder,
            shard_size=10,
            limit=30,
            works_source=works_stream(100),
            seed_works=[fake_work(9_001)],
        )


def test_exhaustive_backfill_refuses_benchmark_seed_overlay(tmp_path: Path):
    import pytest

    with pytest.raises(ValueError, match="exhaustive full-corpus"):
        run_backfill(
            tmp_path / "artifacts",
            HashingEmbedder(),
            works_source=works_stream(10),
            seed_works=[fake_work(9_000)],
        )


def test_crash_between_rename_and_marker_redoes_shard_without_loss(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    embedder = HashingEmbedder()
    run_backfill(
        artifacts, embedder, shard_size=50, limit=100, works_source=works_stream(400)
    )
    # Simulate the crash window: shard 1's files renamed but its marker never
    # written. Resume must delete the orphans and redo the shard.
    (artifacts / "shard-00001.done").unlink()
    manifest = run_backfill(
        artifacts, embedder, shard_size=50, limit=100, works_source=works_stream(400)
    )
    assert manifest["records"] == 100
    assert manifest["shards"] == 2
    import numpy as np

    ids = np.concatenate(
        [np.load(artifacts / f"shard-{index:05d}.ids.npy") for index in range(2)]
    )
    assert len(ids) == 100 and len(set(int(v) for v in ids)) == 100
    # The global truth sample must hold no duplicates after the redo.
    truth_ids = np.load(artifacts / "float32-truth-ids.npy")
    assert len(truth_ids) == len(set(int(v) for v in truth_ids))
    assert set(int(v) for v in truth_ids) <= set(int(v) for v in ids)


def test_resume_refuses_changed_snapshot(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    embedder = HashingEmbedder()
    run_backfill(
        artifacts,
        embedder,
        shard_size=50,
        limit=50,
        works_source=works_stream(400),
        snapshot_meta={"manifest_etag": "etag-A"},
    )
    import pytest

    with pytest.raises(RuntimeError, match="Snapshot changed"):
        run_backfill(
            artifacts,
            embedder,
            shard_size=50,
            limit=100,
            works_source=works_stream(400),
            snapshot_meta={"manifest_etag": "etag-B"},
        )


def test_snapshot_change_during_embedding_leaves_shard_uncommitted(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    embedder = HashingEmbedder()
    observations = iter(
        [
            {"manifest_etag": "etag-A"},
            {"manifest_etag": "etag-B"},
        ]
    )
    import pytest

    with pytest.raises(RuntimeError, match="changed while the backfill was running"):
        run_backfill(
            artifacts,
            embedder,
            shard_size=50,
            limit=50,
            works_source=works_stream(100),
            snapshot_meta={"manifest_etag": "etag-A"},
            snapshot_probe=lambda: next(observations),
        )
    assert not list(artifacts.glob("shard-*.done"))


def test_assemble_refuses_partial_stage(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    embedder = HashingEmbedder()
    run_backfill(
        artifacts, embedder, shard_size=100, limit=320, works_source=works_stream(400)
    )
    import pytest

    big_stage = Stage("full-ish", 10_000, 8, 2, candidate_count=320)
    with pytest.raises(ValueError, match="truncated backfill"):
        assemble_generation(
            artifacts,
            stage=big_stage,
            output=tmp_path / "gen-refused",
            backend="numpy",
            embedder_name=embedder.name,
        )
    assembled = assemble_generation(
        artifacts,
        stage=big_stage,
        output=tmp_path / "gen-allowed",
        backend="numpy",
        embedder_name=embedder.name,
        allow_partial=True,
    )
    assert assembled["records"] == 320


def test_full_stage_requires_source_exhaustion_even_with_override(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    embedder = HashingEmbedder()
    limited = run_backfill(
        artifacts, embedder, shard_size=50, limit=100, works_source=works_stream(200)
    )
    assert limited["source_exhausted"] is False
    import pytest

    with pytest.raises(ValueError, match="exhausted the pinned snapshot"):
        assemble_generation(
            artifacts,
            stage=Stage("full", 100, 8, 2, candidate_count=100),
            output=tmp_path / "full-refused",
            backend="numpy",
            embedder_name=embedder.name,
            allow_partial=True,
        )


def test_exhausted_source_can_publish_full_stage(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    embedder = HashingEmbedder()
    complete = run_backfill(
        artifacts, embedder, shard_size=50, works_source=works_stream(100)
    )
    assert complete["records"] == 100
    assert complete["source_exhausted"] is True
    assembled = assemble_generation(
        artifacts,
        stage=Stage("full", 100, 8, 2, candidate_count=100),
        output=tmp_path / "full-generation",
        backend="numpy",
        embedder_name=embedder.name,
    )
    assert assembled["records"] == 100


def test_compact_full_generation_is_searchable_and_checksum_manifested(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    embedder = HashingEmbedder()
    complete = run_backfill(
        artifacts,
        embedder,
        shard_size=50,
        truth_stride=7,
        works_source=works_stream(100),
    )
    assert complete["source_exhausted"] is True
    generation = tmp_path / "compact-full-generation"
    assembled = assemble_generation(
        artifacts,
        stage=Stage("full", 100, 8, 2, candidate_count=100),
        output=generation,
        backend="numpy",
        metadata_backend="compact",
        metadata_workers=2,
        metadata_block_rows=11,
        embedder_name=embedder.name,
    )
    assert assembled["records"] == 100
    assert assembled["metadata_backend"] == "compact"
    assert (generation / "generation-files.sha256").exists()
    truth_ids = np.load(generation / "float32-truth-ids.npy")
    truth_rows = np.load(generation / "float32-truth-row-ids.npy")
    ids = np.concatenate(
        [np.load(artifacts / f"shard-{index:05d}.ids.npy") for index in range(2)]
    )
    np.testing.assert_array_equal(ids[truth_rows], truth_ids)

    engine = SearchEngine(generation, embedder, selective_filter_threshold=500)
    try:
        response = engine.search("Attention Is All You Need", limit=10)
        assert response.corpus_records == 100
        assert response.results[0].title == "Attention Is All You Need"
        assert response.timings_ms is not None
        filtered = engine.search(
            "research paper",
            limit=10,
            filters=Filters(year_min=2003, year_max=2003),
        )
        assert filtered.filtered_records == 4
        assert all(result.publication_year == 2003 for result in filtered.results)
    finally:
        engine.close()


def test_resumable_assembly_preserves_checkpoints_and_freezes_config(
    tmp_path: Path, monkeypatch
):
    artifacts = tmp_path / "artifacts"
    embedder = HashingEmbedder()
    run_backfill(
        artifacts,
        embedder,
        shard_size=50,
        truth_stride=7,
        works_source=works_stream(100),
    )
    output = tmp_path / "resumable-full"
    original = assemble_module.build_compact_metadata

    def fail_after_int8(*args, **kwargs):
        raise RuntimeError("simulated metadata interruption")

    monkeypatch.setattr(assemble_module, "build_compact_metadata", fail_after_int8)
    with pytest.raises(RuntimeError, match="simulated metadata interruption"):
        assemble_generation(
            artifacts,
            stage=Stage("full", 100, 8, 2, candidate_count=100),
            output=output,
            backend="numpy",
            metadata_backend="compact",
            metadata_workers=2,
            metadata_block_rows=11,
            embedder_name=embedder.name,
            resume=True,
        )

    building = tmp_path / ".resumable-full.building"
    assert (building / "int8-layer.done").is_file()
    assert not output.exists()

    with pytest.raises(RuntimeError, match="frozen pin"):
        assemble_generation(
            artifacts,
            stage=Stage("full", 100, 8, 2, candidate_count=100),
            output=output,
            backend="numpy",
            metadata_backend="compact",
            metadata_workers=2,
            metadata_block_rows=12,
            embedder_name=embedder.name,
            resume=True,
        )

    monkeypatch.setattr(assemble_module, "build_compact_metadata", original)
    assembled = assemble_generation(
        artifacts,
        stage=Stage("full", 100, 8, 2, candidate_count=100),
        output=output,
        backend="numpy",
        metadata_backend="compact",
        metadata_workers=2,
        metadata_block_rows=11,
        embedder_name=embedder.name,
        resume=True,
    )
    assert assembled["records"] == 100
    assert assembled["resumable_assembly"] is True
    assert (output / "int8-layer.done").is_file()


def test_backfill_bounds_embedding_chunks_and_preserves_artifact_rows(
    tmp_path: Path, monkeypatch
):
    class TrackingEmbedder(HashingEmbedder):
        def __init__(self):
            super().__init__()
            self.call_sizes: list[int] = []
            self.thread_ids: set[int] = set()

        def encode_documents(self, texts):
            self.call_sizes.append(len(texts))
            self.thread_ids.add(threading.get_ident())
            return super().encode_documents(texts)

    writer_threads: set[int] = set()
    original_writer = backfill_module._write_embedded_chunk

    def track_writer(*args, **kwargs):
        writer_threads.add(threading.get_ident())
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(backfill_module, "_write_embedded_chunk", track_writer)
    artifacts = tmp_path / "artifacts"
    embedder = TrackingEmbedder()
    manifest = run_backfill(
        artifacts,
        embedder,
        shard_size=20,
        limit=35,
        truth_stride=6,
        embed_subchunk=7,
        works_source=works_stream(60),
    )

    assert manifest["embed_subchunk"] == 7
    assert embedder.call_sizes == [7, 7, 6, 7, 7, 1]
    assert max(embedder.call_sizes) == 7
    assert embedder.thread_ids == {threading.get_ident()}
    assert len(writer_threads) == 1
    assert writer_threads.isdisjoint(embedder.thread_ids)

    expected_papers = [parse_work(fake_work(index)) for index in range(20)]
    assert all(paper is not None for paper in expected_papers)
    expected_matrix = HashingEmbedder().encode_documents(
        [paper.embedding_text for paper in expected_papers if paper is not None]
    )
    expected_scales = fit_int8_scales(expected_matrix)
    np.testing.assert_array_equal(np.load(artifacts / "int8-scales.npy"), expected_scales)
    np.testing.assert_array_equal(
        np.load(artifacts / "shard-00000.int8.npy"),
        quantize_normalized(expected_matrix, expected_scales),
    )
    np.testing.assert_array_equal(
        np.load(artifacts / "shard-00000.tids.npy"),
        np.array([1, 7, 13, 19]),
    )
    np.testing.assert_array_equal(
        np.load(artifacts / "shard-00001.tids.npy"),
        np.array([21, 27, 33]),
    )
    with gzip.open(artifacts / "shard-00001.meta.jsonl.gz", "rt", encoding="utf-8") as rows:
        metadata_ids = [json.loads(row)["openalex_id"] for row in rows]
    assert metadata_ids == [f"https://openalex.org/W{index}" for index in range(21, 36)]


def test_failed_embedding_subchunk_leaves_no_published_shard_or_temp_files(tmp_path: Path):
    class FailingEmbedder(HashingEmbedder):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def encode_documents(self, texts):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated embedding failure")
            return super().encode_documents(texts)

    artifacts = tmp_path / "artifacts"
    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        run_backfill(
            artifacts,
            FailingEmbedder(),
            shard_size=20,
            limit=20,
            embed_subchunk=7,
            works_source=works_stream(40),
        )

    assert not list(artifacts.glob("shard-*"))
    assert not list(artifacts.glob(".tmp-shard-*"))
