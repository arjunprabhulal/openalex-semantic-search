import json
from pathlib import Path

from openalex_semantic_search.progress import read_index_progress


def write_marker(path: Path, shard: int, *, records: int, records_done: int, file_index: int):
    (path / f"shard-{shard:05d}.done").write_text(
        json.dumps(
            {
                "records": records,
                "records_done": records_done,
                "file_index": file_index,
            }
        ),
        encoding="utf-8",
    )


def test_progress_is_unavailable_without_replica(tmp_path):
    progress = read_index_progress(tmp_path / "missing", indexed_records=10_000)

    assert progress["available"] is False
    assert progress["indexed_records"] == 10_000


def test_progress_counts_only_contiguous_consistent_markers(tmp_path):
    write_marker(tmp_path, 0, records=50_000_000, records_done=50_000_000, file_index=252)
    write_marker(tmp_path, 1, records=1_000_000, records_done=51_000_000, file_index=258)
    # A later marker cannot jump a missing shard and inflate durable progress.
    write_marker(tmp_path, 3, records=1_000_000, records_done=53_000_000, file_index=266)
    (tmp_path / "snapshot-pin.json").write_text(
        json.dumps({"file_count": 2446, "manifest_date": "2026-06-26"}),
        encoding="utf-8",
    )

    progress = read_index_progress(tmp_path, indexed_records=10_000)

    assert progress["available"] is True
    assert progress["state"] == "building"
    assert progress["verified_embedding_records"] == 51_000_000
    assert progress["committed_shards"] == 2
    assert progress["source_files_reached"] == 259
    assert progress["source_files_total"] == 2446
    assert progress["source_progress_percent"] == 10.59
    assert progress["last_checkpoint_at"].endswith("+00:00")


def test_progress_rejects_inconsistent_records_done(tmp_path):
    write_marker(tmp_path, 0, records=1_000_000, records_done=1_000_000, file_index=4)
    write_marker(tmp_path, 1, records=1_000_000, records_done=9_000_000, file_index=5)

    progress = read_index_progress(tmp_path, indexed_records=320)

    assert progress["verified_embedding_records"] == 1_000_000
    assert progress["committed_shards"] == 1


def test_progress_is_complete_only_with_exhausted_manifest(tmp_path):
    write_marker(tmp_path, 0, records=1_000_000, records_done=1_000_000, file_index=9)
    (tmp_path / "backfill-manifest.json").write_text(
        json.dumps({"records": 1_000_000, "shards": 1, "source_exhausted": True}),
        encoding="utf-8",
    )

    progress = read_index_progress(tmp_path, indexed_records=1_000_000)

    assert progress["state"] == "complete"
