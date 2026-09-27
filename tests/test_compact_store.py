from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path

import numpy as np

from openalex_semantic_search.compact_store import (
    CompactMetadataStore,
    _compact_shard,
    _load_completed_part,
    build_compact_metadata,
)
from openalex_semantic_search.store import Filters


def _record(index: int, *, year: int | None = None, title: str | None = None) -> dict:
    return {
        "openalex_id": f"https://openalex.org/W{index + 1}",
        "title": title or f"Paper {index}",
        "snippet": f"Abstract for paper {index}",
        "authors": [f"Researcher {index}"],
        "publication_year": year if year is not None else 2000 + index,
        "doi": f"https://doi.org/10.1000/{index}",
        "cited_by_count": index * 10,
        "topic": "Transformers" if index % 2 == 0 else "Databases",
        "field": "Computer Science",
        "is_oa": index % 2 == 0,
        "oa_url": f"https://example.test/{index}" if index % 2 == 0 else "",
        "work_type": "article",
        "venue": "Test Journal",
        "publication_date": f"{2000 + index}-01-01",
        "landing_url": f"https://openalex.org/W{index + 1}",
    }


def _build_store(tmp_path: Path) -> Path:
    generation = tmp_path / "generation"
    generation.mkdir()
    records = [
        _record(0, year=1981, title="Attention Is All You Need"),
        *(_record(index) for index in range(1, 7)),
    ]
    shards: list[Path] = []
    start = 0
    for shard_index, count in enumerate((4, 3)):
        vector_path = generation / f"shard-{shard_index:05d}.int8.npy"
        np.save(vector_path, np.zeros((count, 2), dtype=np.int8))
        metadata_path = generation / f"shard-{shard_index:05d}.meta.jsonl.gz"
        with gzip.open(metadata_path, "wt", encoding="utf-8") as lines:
            for record in records[start : start + count]:
                lines.write(json.dumps(record) + "\n")
        shards.append(vector_path)
        start += count
    build_compact_metadata(
        shards,
        generation,
        shard_rows=(4, 3),
        total=7,
        workers=2,
        block_rows=3,
    )
    return generation


def test_compact_store_preserves_rows_across_shard_and_block_boundaries(tmp_path):
    generation = _build_store(tmp_path)
    store = CompactMetadataStore(generation)
    try:
        rows = store.fetch([6, 0, 4, 3])
        assert [rows[index]["openalex_id"] for index in (0, 3, 4, 6)] == [
            "https://openalex.org/W1",
            "https://openalex.org/W4",
            "https://openalex.org/W5",
            "https://openalex.org/W7",
        ]
        assert rows[0]["authors"] == ["Researcher 0"]
        assert rows[6]["venue"] == "Test Journal"
    finally:
        store.close()


def test_compact_store_exact_title_filters_and_counts(tmp_path):
    generation = _build_store(tmp_path)
    store = CompactMetadataStore(generation)
    try:
        assert store.title_search("attention is all you need!", Filters()) == [0]
        assert store.title_search("attention is all you", Filters()) == []
        assert store.count(Filters()) == 7
        assert store.count(Filters(open_access_only=True)) == 4
        assert store.count(Filters(year_min=1981, year_max=1981)) == 1
        assert store.eligible_ids(Filters(year_min=1981, year_max=1981)) == [0]
        assert store.filter_candidate_ids(
            list(range(7)),
            Filters(open_access_only=True, min_citations=20),
        ) == [2, 4, 6]
        assert store.filter_candidate_ids(
            list(range(7)),
            Filters(topic="Transformers", field="Computer Science"),
        ) == [0, 2, 4, 6]
        assert store.rarest_year() == (1981, 1)
    finally:
        store.close()


def test_compact_store_block_cache_is_thread_safe(tmp_path):
    generation = _build_store(tmp_path)
    store = CompactMetadataStore(generation)
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            outputs = list(executor.map(lambda _: store.fetch(range(7)), range(100)))
        assert all(len(output) == 7 for output in outputs)
        assert all(output[0]["title"] == "Attention Is All You Need" for output in outputs)
    finally:
        store.close()


def test_compact_part_marker_detects_corruption(tmp_path):
    parts = tmp_path / "parts"
    parts.mkdir()
    metadata = tmp_path / "shard.meta.jsonl.gz"
    with gzip.open(metadata, "wt", encoding="utf-8") as lines:
        for index in range(3):
            lines.write(json.dumps(_record(index)) + "\n")

    _compact_shard(0, 0, 3, str(metadata), str(parts), 2)
    output = parts / "shard-00000"
    assert _load_completed_part(
        output, shard_index=0, start_row=0, expected_records=3
    ) is not None

    blocks = output / "metadata.blocks"
    blocks.write_bytes(blocks.read_bytes() + b"corrupt")
    assert _load_completed_part(
        output, shard_index=0, start_row=0, expected_records=3
    ) is None
