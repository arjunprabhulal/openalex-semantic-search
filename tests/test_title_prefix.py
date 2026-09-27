import json
from pathlib import Path

import numpy as np
import pytest

from openalex_semantic_search.compact_store import stable_text_hash
from openalex_semantic_search.title_prefix import (
    MAX_PREFIX_WORDS,
    MIN_PREFIX_WORDS,
    TITLE_PREFIX_FORMAT_VERSION,
    TitlePrefixIndex,
    build_title_prefix_index,
    prefix_hashes,
)

TITLES = [
    "Attention Is All You Need",
    "Deep Residual Learning for Image Recognition",
    "Attention Is All You Have",
    "Short title",
]


class _FakeStore:
    """Minimal stand-in for CompactMetadataStore's read surface."""

    def __init__(self, generation: Path):
        self.record_count = len(TITLES)
        self.closed = False

    def fetch(self, ids):
        return {
            row: {"openalex_id": f"https://openalex.org/W{row}", "title": TITLES[row]}
            for row in ids
            if 0 <= row < len(TITLES)
        }

    def close(self):
        self.closed = True


def _build(tmp_path: Path) -> Path:
    out = tmp_path / "prefixes"
    build_title_prefix_index(
        tmp_path / "generation",
        out,
        hasher=stable_text_hash,
        store_factory=_FakeStore,
        batch_rows=2,
    )
    return out


def test_prefix_hashes_skips_short_titles_and_the_full_title():
    # A title at or below the minimum has no prefix worth indexing.
    assert prefix_hashes("Short title", stable_text_hash) == []
    assert prefix_hashes("One two three", stable_text_hash) == []

    values = prefix_hashes("Attention Is All You Need", stable_text_hash)
    # 3 and 4 words only: the 5-word form is the title, which the exact index
    # already answers.
    assert len(values) == 2
    assert stable_text_hash("attention is all you need") not in values
    assert stable_text_hash("attention is all you") in values
    assert stable_text_hash("attention is all") in values


def test_prefix_count_is_bounded_for_a_long_title():
    long_title = " ".join(f"word{i}" for i in range(40))
    values = prefix_hashes(long_title, stable_text_hash)
    assert len(values) == MAX_PREFIX_WORDS - MIN_PREFIX_WORDS + 1


def test_build_writes_a_sorted_index_and_manifest(tmp_path):
    out = _build(tmp_path)
    manifest = json.loads((out / "title-prefix.json").read_text())
    assert manifest["format_version"] == TITLE_PREFIX_FORMAT_VERSION
    assert manifest["records"] == len(TITLES)
    hashes = np.load(out / "title-prefix-hashes.npy")
    rows = np.load(out / "title-prefix-rows.npy")
    assert hashes.shape == rows.shape
    # Lookup is a binary search, so the hashes must be sorted.
    assert np.all(np.diff(hashes) >= 0)


def test_lookup_finds_the_paper_from_a_partial_title(tmp_path):
    index = TitlePrefixIndex.load(_build(tmp_path))
    assert index is not None
    # The query that motivated this: not an exact title, so the exact index
    # misses it entirely.
    rows = index.lookup("Attention Is All You", stable_text_hash)
    assert 0 in rows  # "Attention Is All You Need"
    assert 2 in rows  # "Attention Is All You Have" shares the prefix
    assert 1 not in rows


def test_lookup_is_case_and_punctuation_insensitive(tmp_path):
    index = TitlePrefixIndex.load(_build(tmp_path))
    assert index.lookup("attention, is all you!", stable_text_hash) == index.lookup(
        "Attention Is All You", stable_text_hash
    )


def test_lookup_rejects_queries_outside_the_indexed_prefix_range(tmp_path):
    index = TitlePrefixIndex.load(_build(tmp_path))
    # Too short to be selective; semantic retrieval covers this case.
    assert index.lookup("attention is", stable_text_hash) == []
    assert index.lookup("", stable_text_hash) == []
    # A full title is answered by the exact index, not this one.
    assert index.lookup("Attention Is All You Need", stable_text_hash) == []


def test_load_rejects_a_missing_or_mismatched_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        TitlePrefixIndex.load(tmp_path / "absent")
    assert TitlePrefixIndex.load(None) is None

    out = _build(tmp_path)
    manifest = json.loads((out / "title-prefix.json").read_text())
    manifest["format_version"] = 99
    (out / "title-prefix.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="format_version"):
        TitlePrefixIndex.load(out)


def test_engine_scores_prefix_rows_the_candidate_index_never_returned():
    """A prefix hit outside the semantic candidate set must still be scored.

    At 315M the candidate generator returns 1,000 rows out of hundreds of
    millions, so a row supplied only by the prefix index is the normal case,
    not the edge case. Scoring only the exact-title rows left those rows
    without an entry and raised KeyError on the first real query.
    """
    import numpy as np

    from openalex_semantic_search.engine import SearchEngine

    engine = SearchEngine.__new__(SearchEngine)
    semantic_ids = [10, 11]
    prefix_ids = [99]           # found by prefix lookup only
    title_ids: list[int] = []   # exact-title lookup missed
    score_by_id = {row: 0.5 for row in semantic_ids}
    all_ids = list(dict.fromkeys([*title_ids, *prefix_ids, *semantic_ids]))

    missing = [row_id for row_id in all_ids if row_id not in score_by_id]
    assert missing == [99], "the prefix-only row must be recognised as unscored"

    score_by_id.update({row: 0.4 for row in missing})
    # Every row that reaches ranking now has a score.
    assert all(row_id in score_by_id for row_id in all_ids)
