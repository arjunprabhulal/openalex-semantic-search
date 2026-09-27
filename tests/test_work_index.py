import json
from pathlib import Path

import numpy as np
import pytest

from openalex_semantic_search.compact_store import stable_text_hash
from openalex_semantic_search.work_index import (
    WORK_INDEX_FORMAT_VERSION,
    WorkIdIndex,
    normalize_work_id,
)


def test_normalize_accepts_every_spelling_a_caller_might_hold():
    for value in (
        "W2626778328",
        "w2626778328",
        "https://openalex.org/W2626778328",
        "https://openalex.org/W2626778328/",
        "https://api.openalex.org/works/W2626778328?select=id",
        "https://openalex.org/W2626778328#frag",
        "  W2626778328  ",
    ):
        assert normalize_work_id(value) == "W2626778328", value


def test_normalize_rejects_anything_that_is_not_a_work_id():
    # A DOI or an author id must not be coerced into a work lookup.
    for value in ("", "10.1234/abc", "A5023888391", "S4306402567", "W", "Wabc", "https://openalex.org/"):
        assert normalize_work_id(value) == ""


def _index(tmp_path: Path, ids: list[str]) -> WorkIdIndex:
    hashes = np.asarray([stable_text_hash(i) for i in ids], dtype=np.uint64)
    rows = np.arange(len(ids), dtype=np.uint32)
    order = np.argsort(hashes, kind="stable")
    directory = tmp_path / "ids"
    directory.mkdir()
    np.save(directory / "work-id-hashes.npy", hashes[order])
    np.save(directory / "work-id-rows.npy", rows[order])
    (directory / "work-id-index.json").write_text(
        json.dumps({"format_version": WORK_INDEX_FORMAT_VERSION, "ids": len(ids)})
    )
    loaded = WorkIdIndex.load(directory)
    assert loaded is not None
    return loaded


def test_lookup_returns_the_row_for_a_known_id(tmp_path):
    index = _index(tmp_path, ["W1", "W2626778328", "W3"])
    assert index.candidate_rows("W2626778328", stable_text_hash) == [1]
    assert index.candidate_rows("https://openalex.org/W1", stable_text_hash) == [0]


def test_lookup_returns_nothing_for_an_unknown_or_malformed_id(tmp_path):
    index = _index(tmp_path, ["W1", "W2"])
    assert index.candidate_rows("W999999", stable_text_hash) == []
    assert index.candidate_rows("10.1234/not-a-work", stable_text_hash) == []
    assert index.candidate_rows("", stable_text_hash) == []


def test_load_rejects_a_missing_directory_or_wrong_format(tmp_path):
    assert WorkIdIndex.load(None) is None
    with pytest.raises(FileNotFoundError):
        WorkIdIndex.load(tmp_path / "absent")
    index_dir = tmp_path / "ids"
    _index(tmp_path, ["W1"])
    manifest = json.loads((index_dir / "work-id-index.json").read_text())
    manifest["format_version"] = 99
    (index_dir / "work-id-index.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="format_version"):
        WorkIdIndex.load(index_dir)
