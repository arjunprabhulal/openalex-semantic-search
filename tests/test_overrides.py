import json
from pathlib import Path

import pytest

from openalex_semantic_search.overrides import (
    OVERRIDE_FORMAT_VERSION,
    MetadataOverrides,
    work_id,
)

REPO_OVERRIDES = Path(__file__).resolve().parents[1] / "overrides" / "openalex-corrections.json"


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _record(**extra) -> dict:
    base = {
        "row_id": 7,
        "openalex_id": "https://openalex.org/W2626778328",
        "title": "Attention Is All You Need",
        "publication_year": 2025,
        "doi": "https://doi.org/10.65215/2q58a426",
        "venue": "",
        "oa_url": "https://langtaosha.org.cn/download/10/108",
        "cited_by_count": 6569,
    }
    base.update(extra)
    return base


def test_work_id_extracts_bare_identifier():
    assert work_id("https://openalex.org/W2626778328") == "W2626778328"
    assert work_id("W2626778328") == "W2626778328"
    assert work_id("") == ""


def test_empty_overrides_return_the_record_unchanged():
    overrides = MetadataOverrides.empty()
    record = _record()
    assert overrides.apply(record) is record


def test_apply_replaces_only_listed_fields_and_records_the_trail(tmp_path):
    overrides = MetadataOverrides.load(
        _write(
            tmp_path,
            {
                "format_version": OVERRIDE_FORMAT_VERSION,
                "entries": {
                    "W2626778328": {
                        "publication_year": 2017,
                        "doi": "https://doi.org/10.48550/arXiv.1706.03762",
                    }
                },
            },
        )
    )
    patched = overrides.apply(_record())
    assert patched["publication_year"] == 2017
    assert patched["doi"] == "https://doi.org/10.48550/arXiv.1706.03762"
    # Untouched fields survive, and the caller can see what was corrected.
    assert patched["cited_by_count"] == 6569
    assert patched["overridden_fields"] == ["doi", "publication_year"]


def test_apply_does_not_mutate_the_cached_source_record(tmp_path):
    overrides = MetadataOverrides.load(
        _write(
            tmp_path,
            {
                "format_version": OVERRIDE_FORMAT_VERSION,
                "entries": {"W2626778328": {"publication_year": 2017}},
            },
        )
    )
    record = _record()
    overrides.apply(record)
    # Compact blocks are cached in memory; patching must never write back.
    assert record["publication_year"] == 2025
    assert "overridden_fields" not in record


def test_unlisted_work_is_untouched(tmp_path):
    overrides = MetadataOverrides.load(
        _write(
            tmp_path,
            {
                "format_version": OVERRIDE_FORMAT_VERSION,
                "entries": {"W999999999": {"publication_year": 1999}},
            },
        )
    )
    record = _record()
    assert overrides.apply(record) is record


def test_identity_and_ranking_fields_cannot_be_overridden(tmp_path):
    overrides = MetadataOverrides.load(
        _write(
            tmp_path,
            {
                "format_version": OVERRIDE_FORMAT_VERSION,
                "entries": {
                    "W2626778328": {
                        "openalex_id": "https://openalex.org/W1",
                        "row_id": 0,
                        "normalized_title": "hijacked",
                        "publication_year": 2017,
                    }
                },
            },
        )
    )
    patched = overrides.apply(_record())
    assert patched["openalex_id"] == "https://openalex.org/W2626778328"
    assert patched["row_id"] == 7
    assert patched["overridden_fields"] == ["publication_year"]


def test_missing_file_and_bad_payloads_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        MetadataOverrides.load(tmp_path / "absent.json")
    with pytest.raises(ValueError, match="format_version"):
        MetadataOverrides.load(_write(tmp_path, {"format_version": 99, "entries": {}}))
    with pytest.raises(ValueError, match="entries"):
        MetadataOverrides.load(_write(tmp_path, {"format_version": OVERRIDE_FORMAT_VERSION}))
    with pytest.raises(ValueError, match="not an OpenAlex work id"):
        MetadataOverrides.load(
            _write(
                tmp_path,
                {"format_version": OVERRIDE_FORMAT_VERSION, "entries": {"nope": {"venue": "x"}}},
            )
        )
    with pytest.raises(ValueError, match="no overridable field"):
        MetadataOverrides.load(
            _write(
                tmp_path,
                {
                    "format_version": OVERRIDE_FORMAT_VERSION,
                    "entries": {"W1": {"note": "documentation only"}},
                },
            )
        )


def test_shipped_correction_file_loads_and_repairs_the_hijacked_record():
    overrides = MetadataOverrides.load(REPO_OVERRIDES)
    assert len(overrides) >= 1
    patched = overrides.apply(_record())
    assert patched["publication_year"] == 2017
    assert patched["doi"] == "https://doi.org/10.48550/arXiv.1706.03762"
    assert patched["oa_url"] == "https://arxiv.org/pdf/1706.03762"
    assert patched["landing_url"] == "https://arxiv.org/abs/1706.03762"
    assert patched["venue"] == "arXiv (Cornell University)"
    # No verified replacement exists for the citation count, so it is left alone.
    assert patched["cited_by_count"] == 6569
    assert "cited_by_count" not in patched["overridden_fields"]
