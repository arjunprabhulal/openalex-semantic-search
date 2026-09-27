"""Unreliable citation counts and supplement records, on the production-shaped corpus."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from conftest import build_production_shaped, corpus
from openalex_semantic_search.embeddings import HashingEmbedder
from openalex_semantic_search.engine import SearchEngine, SearchResult
from openalex_semantic_search.overrides import MetadataOverrides
from openalex_semantic_search.store import Filters
from openalex_semantic_search.supplement import MAX_SUPPLEMENT_RECORDS, SupplementRecords
from openalex_semantic_search.title_prefix import TitlePrefixIndex
from openalex_semantic_search.work_index import WorkIdIndex

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CORRECTIONS = ROOT / "overrides" / "openalex-corrections.json"
SHIPPED_SUPPLEMENT = ROOT / "overrides" / "supplement-records.json"
SUSPECTS = ROOT / "benchmark" / "suspect-citation-records.json"
BERT = "W2963341956"
QUERY = "measured research paper controlled study"
BERT_TITLE = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"


def ids(response) -> list[str]:
    return [result.work_id for result in response.results]


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def supplement_record(**fields) -> dict:
    record = {
        "openalex_id": "W4000000001",
        "title": "Supplemented landmark on quantum widget calibration",
        "authors": ["Wendy Widget"],
        "publication_year": 2018,
        "cited_by_count": 12_000,
        "doi": "https://doi.org/10.1000/widget",
        "topic": "Topic special",
        "field": "Computer Science",
        "snippet": "Quantum widget calibration for landmark measurements",
        "sources": ["https://api.openalex.org/works/W4000000001"],
        "verified_at": "2026-09-24",
    }
    record.update(fields)
    return record


# --- Unreliable citation counts ---------------------------------------------------


@pytest.fixture()
def flagged(production_shaped, tmp_path):
    """(plain, corrected, id): the work leading most_cited, flagged as unreliable."""
    generation, prefixes, work_ids = production_shaped

    def serve(overrides):
        return SearchEngine(
            generation,
            HashingEmbedder(),
            title_prefixes=prefixes,
            work_ids=work_ids,
            overrides=overrides,
        )

    plain = serve(None)
    leader = plain.search(QUERY, limit=20, sort="most_cited").results[0]
    path = write_json(
        tmp_path / "overrides.json",
        {"format_version": 1, "entries": {leader.work_id: {"cited_by_count_unreliable": True}}},
    )
    corrected = serve(MetadataOverrides.load(path))
    yield plain, corrected, leader
    plain.close()
    corrected.close()


def test_unreliable_count_leaves_most_cited_and_sets_the_flag(flagged):
    plain, corrected, leader = flagged
    FLAGGED = leader.work_id
    assert leader.cited_by_count > 1_000
    after = corrected.search(QUERY, limit=50, sort="most_cited")
    assert ids(after)[0] != FLAGGED
    assert FLAGGED in ids(after)
    hit = next(r for r in after.results if r.work_id == FLAGGED)
    # The card keeps the published number and says it is not to be trusted.
    assert hit.cited_by_count == leader.cited_by_count
    assert hit.citation_count_unreliable is True
    assert "cited_by_count_unreliable" in hit.overridden_fields
    assert not any(r.citation_count_unreliable for r in after.results if r.work_id != FLAGGED)


def test_unreliable_count_lowers_relevance_score_and_fails_min_citations(flagged):
    plain, corrected, leader = flagged
    FLAGGED = leader.work_id

    def score(engine):
        return next(
            r.ranking_score for r in engine.search(QUERY, limit=50).results if r.work_id == FLAGGED
        )

    assert score(corrected) < score(plain)
    filtered = corrected.search(QUERY, limit=20, filters=Filters(min_citations=1_000))
    assert FLAGGED not in ids(filtered)
    assert FLAGGED in ids(plain.search(QUERY, limit=20, filters=Filters(min_citations=1_000)))
    # The exact title still finds the paper the text belongs to.
    assert ids(corrected.search(leader.title, limit=5))[0] == FLAGGED


def test_flag_must_be_boolean(tmp_path):
    path = write_json(
        tmp_path / "overrides.json",
        {"format_version": 1, "entries": {"W1": {"cited_by_count_unreliable": "yes"}}},
    )
    with pytest.raises(ValueError, match="true or false"):
        MetadataOverrides.load(path)


def test_shipped_corrections_flag_every_discovered_suspect():
    overrides = MetadataOverrides.load(SHIPPED_CORRECTIONS)
    suspects = json.loads(SUSPECTS.read_text(encoding="utf-8"))
    flagged = {w["work_id"] for w in suspects["works"] if w["flag_citations"]}
    assert {"W2896457183", "W4385245566"} <= flagged
    for identifier in flagged:
        entry = overrides.entries[identifier]
        # Only the flag: the title and abstract are genuine for the LIPIcs paper.
        assert entry == {"cited_by_count_unreliable": True}
    for work in suspects["works"]:
        assert work["doi"].startswith("https://doi.org/10.4230/lipics")
        assert work["cited_by_count"] > 1_000


# --- Supplement records ----------------------------------------------------------


@pytest.fixture(scope="module")
def without_bert(tmp_path_factory):
    """The production-shaped corpus minus BERT, as the full generation is."""
    root = tmp_path_factory.mktemp("without-bert")
    papers = [paper for paper in corpus() if not paper.openalex_id.endswith(BERT)]
    generation = build_production_shaped(root, papers)
    return (
        generation,
        TitlePrefixIndex.load(root / "prefixes"),
        WorkIdIndex.load(root / "work-ids"),
    )


def serve(shaped, supplement: SupplementRecords | None, embedder=None) -> SearchEngine:
    generation, prefixes, work_ids = shaped
    return SearchEngine(
        generation,
        embedder or HashingEmbedder(),
        title_prefixes=prefixes,
        work_ids=work_ids,
        supplement=supplement,
    )


@pytest.fixture()
def bert_engine(without_bert):
    served = serve(without_bert, SupplementRecords.load(SHIPPED_SUPPLEMENT))
    yield served
    served.close()


def test_shipped_supplement_is_verified_and_bounded():
    supplement = SupplementRecords.load(SHIPPED_SUPPLEMENT)
    assert 1 <= len(supplement) <= MAX_SUPPLEMENT_RECORDS
    for record in supplement.records:
        assert record["sources"] and record["verified_at"]
    bert = supplement.records[0]
    assert bert["openalex_id"] == f"https://openalex.org/{BERT}"
    assert bert["title"] == BERT_TITLE
    assert bert["doi"] == "https://doi.org/10.18653/v1/n19-1423"
    assert bert["publication_year"] == 2019
    assert bert["authors"][0] == "Jacob Devlin"


def test_missing_landmark_is_absent_without_the_supplement(without_bert):
    served = serve(without_bert, None)
    try:
        assert BERT not in ids(served.search(BERT_TITLE, limit=20))
        assert served.fetch_work(BERT) is None
    finally:
        served.close()


@pytest.mark.parametrize(
    "typed",
    [
        BERT_TITLE,
        "BERT: Pre‐training of Deep Bidirectional Transformers for Language Understanding",
        "bert pretraining of deep bidirectional transformers for language understanding",
    ],
)
def test_supplement_exact_title_is_first(bert_engine, typed):
    response = bert_engine.search(typed, limit=20)
    first = response.results[0]
    assert first.work_id == BERT
    assert first.supplement is True and first.title_match is True
    assert first.doi == "https://doi.org/10.18653/v1/n19-1423"
    assert ids(response).count(BERT) == 1


def test_supplement_joins_semantic_results(without_bert, tmp_path):
    path = write_json(
        tmp_path / "supplement.json", {"format_version": 1, "records": [supplement_record()]}
    )
    served = serve(without_bert, SupplementRecords.load(path))
    try:
        response = served.search("quantum widget calibration measurements", limit=20)
        assert response.results[0].work_id == "W4000000001"
        assert response.results[0].title_match is False
        unrelated = served.search("Measured research paper 17", limit=20)
        assert "W4000000001" not in ids(unrelated)
    finally:
        served.close()


def test_title_only_supplement_joins_only_as_close_as_the_pool(bert_engine):
    # Sixty decoys carry every typed word in their abstracts and fill the
    # 50-row pool, so a title-only record would not have been retrieved.
    decoyed = bert_engine.search("deep bidirectional transformers language understanding", limit=20)
    assert BERT not in ids(decoyed)
    assert BERT in ids(bert_engine.search("BERT bidirectional transformers", limit=20))


def test_supplement_obeys_filters_and_sorts(bert_engine):
    query = BERT_TITLE
    assert ids(bert_engine.search(query, limit=20, sort="most_cited"))[0] == BERT
    assert BERT in ids(bert_engine.search(BERT_TITLE, limit=20, filters=Filters(year_min=2019)))
    assert BERT not in ids(bert_engine.search(BERT_TITLE, limit=20, filters=Filters(year_max=2018)))
    assert BERT not in ids(
        bert_engine.search(BERT_TITLE, limit=20, filters=Filters(field="Medicine"))
    )
    newest = bert_engine.search(query, limit=20, sort="newest")
    years = [r.publication_year for r in newest.results]
    assert BERT in ids(newest) and years == sorted(years, reverse=True)


def test_supplement_work_id_lookup(bert_engine):
    work = bert_engine.fetch_work(f"https://openalex.org/{BERT}")
    assert set(work) == set(SearchResult.__slots__)
    assert work["work_id"] == BERT and work["supplement"] is True
    assert work["view_url"] == "https://doi.org/10.18653/v1/n19-1423"
    # Generation lookups are unchanged.
    assert bert_engine.fetch_work("W2194775991")["supplement"] is False


def test_supplement_lookup_works_without_a_work_id_index(without_bert):
    generation, prefixes, _ = without_bert
    served = SearchEngine(
        generation,
        HashingEmbedder(),
        title_prefixes=prefixes,
        supplement=SupplementRecords.load(SHIPPED_SUPPLEMENT),
    )
    try:
        assert served.fetch_work(BERT)["work_id"] == BERT
        with pytest.raises(RuntimeError):
            served.fetch_work("W2194775991")
    finally:
        served.close()


def test_supplement_record_already_in_the_generation_is_skipped(without_bert, tmp_path):
    path = write_json(
        tmp_path / "supplement.json",
        {
            "format_version": 1,
            "records": [
                supplement_record(
                    openalex_id="W2194775991", title="A different title the card must not show"
                )
            ],
        },
    )
    served = serve(without_bert, SupplementRecords.load(path))
    try:
        assert served.supplement_count == 0
        assert served.fetch_work("W2194775991")["title"] == "Deep Residual Learning for Image Recognition"
    finally:
        served.close()


def test_supplement_replaces_a_same_work_copy_by_title(without_bert, tmp_path):
    path = write_json(
        tmp_path / "supplement.json",
        {
            "format_version": 1,
            "records": [
                supplement_record(
                    openalex_id="W4000000002",
                    title="Deep Residual Learning for Image Recognition",
                    authors=["Kaiming He"],
                    publication_year=2016,
                    cited_by_count=1_000,
                    doi="https://doi.org/10.1109/cvpr.2016.90",
                    snippet="Residual networks make deep image models easier to optimize",
                )
            ],
        },
    )
    served = serve(without_bert, SupplementRecords.load(path))
    try:
        response = served.search("Deep Residual Learning for Image Recognition", limit=20)
        he = [r.work_id for r in response.results if r.authors == ["Kaiming He"]]
        assert he == ["W4000000002"]
        # A different paper that shares the title stays.
        assert "WRESNETOTHER" in ids(response)
    finally:
        served.close()


def test_supplement_file_problems_degrade_to_serving_without(tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    unsourced = write_json(
        tmp_path / "unsourced.json",
        {"format_version": 1, "records": [supplement_record(sources=[])]},
    )
    unverified = write_json(
        tmp_path / "unverified.json",
        {"format_version": 1, "records": [supplement_record(verified_at="")]},
    )
    oversized = write_json(
        tmp_path / "oversized.json",
        {
            "format_version": 1,
            "records": [
                supplement_record(openalex_id=f"W{5_000_000 + i}")
                for i in range(MAX_SUPPLEMENT_RECORDS + 1)
            ],
        },
    )
    wrong_version = write_json(tmp_path / "v2.json", {"format_version": 2, "records": []})
    for path in (tmp_path / "missing.json", bad_json, unsourced, unverified, oversized, wrong_version):
        with pytest.raises((OSError, ValueError)):
            SupplementRecords.load(path)
        assert len(SupplementRecords.load_or_empty(path)) == 0
    assert "serving without them" in caplog.text


class FailingEmbedder(HashingEmbedder):
    def encode_documents(self, texts):
        raise RuntimeError("model unavailable")


def test_supplement_embedding_failure_serves_without(without_bert, caplog):
    caplog.set_level(logging.ERROR)
    served = serve(
        without_bert, SupplementRecords.load(SHIPPED_SUPPLEMENT), embedder=FailingEmbedder()
    )
    try:
        assert served.supplement_count == 0
        assert BERT not in ids(served.search(BERT_TITLE, limit=20))
    finally:
        served.close()
    assert "serving without them" in caplog.text


def test_api_serves_supplement_and_reports_it(without_bert, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from openalex_semantic_search import api

    generation, _, _ = without_bert
    monkeypatch.setattr(api, "SentenceTransformerEmbedder", lambda *a, **k: HashingEmbedder())
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    for supplement, expected in ((SHIPPED_SUPPLEMENT, 1), (bad, 0)):
        app = api.create_app(
            generation=generation,
            api_keys=frozenset({"k"}),
            work_ids=generation.parent / "work-ids",
            supplement=supplement,
        )
        with TestClient(app) as client:
            assert client.get("/healthz/details", headers={"X-API-Key": "k"}).json()["supplement_records"] == expected
            response = client.post(
                "/search", json={"query": "lookup", "work_id": BERT}, headers={"X-API-Key": "k"}
            )
            assert response.status_code == (200 if expected else 404)
