"""POST /search through the real FastAPI app and a production-shaped engine."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from openalex_semantic_search.api import create_app
from openalex_semantic_search.embeddings import HashingEmbedder
from openalex_semantic_search.engine import SearchEngine, SearchResult

KEY = "test-application-key"


@pytest.fixture()
def client(production_shaped):
    generation, prefixes, work_ids = production_shaped
    engine = SearchEngine(
        generation, HashingEmbedder(), title_prefixes=prefixes, work_ids=work_ids
    )
    app = create_app(api_keys=frozenset({KEY}), engine=engine)
    with TestClient(app) as test_client:
        yield test_client
    engine.close()


def post(client, body, key=KEY):
    headers = {"X-API-Key": key} if key is not None else {}
    return client.post("/search", json=body, headers=headers)


def test_the_application_key_is_required(client):
    assert post(client, {"query": "research paper"}, key=None).status_code == 401
    assert post(client, {"query": "research paper"}, key="wrong").status_code == 401
    assert post(client, {"query": "research paper"}).status_code == 200


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"query": ""},
        {"query": "x", "limit": 0},
        {"query": "x", "limit": 51},
        {"query": "x", "offset": -1},
        {"query": "x", "offset": 10_001},
        {"query": "x", "sort": "banana"},
        {"query": "x", "year_min": 3000},
    ],
)
def test_invalid_requests_are_rejected(client, body):
    assert post(client, body).status_code == 422


def test_an_inverted_year_range_is_a_422(client):
    response = post(client, {"query": "x", "year_min": 2020, "year_max": 2010})
    assert response.status_code == 422
    assert "year_min" in response.json()["detail"]


@pytest.mark.parametrize("sort", ["relevance", "most_cited", "newest", "oldest"])
def test_every_sort_value_is_served(client, sort):
    body = post(client, {"query": "research paper", "limit": 10, "sort": sort}).json()
    assert body["sort"] == sort
    assert body["results"] and len(body["results"]) <= 10


def test_pagination_fields(client):
    first = post(client, {"query": "research paper", "limit": 20}).json()
    for field in (
        "offset",
        "page_size",
        "total_matches",
        "total_matches_capped",
        "has_more",
        "next_offset",
        "candidates",
        "filtered_records",
        "corpus_records",
        "generation_id",
    ):
        assert field in first, field
    assert first["offset"] == 0 and first["page_size"] == 20
    assert first["has_more"] is True and first["next_offset"] == 20
    assert first["total_matches_capped"] is True

    last = post(client, {"query": "research paper", "limit": 20, "offset": 10_000}).json()
    assert last["results"] == []
    assert last["has_more"] is False and last["next_offset"] is None
    assert last["offset"] == last["total_matches"] == first["total_matches"]


def test_work_id_lookup_shape(client):
    body = post(client, {"query": "lookup", "work_id": "W2194775991"}).json()
    assert body["total_matches"] == 1 and body["has_more"] is False
    assert set(body["results"][0]) == set(SearchResult.__slots__)
    assert body["results"][0]["work_id"] == "W2194775991"
    assert post(client, {"query": "lookup", "work_id": "W1"}).status_code == 404


def test_work_id_lookup_without_an_index_is_a_501(production_shaped):
    generation, _, _ = production_shaped
    engine = SearchEngine(generation, HashingEmbedder())
    app = create_app(api_keys=frozenset({KEY}), engine=engine)
    try:
        with TestClient(app) as test_client:
            response = post(test_client, {"query": "lookup", "work_id": "W2194775991"})
            assert response.status_code == 501
    finally:
        engine.close()


def test_health_details_report_the_effective_tuning(client):
    body = client.get("/healthz/details", headers={"X-API-Key": KEY}).json()
    assert body["search_tuning"]["candidate_count"] == 50
    assert body["search_tuning"]["selective_filter_threshold"] == 10_000
    assert body["embedder_matches_generation"] is True
