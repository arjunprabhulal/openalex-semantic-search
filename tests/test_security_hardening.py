"""Filter-scan budget, cache caps, request bounds and the unauthenticated surface."""

from __future__ import annotations

from collections import OrderedDict
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from openalex_semantic_search import compact_store
from openalex_semantic_search.api import bucket_min_citations, create_app
from openalex_semantic_search.compact_store import (
    CompactMetadataStore,
    FilterScanBudget,
    FilterScanBudgetExceeded,
)
from openalex_semantic_search.store import Filters

KEY = "test-application-key"
AUTH = {"X-API-Key": KEY}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def as_dict(self):
        return self.payload


def fake_engine(search=None):
    calls: list[Filters] = []

    def default_search(query, *, limit, offset, sort, filters):
        calls.append(filters)
        return FakeResponse({"query": query, "results": []})

    engine = SimpleNamespace(
        record_count=10,
        manifest={"stage": "test", "backend": "numpy"},
        generation_id="gen",
        snapshot_at="2026-01-01",
        overrides=(),
        title_prefixes=None,
        work_ids=None,
        search=search or default_search,
        calls=calls,
    )
    return engine


@pytest.fixture()
def make_client(tmp_path):
    clients = []

    def build(engine=None):
        engine = engine or fake_engine()
        app = create_app(
            api_keys=frozenset({KEY}), engine=engine, progress_artifacts=tmp_path
        )
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client, engine

    yield build
    for client in clients:
        client.__exit__(None, None, None)


# Budget -------------------------------------------------------------------


def test_the_budget_refuses_without_waiting_and_recovers_after_the_window():
    now = [100.0]
    budget = FilterScanBudget(scans=2, window_seconds=10, clock=lambda: now[0])
    budget.acquire()
    budget.acquire()
    started = time.perf_counter()
    with pytest.raises(FilterScanBudgetExceeded) as refused:
        budget.acquire()
    assert time.perf_counter() - started < 0.1
    assert refused.value.retry_after == 10
    assert not isinstance(refused.value, RuntimeError)
    now[0] += 10
    budget.acquire()


def test_the_budget_reads_its_environment():
    budget = FilterScanBudget.from_environment(
        {
            "OPENALEX_SEARCH_FILTER_SCANS": "5",
            "OPENALEX_SEARCH_FILTER_SCAN_WINDOW_SECONDS": "2.5",
        }
    )
    assert (budget.scans, budget.window_seconds) == (5, 2.5)
    with pytest.raises(ValueError):
        FilterScanBudget.from_environment({"OPENALEX_SEARCH_FILTER_SCANS": "0"})


def test_an_exhausted_budget_is_a_fast_429(make_client):
    def refuse(*_args, **_kwargs):
        raise FilterScanBudgetExceeded(3.2)

    client, _ = make_client(fake_engine(search=refuse))
    started = time.perf_counter()
    response = client.post("/search", json={"query": "x", "topic": "t"}, headers=AUTH)
    assert time.perf_counter() - started < 1.0
    assert response.status_code == 429
    assert response.headers["retry-after"] == "4"
    assert response.json() == {"detail": "Too many new filter combinations; retry shortly"}


def test_a_real_engine_surfaces_the_refusal_as_429(production_shaped):
    """The engine's RuntimeError fallback must not swallow the refusal."""
    from openalex_semantic_search.embeddings import HashingEmbedder
    from openalex_semantic_search.engine import SearchEngine

    generation, prefixes, work_ids = production_shaped
    engine = SearchEngine(
        generation, HashingEmbedder(), title_prefixes=prefixes, work_ids=work_ids
    )
    engine.store.scan_budget = FilterScanBudget(scans=1, window_seconds=60)
    app = create_app(api_keys=frozenset({KEY}), engine=engine)
    try:
        with TestClient(app) as client:
            first = {"query": "research paper", "min_citations": 10, "field": "Medicine"}
            assert client.post("/search", json=first, headers=AUTH).status_code == 200
            # No part shared with the first, so nothing cached can answer it.
            second = {"query": "research paper", "min_citations": 100, "topic": "Other"}
            refused = client.post("/search", json=second, headers=AUTH)
            assert refused.status_code == 429
            assert int(refused.headers["retry-after"]) >= 1
            # The first combination is cached and still served.
            assert client.post("/search", json=first, headers=AUTH).status_code == 200
    finally:
        engine.close()


# Cache caps ---------------------------------------------------------------


def bare_store() -> CompactMetadataStore:
    store = object.__new__(CompactMetadataStore)
    store._filter_cache = OrderedDict()
    store._cache_lock = threading.RLock()
    return store


def test_the_filter_cache_evicts_least_recent_entries_past_the_entry_cap(monkeypatch):
    monkeypatch.setattr(compact_store, "FILTER_CACHE_MAX_ENTRIES", 3)
    store = bare_store()
    keys = [Filters(min_citations=value) for value in range(5)]
    for key in keys[:3]:
        store._remember_filter(key, (1, None))
    assert store._cached_filter(keys[0]) is not None  # Now most recent.
    store._remember_filter(keys[3], (1, None))
    store._remember_filter(keys[4], (1, None))
    assert list(store._filter_cache) == [keys[0], keys[3], keys[4]]


def test_the_filter_cache_evicts_past_the_id_byte_cap(monkeypatch):
    monkeypatch.setattr(compact_store, "FILTER_CACHE_MAX_ID_BYTES", 200)
    store = bare_store()
    ids = np.arange(10, dtype=np.int64)  # 80 bytes each.
    for value in range(4):
        store._remember_filter(Filters(min_citations=value), (10, ids.copy()))
    assert len(store._filter_cache) == 2
    assert sum(item[1].nbytes for item in store._filter_cache.values()) <= 200
    # A single oversized entry is still kept rather than thrashing.
    store._remember_filter(Filters(topic="big"), (1000, np.arange(1000, dtype=np.int64)))
    assert list(store._filter_cache) == [Filters(topic="big")]


# min_citations buckets ----------------------------------------------------


@pytest.mark.parametrize(
    ("value", "bucket"),
    [
        (None, None),
        (0, 0),
        (9, 0),
        (10, 10),
        (49, 10),
        (50, 50),
        (99, 50),
        (100, 100),
        (499, 100),
        (500, 500),
        (999, 500),
        (1000, 1000),
        (4999, 1000),
        (5000, 5000),
        (9999, 5000),
        (10000, 10000),
        (2_147_483_647, 10000),
    ],
)
def test_min_citations_round_down_to_a_bucket(value, bucket):
    assert bucket_min_citations(value) == bucket


def test_search_passes_the_bucketed_value_to_the_engine(make_client):
    client, engine = make_client()
    response = client.post("/search", json={"query": "x", "min_citations": 777}, headers=AUTH)
    assert response.status_code == 200
    assert engine.calls[-1].min_citations == 500


# Request bounds -----------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {"min_citations": -1},
        {"min_citations": 2_147_483_648},
        {"year_min": -1},
        {"year_max": 2101},
        {"limit": 51},
        {"offset": 10_001},
    ],
)
def test_out_of_range_values_are_a_422(make_client, extra):
    client, _ = make_client()
    response = client.post("/search", json={"query": "x", **extra}, headers=AUTH)
    assert response.status_code == 422


def test_the_largest_citation_bound_is_accepted(make_client):
    client, _ = make_client()
    body = {"query": "x", "min_citations": 2_147_483_647, "year_min": 0, "year_max": 2100}
    assert client.post("/search", json=body, headers=AUTH).status_code == 200


# Unauthenticated surface --------------------------------------------------


def test_the_openapi_schema_is_hidden_by_default(make_client):
    client, _ = make_client()
    assert client.get("/openapi.json").status_code == 404


def test_the_openapi_schema_can_be_exposed(make_client, monkeypatch):
    monkeypatch.setenv("OPENALEX_SEARCH_EXPOSE_OPENAPI", "1")
    client, _ = make_client()
    assert client.get("/openapi.json").status_code == 200


def test_healthz_is_minimal(make_client):
    client, _ = make_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_health_details_need_the_key(make_client):
    client, _ = make_client()
    assert client.get("/healthz/details").status_code == 401
    body = client.get("/healthz/details", headers=AUTH).json()
    assert body["status"] == "ok" and body["records"] == 10


def test_index_progress_needs_the_key(make_client):
    client, _ = make_client()
    assert client.get("/index-progress").status_code == 401
    assert client.get("/index-progress", headers={"X-API-Key": "wrong"}).status_code == 401
    response = client.get("/index-progress", headers=AUTH)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("path", ["/", "/app.js", "/app.css"])
def test_the_lab_ui_is_off_by_default(make_client, path):
    client, _ = make_client()
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("path", ["/", "/app.js", "/app.css"])
def test_the_lab_ui_can_be_enabled(make_client, monkeypatch, path):
    monkeypatch.setenv("OPENALEX_SEARCH_UI", "1")
    client, _ = make_client()
    assert client.get(path).status_code == 200
