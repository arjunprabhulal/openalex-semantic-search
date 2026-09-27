"""The benchmark scripts, exercised against a local app and index."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from openalex_semantic_search.embeddings import HashingEmbedder
from openalex_semantic_search.engine import SearchEngine

BENCHMARK = Path(__file__).resolve().parents[1] / "benchmark"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, BENCHMARK / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_labelled_queries_file_is_well_formed():
    queries = json.loads((BENCHMARK / "labelled-queries.json").read_text(encoding="utf-8"))["queries"]
    ids = [entry["id"] for entry in queries]
    assert len(ids) == len(set(ids))
    for entry in queries:
        assert entry["request"]["query"]
        assert ("expect" in entry) != ("check" in entry), entry["id"]
    for required in ("E01", "E02", "E03", "E04", "E07", "H01", "P01", "T01", "S01", "S06", "B07"):
        assert required in ids


def test_run_labelled_scores_a_live_endpoint(production_shaped, tmp_path, monkeypatch, capsys):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from openalex_semantic_search.api import create_app

    generation, prefixes, work_ids = production_shaped
    engine = SearchEngine(generation, HashingEmbedder(), title_prefixes=prefixes, work_ids=work_ids)
    app = create_app(api_keys=frozenset({"k"}), engine=engine)
    runner = load("run_labelled")
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "H01",
                        "type": "exact-variant",
                        "request": {"query": "Retrieval Augmented Generation for Knowledge Intensive NLP Tasks", "limit": 20},
                        "expect": {"work_ids": ["W3098425262"]},
                    },
                    {
                        "id": "S01",
                        "type": "sort",
                        "request": {"query": "research paper", "limit": 20, "sort": "most_cited"},
                        "check": "sorted_desc:cited_by_count",
                    },
                    {
                        "id": "C01",
                        "type": "duplicates",
                        "request": {"query": "Deep Residual Learning for Image Recognition", "limit": 20},
                        "check": "no_duplicate_titles",
                    },
                    {
                        "id": "B07",
                        "type": "offset",
                        "request": {"query": "research paper", "limit": 50, "offset": 5000},
                        "check": "past_pool",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.md"
    baseline.write_text("| H01 | exact | `q` | RAG | 17 | - | - | Y | 45 |\n", encoding="utf-8")
    try:
        with TestClient(app) as client:

            def post(url, key, body, timeout):
                response = client.post("/search", json=body, headers={"X-API-Key": key})
                return response.status_code, response.json(), 1.0

            monkeypatch.setattr(runner, "post", post)
            monkeypatch.setenv("OPENALEX_SEARCH_API_KEY", "k")
            code = runner.main(
                ["--queries", str(queries), "--baseline", str(baseline), "--delay", "0"]
            )
    finally:
        engine.close()
    output = capsys.readouterr().out
    assert "| H01 | exact-variant |" in output and "| 1 | 17 |" in output
    assert "1 better, 0 worse" in output
    assert code == 0, output


def test_tune_search_sweeps_a_faiss_generation(tmp_path, capsys):
    pytest.importorskip("faiss")
    from conftest import build_production_shaped, filler

    generation = build_production_shaped(
        tmp_path, [filler(index) for index in range(1_200)], candidate_count=50, backend="faiss"
    )
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "Q1",
                        "type": "topic",
                        "request": {"query": "research paper topic 3"},
                        "expect": {"work_ids": ["W9000003"]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    code = load("tune_search").main(
        [
            "--generation", str(generation),
            "--work-ids", str(tmp_path / "work-ids"),
            "--queries", str(queries),
            "--nprobe", "1,4",
            "--candidates", "20,50",
            "--reference-nprobe", "30",
            "--reference-k", "400",
            "--repeats", "1",
            "--threads", "1",
            "--out", str(out),
        ],
        embedder=HashingEmbedder(),
    )
    assert code == 0
    report = json.loads(out.read_text())
    assert set(report) == {"nprobe=1 k=20", "nprobe=1 k=50", "nprobe=4 k=20", "nprobe=4 k=50"}
    assert all(0.0 <= row["recall_at_20_mean"] <= 1.0 for row in report.values())
    # Recall is not guaranteed to rise with nprobe on a tiny index: extra PQ candidates
    # from other lists can displace true neighbours, and FAISS builds differ by platform.


def test_first_not_flagged_check_uses_the_flag_and_the_suspect_list():
    runner = load("run_labelled")
    assert "W2896457183" in runner.suspect_ids()
    ok = {"results": [{"work_id": "W1", "cited_by_count": 5}]}
    flagged = {"results": [{"work_id": "W1", "citation_count_unreliable": True}]}
    suspect = {"results": [{"work_id": "W2896457183", "cited_by_count": 45_681}]}
    assert runner.run_check("first_not_flagged", ok)[0] is True
    assert runner.run_check("first_not_flagged", flagged)[0] is False
    assert runner.run_check("first_not_flagged", suspect)[0] is False
    assert runner.run_check("first_not_flagged", {"results": []})[0] is False
