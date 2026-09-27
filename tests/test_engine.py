from dataclasses import replace
from pathlib import Path

import pytest

from openalex_semantic_search.benchmark import run_benchmark
from openalex_semantic_search.builder import build_generation
from openalex_semantic_search.config import Stage
from openalex_semantic_search.embeddings import HashingEmbedder
from openalex_semantic_search.engine import SearchEngine
from openalex_semantic_search.records import Paper
from openalex_semantic_search.store import Filters


def paper(index: int) -> Paper:
    if index == 0:
        title = "Attention Is All You Need"
        abstract = "Transformer attention architecture for sequence modeling"
        year = 2017
        citations = 150_000
    elif index == 1:
        title = "Deep Residual Learning for Image Recognition"
        abstract = "Residual networks make deep image models easier to optimize"
        year = 2016
        citations = 210_000
    elif index == 2:
        title = "Graph Attention Networks"
        abstract = "Neural graph representation learning with masked self attention"
        year = 2018
        citations = 45_000
    else:
        title = f"Measured research paper {index}"
        abstract = f"A controlled study of topic {index % 17} and method {index % 31}"
        year = 1981 if index == 3 else 2000 + index % 25
        citations = index % 200
    return Paper(
        openalex_id=f"https://openalex.org/W{index + 1}",
        title=title,
        embedding_text=f"{title}. {abstract}",
        snippet=abstract,
        authors=(f"Researcher {index}",),
        publication_year=year,
        doi=f"https://doi.org/10.1000/{index}",
        cited_by_count=citations,
        topic=f"Topic {index % 17}",
        field="Computer Science",
        is_oa=index % 2 == 0,
        oa_url=f"https://example.test/{index}" if index % 2 == 0 else "",
    )


def built_generation(tmp_path: Path) -> tuple[Path, HashingEmbedder]:
    embedder = HashingEmbedder()
    generation = tmp_path / "generation"
    build_generation(
        (paper(index) for index in range(320)),
        stage=Stage("test", 320, 8, 2, candidate_count=320),
        output=generation,
        embedder=embedder,
        backend="numpy",
        batch_size=64,
    )
    return generation, embedder


def test_exact_title_and_selective_filters(tmp_path):
    generation, embedder = built_generation(tmp_path)
    engine = SearchEngine(generation, embedder, selective_filter_threshold=500)
    try:
        response = engine.search("Attention is all you", limit=20)
        assert response.results[0].title == "Attention Is All You Need"
        assert response.results[0].title_match is True

        filtered = engine.search(
            "research paper",
            limit=20,
            filters=Filters(year_min=1981, year_max=1981),
        )
        assert filtered.filtered_records == 1
        assert [result.publication_year for result in filtered.results] == [1981]

        open_access = engine.search(
            "research paper",
            limit=20,
            filters=Filters(open_access_only=True, min_citations=10),
        )
        assert open_access.results
        assert all(result.is_oa and result.cited_by_count >= 10 for result in open_access.results)
        with pytest.raises(ValueError, match="year_min"):
            engine.search(
                "research paper",
                filters=Filters(year_min=2020, year_max=2010),
            )
    finally:
        engine.close()


def test_exact_title_boost_outweighs_broad_fts_match():
    metadata = {"cited_by_count": 100, "publication_year": 2020}
    exact = SearchEngine._ranking_score(
        metadata,
        semantic_score=0.70,
        title_match=True,
        exact_title_match=True,
    )
    broader_phrase = SearchEngine._ranking_score(
        metadata,
        semantic_score=0.90,
        title_match=True,
        exact_title_match=False,
    )
    assert exact > broader_phrase


def test_pagination_pages_through_ranked_results(tmp_path):
    generation, embedder = built_generation(tmp_path)
    engine = SearchEngine(generation, embedder)
    try:
        first = engine.search("research paper", limit=50, offset=0)
        second = engine.search("research paper", limit=50, offset=50)

        assert len(first.results) == 50
        assert first.has_more is True
        assert first.total_matches > 50
        assert second.offset == 50
        assert second.results
        first_ids = {result.row_id for result in first.results}
        second_ids = {result.row_id for result in second.results}
        assert not first_ids & second_ids

        tail = engine.search("research paper", limit=50, offset=10_000)
        assert tail.results == [] and tail.has_more is False
        with pytest.raises(ValueError, match="offset"):
            engine.search("research paper", offset=-1)
    finally:
        engine.close()


def test_sort_orders_are_deterministic_and_page_stable(tmp_path):
    generation, embedder = built_generation(tmp_path)
    engine = SearchEngine(generation, embedder)
    try:
        cited = engine.search("research paper", limit=50, sort="most_cited")
        counts = [result.cited_by_count for result in cited.results]
        assert counts == sorted(counts, reverse=True)
        assert cited.sort == "most_cited" and cited.page_size == 50

        newest = engine.search("research paper", limit=50, sort="newest")
        years = [result.publication_year for result in newest.results]
        assert years == sorted(years, reverse=True)

        oldest = engine.search("research paper", limit=50, sort="oldest")
        assert oldest.results[0].publication_year <= oldest.results[-1].publication_year

        # Pages under a non-relevance sort must not overlap or reshuffle.
        page_one = engine.search("research paper", limit=50, sort="most_cited", offset=0)
        page_two = engine.search("research paper", limit=50, sort="most_cited", offset=50)
        one_ids = [result.row_id for result in page_one.results]
        two_ids = [result.row_id for result in page_two.results]
        assert not set(one_ids) & set(two_ids)
        rerun = engine.search("research paper", limit=50, sort="most_cited", offset=0)
        assert [result.row_id for result in rerun.results] == one_ids

        with pytest.raises(ValueError, match="sort"):
            engine.search("research paper", sort="banana")
    finally:
        engine.close()


def test_exact_title_is_pinned_only_under_relevance(tmp_path):
    embedder = HashingEmbedder()
    papers = [
        replace(
            paper(0),
            openalex_id="https://openalex.org/WCANONICAL",
            cited_by_count=150_000,
        ),
        replace(
            paper(1),
            openalex_id="https://openalex.org/WDUPLICATE",
            title="Attention Is All You Need",
            embedding_text="Attention Is All You Need. A later work with the same title.",
            cited_by_count=5,
        ),
        replace(
            paper(2),
            openalex_id="https://openalex.org/WBROADER",
            title="Attention and Effort",
            embedding_text="Attention and effort in cognition.",
            cited_by_count=500_000,
        ),
    ]
    generation = tmp_path / "exact-title-generation"
    build_generation(
        iter(papers),
        stage=Stage("test", len(papers), 2, 1, candidate_count=len(papers)),
        output=generation,
        embedder=embedder,
        backend="numpy",
        batch_size=3,
    )
    engine = SearchEngine(generation, embedder)
    ids = lambda response: [result.openalex_id.rsplit("/", 1)[-1] for result in response.results]
    try:
        # Relevance pins the exact title, same-title works by citations.
        # (Different first authors, so the two are not collapsed as duplicates.)
        relevance = engine.search("Attention Is All You Need", limit=3)
        assert ids(relevance) == ["WCANONICAL", "WDUPLICATE", "WBROADER"]
        # Other sorts order the merged pool by their own key and are not
        # overridden by the pin. The relevance floor keeps a famous but
        # unrelated work (500k citations) from leading "most cited" for a
        # title query -- the "Attention and Effort" case seen in production.
        cited = engine.search("Attention Is All You Need", limit=3, sort="most_cited")
        assert ids(cited) == ["WCANONICAL", "WDUPLICATE"]
        oldest = engine.search("Attention Is All You Need", limit=3, sort="oldest")
        assert ids(oldest)[:2] == ["WDUPLICATE", "WCANONICAL"]  # 2016, then 2017
        newest = engine.search("Attention Is All You Need", limit=3, sort="newest")
        assert ids(newest)[:2] == ["WCANONICAL", "WDUPLICATE"]
    finally:
        engine.close()


def test_benchmark_reports_quality_size_and_concurrency(tmp_path, monkeypatch):
    # The production latency budget measures the serving host, not this test
    # machine; a generous budget keeps the gate wiring tested without timing flakes.
    monkeypatch.setattr("openalex_semantic_search.benchmark.QUERY_BUDGET_MS", 60_000.0)
    generation, embedder = built_generation(tmp_path)
    report = run_benchmark(
        generation,
        embedder,
        [
            "Attention Is All You Need",
            "Deep Residual Learning for Image Recognition",
            "Graph Attention Networks",
        ],
        repeats=1,
        concurrency_levels=(1, 2),
    )
    assert report["pq_recall_at_candidates_mean"] == 1.0
    # The smoke embedder intentionally has many tied zero-similarity scores;
    # calibrated INT8 recall itself is tested on dense vectors separately.
    assert 0.0 <= report["int8_refined_recall_at_20_mean"] <= 1.0
    assert report["exact_title_success_rate"] == 1.0
    assert report["selective_filter_correct"] is True
    assert report["measured_production_bytes_per_record"] > 0
    assert report["projected_full_corpus_bytes"] > 0
    assert "storage_projection_pass" in report["gate"]
    assert report["gate"]["latency_and_stability_pass"] is True
    assert set(report["component_build_seconds"]) == {
        "total",
        "embedding",
        "candidate_index",
        "metadata_and_title_fts",
    }
    assert set(report["concurrency"]) == {"1", "2"}


def test_pagination_terminates_when_the_tail_is_filtered_away(tmp_path):
    """A client must never be handed the same next_offset forever.

    Rows are verified against rich metadata after ranking, so a filter can
    reject the tail of the candidate set. has_more was judged against the
    unfiltered candidate count, so the last page reported more results, the
    client re-requested the same offset, received nothing, and was told again
    that more existed.
    """
    generation, embedder = built_generation(tmp_path)
    engine = SearchEngine(generation, embedder)
    try:
        seen = set()
        offset = 0
        for _ in range(60):  # Bounded: an unfixed loop never terminates.
            page = engine.search(
                "research paper", limit=20, offset=offset, filters=Filters(min_citations=1)
            )
            for result in page.results:
                assert result.row_id not in seen, "a page repeated an earlier row"
                seen.add(result.row_id)
            if not page.has_more:
                break
            assert page.results, "has_more with an empty page traps a client"
            assert page.next_offset() > offset
            offset = page.next_offset()
        else:
            raise AssertionError("pagination did not terminate")
    finally:
        engine.close()


def test_empty_topic_and_field_filters_are_ignored_not_rejected(tmp_path):
    """The stores skip a falsy topic, so the post-ranking recheck must too.

    Checking `is not None` meant a caller sending topic="" had every candidate
    rejected after ranking and received zero results.
    """
    generation, embedder = built_generation(tmp_path)
    engine = SearchEngine(generation, embedder)
    try:
        baseline = engine.search("research paper", limit=5)
        blank = engine.search("research paper", limit=5, filters=Filters(topic="", field=""))
        assert blank.results, "an empty topic filter rejected every result"
        assert [r.row_id for r in blank.results] == [r.row_id for r in baseline.results]
    finally:
        engine.close()


def test_selective_filters_do_not_collapse_the_candidate_set(tmp_path):
    """A filter matching few rows must not shrink results to almost nothing.

    Candidates are post-filtered, so a fixed request multiplier keeps only the
    eligible share: on the 315M generation a filter matching 0.04% of the
    corpus left 3 of 8,000 candidates, and a query with 135,963 matching works
    returned 85 results.
    """
    generation, embedder = built_generation(tmp_path)
    engine = SearchEngine(generation, embedder, selective_filter_threshold=0)
    try:
        # threshold 0 forces the broad, post-filtered path for every filter.
        unfiltered = engine.search("research paper", limit=50)
        filtered = engine.search(
            "research paper", limit=50, filters=Filters(min_citations=1)
        )
        assert filtered.filtered_records > 0
        # The surviving candidate set must stay a meaningful fraction of what
        # an unfiltered query ranks, not collapse to a handful.
        assert filtered.total_matches >= min(
            filtered.filtered_records, unfiltered.total_matches
        ) * 0.5
    finally:
        engine.close()
