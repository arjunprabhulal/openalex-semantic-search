"""Query-time search quality on a production-shaped generation.

Every test here runs against 5,000 rows served with a 50-row candidate pool on
the compact backend (see conftest.py), so the candidate set is a small slice
of the corpus, as it is at 315M rows.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import CANDIDATE_POOL, CORPUS_ROWS
from openalex_semantic_search.config import SearchTuning
from openalex_semantic_search.embeddings import HashingEmbedder
from openalex_semantic_search.engine import (
    SearchEngine,
    SearchResult,
    TITLE_ONLY_PENALTY,
    _RankedCandidate,
    _same_work,
)
from openalex_semantic_search.overrides import MetadataOverrides
from openalex_semantic_search.store import Filters
from openalex_semantic_search.title_prefix import TitlePrefixIndex


@pytest.fixture()
def engine(production_shaped):
    generation, prefixes, work_ids = production_shaped
    served = SearchEngine(
        generation, HashingEmbedder(), title_prefixes=prefixes, work_ids=work_ids
    )
    yield served
    served.close()


def work_ids(response) -> list[str]:
    return [result.work_id for result in response.results]


def test_generation_is_production_shaped(engine):
    assert engine.metadata_backend == "compact"
    assert engine.record_count == CORPUS_ROWS
    assert engine.candidate_count == CANDIDATE_POOL
    response = engine.search("research paper", limit=20)
    assert response.candidates == CANDIDATE_POOL
    assert response.total_matches_capped is True


# --- 1. Title-key variants -------------------------------------------------


@pytest.mark.parametrize(
    "typed",
    [
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        # Typed without hyphens: dropped from rank 1 to 17 in production.
        "Retrieval Augmented Generation for Knowledge Intensive NLP Tasks",
        "retrieval augmented generation for knowledge intensive nlp tasks",
        # Typed with typographic dashes, stored with ASCII hyphens.
        "Retrieval‐Augmented Generation for Knowledge–Intensive NLP Tasks",
        # Pasted from a publisher page.
        "<i>Retrieval-Augmented</i> Generation for Knowledge-Intensive NLP Tasks",
    ],
)
def test_exact_title_resolves_across_hyphen_spellings(engine, typed):
    response = engine.search(typed, limit=5)
    assert response.results[0].work_id == "W3098425262"
    assert response.results[0].title_match is True


def test_typographic_hyphen_stored_ascii_hyphen_typed(engine):
    # Stored "Pre‐training" normalizes to "pre training"; the typed
    # ASCII hyphen stays inside the token. The space variant bridges them.
    typed = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    response = engine.search(typed, limit=5)
    assert response.results[0].work_id == "W2963341956"
    assert response.results[0].title_match is True


def test_hyphen_removed_variant_matches_a_joined_stored_word(engine):
    response = engine.search("Pre-training Language Models at Scale", limit=5)
    assert response.results[0].work_id == "WPRETRAIN"
    assert response.results[0].title_match is True


def test_partial_title_without_hyphens_resolves_through_the_prefix_table(engine):
    # Sixty decoy titles begin with exactly the typed words, so the paper is
    # not first -- but it is now reached at all, as a prefix match.
    query = "retrieval augmented generation for knowledge intensive"
    _, prefix_rows, _ = engine._title_candidates(query, Filters())
    stored = engine.store.fetch(prefix_rows)
    assert "https://openalex.org/W3098425262" in {item["openalex_id"] for item in stored.values()}
    found = [
        result
        for offset in (0, 50)
        for result in engine.search(query, limit=50, offset=offset).results
        if result.work_id == "W3098425262"
    ]
    assert found and found[0].title_match is True


# --- 2. Sort ----------------------------------------------------------------


def test_every_sort_is_honoured_over_the_merged_pool(engine):
    # "CRISPR/Cas9 genome editing" is an exact title of three works and the
    # opening of two more; they used to be pinned first under every sort.
    query = "CRISPR Cas9 genome editing"
    relevance = engine.search(query, limit=50)
    exact = {"WCRISPREXACT0", "WCRISPREXACT1", "WCRISPREXACT2"}
    assert set(work_ids(relevance)[:3]) == exact
    assert work_ids(relevance)[0] == "WCRISPREXACT0"  # most cited exact title

    for sort, key, reverse in (
        ("most_cited", lambda r: r.cited_by_count, True),
        ("newest", lambda r: r.publication_year, True),
        ("oldest", lambda r: r.publication_year, False),
    ):
        response = engine.search(query, limit=50, sort=sort)
        # The rows that clear the relevance floor, exact titles included, in
        # the requested order throughout; the pinned titles do not interrupt it.
        assert exact <= set(work_ids(response))
        assert {r.row_id for r in response.results} <= {r.row_id for r in relevance.results}
        values = [key(r) for r in response.results]
        assert values == sorted(values, reverse=reverse), sort
        assert response.total_matches == len(response.results) < relevance.total_matches
    newest = engine.search(query, limit=5, sort="newest")
    # Previously the pinned, most-cited 2014 exact title led every sort.
    assert work_ids(newest)[0] != "WCRISPREXACT0"
    assert newest.results[0].publication_year >= 2023


def test_relevance_floor_keeps_famous_unrelated_work_out_of_most_cited(engine):
    response = engine.search("Attention Is All You Need", limit=20, sort="most_cited")
    assert response.results[0].work_id == "W2626778328"
    assert "WEFFORT" not in work_ids(response)  # 500k citations, 1973, unrelated


def test_order_pins_exact_titles_only_under_relevance():
    def row(row_id, score, cited, year, exact=False):
        return _RankedCandidate(row_id, score, score, cited, year, exact, exact)

    rows = [row(1, 0.5, 10, 2001, exact=True), row(2, 0.9, 1000, 2020), row(3, 0.88, 5, 2024)]
    assert [r.row_id for r in SearchEngine._order(rows, "relevance")] == [1, 2, 3]
    assert [r.row_id for r in SearchEngine._order(rows, "most_cited")] == [2, 1, 3]
    assert [r.row_id for r in SearchEngine._order(rows, "newest")] == [3, 2, 1]
    assert [r.row_id for r in SearchEngine._order(rows, "oldest")] == [1, 2, 3]
    undated = row(4, 0.9, 0, 0)
    assert SearchEngine._order([undated, *rows], "oldest")[-1] is undated


# --- 3. Prefix rows: bounded boost, chosen by citations, verified -------------


def test_prefix_match_is_a_bounded_boost_not_a_pin():
    prefix = SearchEngine._ranking_score_values(2000, 0, 0.50, True, False, prefix_match=True)
    semantic = SearchEngine._ranking_score_values(2000, 0, 0.90, False, False)
    near = SearchEngine._ranking_score_values(2000, 0, 0.60, False, False)
    assert prefix < semantic  # a far better semantic match still wins
    assert prefix > near  # but a prefix match beats a comparable one


def test_prefix_rows_are_chosen_by_citations_not_storage_order(engine):
    rows = engine._prefix_rows(["a study of the effect of"], Filters())
    assert len(rows) == 50
    stored = engine.store.fetch(rows)
    ids = {stored[row]["openalex_id"].rsplit("/", 1)[-1] for row in rows}
    assert "WSTUDYLANDMARK" in ids  # stored last, most cited
    assert "WSTUDY0" not in ids  # stored first, least cited
    response = engine.search("A study of the effect of", limit=3)
    assert response.results[0].work_id == "WSTUDYLANDMARK"


def test_prefix_hits_are_verified_against_the_stored_title(production_shaped, tmp_path):
    generation, prefixes, work_ids_index = production_shaped
    # A prefix index whose rows point at the wrong works (stale or corrupt).
    shuffled = TitlePrefixIndex(
        hashes=prefixes.hashes,
        rows=(np.asarray(prefixes.rows) + CORPUS_ROWS // 2) % CORPUS_ROWS,
        directory=tmp_path,
        manifest=dict(prefixes.manifest),
    )
    served = SearchEngine(
        generation, HashingEmbedder(), title_prefixes=shuffled, work_ids=work_ids_index
    )
    try:
        assert served._prefix_rows(["a study of the effect of"], Filters()) == []
    finally:
        served.close()


def test_lookup_indexes_for_another_generation_are_disabled(production_shaped, tmp_path):
    generation, prefixes, work_ids_index = production_shaped
    stale = TitlePrefixIndex(
        hashes=prefixes.hashes,
        rows=prefixes.rows,
        directory=tmp_path,
        manifest={**prefixes.manifest, "records": CORPUS_ROWS + 1},
    )
    served = SearchEngine(
        generation, HashingEmbedder(), title_prefixes=stale, work_ids=work_ids_index
    )
    try:
        assert served.title_prefixes is None
        assert served.work_ids is work_ids_index
    finally:
        served.close()


# --- 4. Duplicate collapse -------------------------------------------------------


def test_duplicate_records_of_one_work_collapse_to_the_most_cited(engine):
    response = engine.search("Deep Residual Learning for Image Recognition", limit=10)
    ids = work_ids(response)
    assert ids[0] == "W2194775991"
    assert "W2949650786" not in ids  # same first author: the same work
    assert "WRESNETOTHER" in ids  # a different author stays separate


def test_duplicate_rules():
    base = {"title": "T", "doi": "", "authors": ["Kaiming He"], "publication_year": 2016}
    assert _same_work(base, {**base, "publication_year": 2009})  # same first author
    assert _same_work(base, {**base, "authors": ["K. He"]})  # surname match
    assert not _same_work(base, {**base, "authors": ["Someone Else"]})
    assert _same_work(
        {**base, "doi": "https://doi.org/10.1/X"},
        {**base, "authors": ["Other Person"], "doi": "10.1/x"},
    )  # same DOI wins
    anonymous = {**base, "authors": []}
    assert _same_work(anonymous, {**anonymous, "publication_year": 2017})
    assert not _same_work(anonymous, {**anonymous, "publication_year": 2019})


def test_authorless_title_duplicates_within_a_year_collapse(engine):
    response = engine.search("Proceedings of the tiny workshop", limit=5)
    ids = work_ids(response)
    assert ids[0] == "WNOAUTHORB"
    assert "WNOAUTHORA" not in ids


# --- 5. Filters -------------------------------------------------------------


def test_empty_topic_and_field_are_absent(engine):
    baseline = engine.search("research paper", limit=10)
    blank = engine.search("research paper", limit=10, filters=Filters(topic="", field=""))
    assert blank.filtered_records == CORPUS_ROWS
    assert work_ids(blank) == work_ids(baseline)


def test_year_filters_exclude_records_with_no_year(engine):
    response = engine.search("Undated research paper", limit=50, filters=Filters(year_max=2000))
    assert response.results
    assert all(0 < r.publication_year <= 2000 for r in response.results)
    assert engine._count(Filters(year_max=2100)) == CORPUS_ROWS - 5


def test_exact_filter_path_defaults_to_ten_thousand_rows(engine):
    assert engine.selective_filter_threshold == 10_000


def test_filtered_candidates_are_widened_and_totals_are_honest(production_shaped):
    generation, prefixes, work_ids_index = production_shaped
    # Force the candidate-index path for a filter matching 1/3 of the corpus.
    served = SearchEngine(
        generation,
        HashingEmbedder(),
        title_prefixes=prefixes,
        work_ids=work_ids_index,
        selective_filter_threshold=0,
    )
    try:
        filters = Filters(field="Medicine")
        response = served.search("research paper", limit=20, filters=filters)
        assert served.filter_candidate_count == 2 * CANDIDATE_POOL
        assert response.candidates > CANDIDATE_POOL  # widened pool
        assert all(r.field == "Medicine" for r in response.results)
        assert response.total_matches_capped is True
        # Paging to the end never yields an empty page while has_more is set.
        offset, seen = 0, set()
        while True:
            page = served.search("research paper", limit=20, offset=offset, filters=filters)
            assert page.results or not page.has_more
            seen.update(r.row_id for r in page.results)
            if not page.has_more:
                break
            offset = page.next_offset()
        assert len(seen) == response.total_matches
    finally:
        served.close()


def test_under_filled_filter_widens_nprobe(tmp_path):
    pytest.importorskip("faiss")
    from conftest import build_production_shaped, filler

    papers = [filler(index) for index in range(2_000)]
    generation = build_production_shaped(tmp_path, papers, candidate_count=50, backend="faiss")
    manifest = json.loads((generation / "manifest.json").read_text())
    assert manifest["nprobe"] == 4
    filters = Filters(field="Medicine", min_citations=4_000)
    narrow = SearchEngine(
        generation,
        HashingEmbedder(),
        selective_filter_threshold=0,
        tuning=SearchTuning(nprobe=1, filter_nprobe=1),
    )
    widened = SearchEngine(
        generation,
        HashingEmbedder(),
        selective_filter_threshold=0,
        tuning=SearchTuning(nprobe=1, filter_nprobe=50),
    )
    try:
        few = narrow.search("research paper", limit=50, filters=filters)
        more = widened.search("research paper", limit=50, filters=filters)
        assert more.candidates > few.candidates
        assert all(r.field == "Medicine" and r.cited_by_count >= 4_000 for r in more.results)
    finally:
        narrow.close()
        widened.close()


# --- 6. Corrections drive ranking, sort and filters ---------------------------


@pytest.fixture()
def corrected(production_shaped, tmp_path):
    generation, prefixes, work_ids_index = production_shaped
    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "entries": {"W2626778328": {"publication_year": 2017, "cited_by_count": 173_000}},
            }
        ),
        encoding="utf-8",
    )
    served = SearchEngine(
        generation,
        HashingEmbedder(),
        title_prefixes=prefixes,
        work_ids=work_ids_index,
        overrides=MetadataOverrides.load(path),
    )
    yield served
    served.close()


def test_corrected_year_decides_year_filters(corrected):
    query = "Attention Is All You Need"
    inside = corrected.search(query, limit=5, filters=Filters(year_max=2020))
    assert inside.results[0].work_id == "W2626778328"
    assert inside.results[0].publication_year == 2017
    exact_year = corrected.search(query, limit=5, filters=Filters(year_min=2017, year_max=2017))
    assert work_ids(exact_year)[:1] == ["W2626778328"]
    outside = corrected.search(query, limit=5, filters=Filters(year_min=2024))
    assert "W2626778328" not in work_ids(outside)


def test_corrected_values_drive_sorting(corrected):
    row = next(iter(corrected._patched_rows))
    years, citations = corrected._rank_features([row])
    assert (int(years[0]), int(citations[0])) == (2017, 173_000)
    # Ranked and displayed on the same, corrected values.
    response = corrected.search("Attention Is All You Need", limit=5, sort="most_cited")
    hit = response.results[0]
    assert hit.work_id == "W2626778328"
    assert (hit.publication_year, hit.cited_by_count) == (2017, 173_000)
    assert hit.overridden_fields == ["cited_by_count", "publication_year"]


# --- 7. Title-only penalty -------------------------------------------------------


def test_title_only_records_are_demoted_for_multi_word_queries(engine):
    def row(row_id, score):
        return _RankedCandidate(row_id, score, score, 0, 2000, False, False)

    ranked = [row(1, 0.80), row(2, 0.79)]
    items = {
        1: {"title": "Alpha", "snippet": "", "authors": ["A"], "publication_year": 2000, "doi": ""},
        2: {"title": "Beta", "snippet": "abstract", "authors": ["B"], "publication_year": 2000, "doi": ""},
    }
    rich = lambda batch: [(candidate, items[candidate.row_id]) for candidate in batch]
    kwargs = dict(sort="relevance", scale=1.0)
    demoted = engine._rerank_window(ranked, rich, multi_word=True, **kwargs)
    assert [r.row_id for r in demoted] == [2, 1]
    assert demoted[1].ranking_score == pytest.approx(0.80 - TITLE_ONLY_PENALTY)
    single = engine._rerank_window(ranked, rich, multi_word=False, **kwargs)
    assert [r.row_id for r in single] == [1, 2]


# --- 8. Pagination ---------------------------------------------------------------


def test_offsets_past_the_pool_are_clamped_with_an_honest_total(engine):
    first = engine.search("research paper", limit=20)
    past = engine.search("research paper", limit=20, offset=10_000)
    assert past.results == []
    assert past.has_more is False
    assert past.offset == first.total_matches == past.total_matches
    assert past.as_dict()["next_offset"] is None


def test_pages_cover_the_pool_exactly_once(engine):
    offset, seen, pages = 0, [], 0
    while True:
        page = engine.search("research paper", limit=20, offset=offset, sort="newest")
        seen.extend(r.row_id for r in page.results)
        pages += 1
        if not page.has_more:
            break
        assert page.as_dict()["next_offset"] == page.next_offset() == offset + 20
        offset = page.next_offset()
    assert len(seen) == len(set(seen)) == page.total_matches
    assert pages == -(-page.total_matches // 20)


# --- Lookup shape, tuning and model check ------------------------------------------


def test_work_id_lookup_returns_the_search_result_field_set(engine):
    work = engine.fetch_work("https://openalex.org/W2194775991")
    assert set(work) == set(SearchResult.__slots__)
    assert work["work_id"] == "W2194775991"
    assert work["view_url"] == "https://doi.org/10.1109/cvpr.2016.90"


def test_default_tuning_serves_the_manifest_values(production_shaped):
    generation, _, _ = production_shaped
    manifest = json.loads((generation / "manifest.json").read_text())
    served = SearchEngine(generation, HashingEmbedder())
    try:
        assert served.nprobe == manifest["nprobe"]
        assert served.candidate_count == manifest["candidate_count"]
    finally:
        served.close()
    tuned = SearchEngine(
        generation, HashingEmbedder(), tuning=SearchTuning(nprobe=3, candidates=80)
    )
    try:
        assert (tuned.nprobe, tuned.candidate_count, tuned.filter_candidate_count) == (3, 80, 160)
    finally:
        tuned.close()


def test_tuning_reads_the_environment():
    tuning = SearchTuning.from_environment(
        {"OPENALEX_SEARCH_NPROBE": "512", "OPENALEX_SEARCH_CANDIDATES": " 3000 "}
    )
    assert (tuning.nprobe, tuning.candidates, tuning.filter_nprobe) == (512, 3000, None)
    assert SearchTuning.from_environment({}) == SearchTuning()
    with pytest.raises(ValueError, match="OPENALEX_SEARCH_NPROBE"):
        SearchTuning.from_environment({"OPENALEX_SEARCH_NPROBE": "many"})
    with pytest.raises(ValueError, match="positive"):
        SearchTuning.from_environment({"OPENALEX_SEARCH_CANDIDATES": "0"})


def test_a_different_query_model_is_flagged(production_shaped):
    generation, _, _ = production_shaped

    class Renamed(HashingEmbedder):
        def __init__(self):
            super().__init__()
            self.name = "some-other-model"

    served = SearchEngine(generation, Renamed())
    try:
        assert served.embedder_matches_generation is False
    finally:
        served.close()
