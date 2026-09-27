from openalex_semantic_search.records import (
    iter_benchmark_seed_works,
    latest_first_manifest_files,
    parse_work,
    reconstruct_abstract,
)


def test_snapshot_partitions_are_processed_newest_first():
    files = [
        {"url": "s3://openalex/data/jsonl/works/updated_date=2025-01-01/part_0000.gz"},
        {"url": "s3://openalex/data/jsonl/works/updated_date=2026-06-01/part_0000.gz"},
        {"url": "s3://openalex/data/jsonl/works/updated_date=2025-12-01/part_0000.gz"},
    ]

    ordered = latest_first_manifest_files(files)

    assert [item["url"] for item in ordered] == [
        "s3://openalex/data/jsonl/works/updated_date=2026-06-01/part_0000.gz",
        "s3://openalex/data/jsonl/works/updated_date=2025-12-01/part_0000.gz",
        "s3://openalex/data/jsonl/works/updated_date=2025-01-01/part_0000.gz",
    ]


def test_record_fields_do_not_leak_between_works():
    first = parse_work(
        {
            "id": "https://openalex.org/W1",
            "title": "Attention Is All You Need",
            "publication_year": 2017,
            "abstract_inverted_index": {"attention": [0], "transformers": [1]},
            "authorships": [{"author": {"display_name": "Ashish Vaswani"}}],
            "primary_topic": {
                "display_name": "Transformers",
                "field": {"display_name": "Computer Science"},
            },
            "open_access": {"is_oa": True, "oa_url": "https://example.test/paper"},
        }
    )
    second = parse_work(
        {
            "id": "https://openalex.org/W2",
            "display_name": "Title-only historical paper",
            "publication_year": 1981,
        }
    )

    assert first is not None and second is not None
    assert first.authors == ("Ashish Vaswani",)
    assert second.authors == ()
    assert second.snippet == ""
    assert second.embedding_text == "Title-only historical paper"
    assert second.topic == ""
    assert second.is_oa is False


def test_reconstruct_abstract_orders_positions_and_ignores_bad_values():
    assert reconstruct_abstract({"world": [1], "hello": [0], "bad": "x"}) == "hello world"
    assert reconstruct_abstract(None) == ""


def test_benchmark_title_seeds_are_usable_and_unique():
    papers = [parse_work(work) for work in iter_benchmark_seed_works()]

    assert all(paper is not None for paper in papers)
    assert len({paper.openalex_id for paper in papers if paper}) == len(papers)
    assert any(paper and paper.title == "Attention Is All You Need" for paper in papers)


def test_benchmark_seeds_carry_full_card_metadata():
    """Seed probes must render as complete result cards, not title-only stubs."""
    papers = [parse_work(work) for work in iter_benchmark_seed_works()]

    for paper in papers:
        assert paper is not None
        assert paper.snippet, f"{paper.openalex_id} is missing an abstract snippet"
        assert paper.authors, f"{paper.openalex_id} is missing authors"
        assert paper.publication_year > 0, f"{paper.openalex_id} is missing a year"
        assert paper.view_url.startswith("https://"), f"{paper.openalex_id} lacks a link"


def test_requires_usable_openalex_id_and_title():
    assert parse_work({"id": "https://openalex.org/W1", "title": ""}) is None
    assert parse_work({"id": "not-openalex", "title": "Paper"}) is None
    expansion = {
        "id": "https://openalex.org/W99",
        "title": "Expansion work",
        "is_xpac": True,
    }
    assert parse_work(expansion) is None
    assert parse_work(expansion, include_expansion=True) is not None
