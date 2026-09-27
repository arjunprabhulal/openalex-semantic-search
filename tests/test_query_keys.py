import pytest

from conftest import build_production_shaped, filler, work
from openalex_semantic_search.compact_store import CompactMetadataStore
from openalex_semantic_search.query_keys import (
    MAX_TITLE_KEY_VARIANTS,
    hyphenation_candidates,
    title_key_variants,
)
from openalex_semantic_search.store import Filters, normalize_title


def test_the_typed_key_comes_first_and_keys_are_normalized_and_distinct():
    keys = title_key_variants("BERT: Pre-training of Deep Bidirectional Transformers")
    assert keys[0] == normalize_title("BERT: Pre-training of Deep Bidirectional Transformers")
    assert len(keys) == len(set(keys)) <= MAX_TITLE_KEY_VARIANTS
    assert all(normalize_title(key) == key for key in keys)


@pytest.mark.parametrize(
    "typed, stored",
    [
        # ASCII typed; stored with a typographic hyphen, which splits the word.
        ("Pre-training of transformers", "Pre‐training of transformers"),
        ("Pre-training of transformers", "Pre‑training of transformers"),
        ("Pre-training of transformers", "Pre–training of transformers"),
        ("Pre-training of transformers", "Pre—training of transformers"),
        ("Pre-training of transformers", "Pre−training of transformers"),
        ("Pre-training of transformers", "Pre‒training of transformers"),
        # Typographic typed; stored with an ASCII hyphen.
        ("Pre‐training of transformers", "Pre-training of transformers"),
        # Stored as one word.
        ("Pre-training of transformers", "Pretraining of transformers"),
        # Colon and other punctuation already drop out of the key.
        ("BERT Pre-training", "BERT: Pre-training"),
        # Decomposed accent typed; stored precomposed, and the reverse.
        ("Zur Elektrodynamik bewegter Körper", "Zur Elektrodynamik bewegter Körper"),
        ("Zur Elektrodynamik bewegter Körper", "Zur Elektrodynamik bewegter Körper"),
        # Accented typed; stored without the accent.
        ("Zur Elektrodynamik bewegter Körper", "Zur Elektrodynamik bewegter Korper"),
        # Full-width letters (NFKC).
        ("ＢＥＲＴ models", "BERT models"),
        # Markup pasted from a publisher page.
        ("<i>E. coli</i> growth", "E. coli growth"),
        ("R&amp;D policy", "R&D policy"),
        # A stored literal entity.
        ("R&D policy", "R&amp;D policy"),
        # Apostrophe variants.
        ("Parkinson’s disease", "Parkinson's disease"),
        ("Parkinson's disease", "Parkinsons disease"),
    ],
)
def test_variants_reach_the_key_the_build_stored(typed, stored):
    assert normalize_title(stored) in title_key_variants(typed)


def test_hyphenation_is_pruned_by_the_prefix_table():
    stored = "retrieval-augmented generation for knowledge-intensive nlp tasks".split()
    prefixes = {" ".join(stored[:count]) for count in range(3, len(stored))}
    probes: list[str] = []

    def exists(text):
        probes.append(text)
        return text in prefixes

    candidates = hyphenation_candidates(
        "retrieval augmented generation for knowledge intensive nlp tasks", prefix_exists=exists
    )
    assert candidates[0] == " ".join(stored)
    assert len(probes) <= 64
    # A partial title rejoins the same way.
    partial = hyphenation_candidates(
        "retrieval augmented generation for knowledge intensive nlp", prefix_exists=exists
    )
    assert partial[0] == " ".join(stored[:5])


def test_hyphenation_is_bounded_and_skips_what_was_typed():
    assert hyphenation_candidates("single", prefix_exists=lambda _: True) == []
    words = " ".join(f"w{index}" for index in range(30))
    assert hyphenation_candidates(words, prefix_exists=lambda _: True) == []
    calls = []
    hyphenation_candidates(
        " ".join(f"w{index}" for index in range(16)),
        prefix_exists=lambda text: calls.append(text) or True,
        max_probes=10,
    )
    assert len(calls) <= 10
    without_table = hyphenation_candidates("pre training of transformers", prefix_exists=None)
    assert "pre-training of transformers" in without_table
    assert "pre training of transformers" not in without_table


def test_exact_title_verification_is_bounded_for_generic_titles(tmp_path, monkeypatch):
    papers = [filler(index) for index in range(700)]
    papers += [
        work(f"WINTRO{index}", "Introduction", year=2000, citations=index) for index in range(600)
    ]
    generation = build_production_shaped(tmp_path, papers, candidate_count=50)
    store = CompactMetadataStore(generation)
    fetched: list[int] = []
    original = store.fetch

    def counting(ids):
        fetched.extend(ids)
        return original(ids)

    monkeypatch.setattr(store, "fetch", counting)
    try:
        rows = store.title_search("Introduction", Filters(), limit=100)
        assert len(rows) == 100
        # The most cited 100 of 600, verified without reading all 600.
        citations = [int(store.citations[row]) for row in rows]
        assert citations == sorted(citations, reverse=True)
        assert min(citations) == 500
        assert len(fetched) <= 128
        assert store.title_key_exists("introduction")
        assert not store.title_key_exists("introductions")
    finally:
        store.close()
