"""Query-side spellings of a title key, for the existing title and prefix tables.

The exact-title and prefix tables are keyed by ``normalize_title`` of the
stored title, and that function is frozen: changing it would change every key
already on disk. It is also literal about punctuation that readers type
inconsistently:

* An ASCII hyphen stays inside its token ("pre-training"), while U+2010-U+2015
  and U+2212 are not word characters and split it ("pre training"). A stored
  "Pre‐training" therefore misses a typed "Pre-training", and a stored
  "Retrieval-Augmented" misses a typed "Retrieval Augmented".
* Apostrophes split a word ("don't" -> "don t"); U+02BC does not.
* There is no Unicode normalisation, so a decomposed accent drops its mark
  while a precomposed one keeps it, and markup such as ``<i>`` is kept as words.

Rather than change the stored keys, this module produces the handful of keys
the build could have produced for what the reader typed, and each is looked up
in the unchanged tables. Every key it returns is already in normalized form.
"""

from __future__ import annotations

import html
import re
from typing import Callable
import unicodedata

from .store import normalize_title
from .title_prefix import MAX_PREFIX_WORDS, MIN_PREFIX_WORDS

MAX_TITLE_KEY_VARIANTS = 12

# Every dash the build-time tokenizer treats as a separator. They all
# normalize to the same key, so one substitution covers the set.
_TYPOGRAPHIC_DASHES = re.compile("[‐‑‒–—―−]")
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]{0,15}(?:\s[^<>]{0,200})?/?>")
_APOSTROPHES = re.compile("['’‘ʼ`]")


def _fold_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def title_key_variants(query: str, *, limit: int = MAX_TITLE_KEY_VARIANTS) -> list[str]:
    """Distinct normalized title keys for ``query``, most literal first.

    The first key is always ``normalize_title(query)``, so an exact match on
    what was typed is never displaced by a variant.
    """
    base = normalize_title(query)
    if not base:
        return []
    keys: list[str] = []

    def add(value: str) -> None:
        key = normalize_title(value)
        if key and key not in keys:
            keys.append(key)

    def add_hyphen_forms(key: str) -> None:
        if "-" in key:
            # Stored with a typographic dash (or typed without the hyphen).
            add(key.replace("-", " "))
            # Stored as one word: "pretraining".
            add(key.replace("-", ""))

    add(query)
    add_hyphen_forms(base)
    # Typed with a typographic dash, stored with an ASCII hyphen.
    if _TYPOGRAPHIC_DASHES.search(query):
        add(_TYPOGRAPHIC_DASHES.sub("-", query))
    unescaped = html.unescape(query)
    forms = [
        unicodedata.normalize("NFC", query),
        unicodedata.normalize("NFKC", query),
        # Pasted with markup or entities from a publisher page.
        _TAG.sub(" ", unescaped),
        _fold_accents(unicodedata.normalize("NFKC", query)),
        # "Parkinson's" stored without the apostrophe.
        _APOSTROPHES.sub("", query),
        unicodedata.normalize("NFD", query),
    ]
    if "&" in query and "&amp;" not in query:
        # Titles stored with a literal entity normalize to the word "amp".
        forms.append(query.replace("&", " &amp; "))
    for form in forms:
        add(form)
    for key in list(keys[1:]):
        add_hyphen_forms(key)
    return keys[:limit]


def hyphenation_candidates(
    key: str,
    *,
    prefix_exists: Callable[[str], bool] | None,
    max_states: int = 8,
    max_probes: int = 64,
    max_words: int = 24,
) -> list[str]:
    """Keys that rejoin typed words with ASCII hyphens, pruned by the prefix table.

    "retrieval augmented generation for knowledge intensive nlp tasks" was
    stored as "retrieval-augmented generation for knowledge-intensive nlp
    tasks". Each gap between typed words is either a space or a hyphen, so the
    candidates grow as 2^gaps; the prefix table bounds that. Every leading run
    of words that a stored title begins with is in it, so a spelling whose
    completed leading words are not is discarded as soon as it has three.

    Returns only spellings that differ from ``key``, fewest hyphens first. The
    caller confirms each against the exact or prefix table before using it.
    Without a prefix table there is nothing to prune with, so only single
    hyphen insertions are offered.
    """
    words = key.split()
    if not 2 <= len(words) <= max_words:
        return []
    if prefix_exists is None:
        return [
            " ".join([*words[:gap], words[gap] + "-" + words[gap + 1], *words[gap + 2 :]])
            for gap in range(len(words) - 1)
        ][:max_states]

    probes = 0
    known: dict[str, bool] = {}

    def begins_a_title(tokens: tuple[str, ...]) -> bool:
        nonlocal probes
        text = " ".join(tokens)
        if text not in known:
            if probes >= max_probes:
                return False
            probes += 1
            known[text] = prefix_exists(text)
        return known[text]

    states: list[tuple[str, ...]] = [(words[0],)]
    for word in words[1:]:
        children: list[tuple[str, ...]] = []
        for tokens in states:
            children.append((*tokens, word))
            children.append((*tokens[:-1], tokens[-1] + "-" + word))
        survivors = []
        for tokens in children:
            # All but the last token are final: a later hyphen can only extend
            # the last one. A stored title longer than those final tokens puts
            # them in the prefix table.
            final = tokens[:-1]
            if MIN_PREFIX_WORDS <= len(final) <= MAX_PREFIX_WORDS and not begins_a_title(final):
                continue
            survivors.append(tokens)
        # Fewest hyphens first: closest to what was typed.
        survivors.sort(key=lambda tokens: -len(tokens))
        states = survivors[:max_states]
        if not states:
            return []
    return [" ".join(tokens) for tokens in states if len(tokens) != len(words)]
