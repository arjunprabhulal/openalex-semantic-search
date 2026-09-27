"""Hand-verified records for landmark works missing from a generation.

A generation is immutable, so a work that was absent from the snapshot (BERT's
NAACL record, W2963341956, is not in the full generation) cannot be added without a
rebuild. This file supplies a few such works at query time: the engine embeds
them once at startup with the embedder it already serves queries with, and
merges them into exact-title, semantic, sorted and filtered results and into
the work id lookup.

The file is deliberately small and strict:

* At most ``MAX_SUPPLEMENT_RECORDS`` records, each embedded at startup on CPU.
* Every record names the sources its metadata was checked against and the
  date it was checked (``sources``, ``verified_at``). A record without them is
  refused: nothing unverified reaches a card.
* A malformed or missing file is logged and the service runs without it. A
  supplement only adds results, so serving without it is the pre-existing
  behaviour, not a silently wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any

from .overrides import work_id
from .query_keys import title_key_variants
from .store import normalize_title

logger = logging.getLogger(__name__)

SUPPLEMENT_FORMAT_VERSION = 1
MAX_SUPPLEMENT_RECORDS = 500
# The build embeds "title. abstract" truncated to this many characters.
EMBEDDING_TEXT_CHARS = 2_000

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_REQUIRED_STRINGS = ("openalex_id", "title", "verified_at")
_OPTIONAL_STRINGS = (
    "snippet",
    "doi",
    "venue",
    "work_type",
    "topic",
    "field",
    "oa_url",
    "landing_url",
    "publication_date",
)


@dataclass(frozen=True, slots=True)
class SupplementRecords:
    """Verified card records keyed by bare OpenAlex work id, in file order."""

    records: tuple[dict[str, Any], ...]
    source: Path | None = None

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def empty(cls) -> "SupplementRecords":
        return cls(records=())

    @classmethod
    def load(cls, path: Path | None) -> "SupplementRecords":
        """Read and validate ``path``; raise ValueError or OSError if unusable."""
        if path is None:
            return cls.empty()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        version = payload.get("format_version")
        if version != SUPPLEMENT_FORMAT_VERSION:
            raise ValueError(
                f"{path} declares format_version {version!r}; "
                f"this build reads {SUPPLEMENT_FORMAT_VERSION}"
            )
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ValueError(f"{path} must contain a 'records' list")
        if len(raw_records) > MAX_SUPPLEMENT_RECORDS:
            raise ValueError(
                f"{path} holds {len(raw_records)} records; the limit is "
                f"{MAX_SUPPLEMENT_RECORDS}"
            )
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for position, raw in enumerate(raw_records):
            record = _validate(raw, f"{path} record {position}")
            identifier = work_id(record["openalex_id"])
            if identifier in seen:
                raise ValueError(f"{path}: {identifier} appears twice")
            seen.add(identifier)
            records.append(record)
        logger.info("loaded %d supplement records from %s", len(records), path)
        return cls(records=tuple(records), source=Path(path))

    @classmethod
    def load_or_empty(cls, path: Path | None) -> "SupplementRecords":
        """Like ``load``, but log a failure and serve without the supplement."""
        try:
            return cls.load(path)
        except (OSError, ValueError, TypeError) as error:
            logger.error("supplement records not loaded, serving without them: %s", error)
            return cls.empty()


def _validate(raw: object, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object")
    for name in _REQUIRED_STRINGS:
        if not isinstance(raw.get(name), str) or not raw[name].strip():
            raise ValueError(f"{where} needs a non-empty string {name!r}")
    openalex_id = raw["openalex_id"].strip()
    identifier = work_id(openalex_id)
    if not re.fullmatch(r"W\d+", identifier):
        raise ValueError(f"{where}: {openalex_id!r} is not an OpenAlex work id")
    if not _DATE.match(raw["verified_at"]):
        raise ValueError(f"{where}: verified_at must be YYYY-MM-DD")
    sources = raw.get("sources")
    if (
        not isinstance(sources, list)
        or not sources
        or not all(isinstance(url, str) and url.startswith("https://") for url in sources)
    ):
        raise ValueError(f"{where} needs 'sources': the https URLs its metadata was checked against")
    authors = raw.get("authors", [])
    if not isinstance(authors, list) or not all(isinstance(name, str) for name in authors):
        raise ValueError(f"{where}: authors must be a list of strings")
    year = raw.get("publication_year")
    citations = raw.get("cited_by_count", 0)
    if isinstance(year, bool) or not isinstance(year, int) or not 0 <= year <= 2100:
        raise ValueError(f"{where}: publication_year must be an integer year")
    if isinstance(citations, bool) or not isinstance(citations, int) or citations < 0:
        raise ValueError(f"{where}: cited_by_count must be a non-negative integer")
    for name in _OPTIONAL_STRINGS:
        if not isinstance(raw.get(name, ""), str):
            raise ValueError(f"{where}: {name} must be a string")
    if not isinstance(raw.get("is_oa", False), bool):
        raise ValueError(f"{where}: is_oa must be true or false")

    title = " ".join(raw["title"].split())
    record: dict[str, Any] = {
        "openalex_id": f"https://openalex.org/{identifier}",
        "title": title,
        "authors": [" ".join(name.split()) for name in authors][:8],
        "publication_year": year,
        "cited_by_count": citations,
        "is_oa": bool(raw.get("is_oa", False)),
        "normalized_title": normalize_title(title),
        "supplement": True,
        "sources": list(sources),
        "verified_at": raw["verified_at"],
    }
    for name in _OPTIONAL_STRINGS:
        record[name] = " ".join(str(raw.get(name, "")).split())
    return record


def embedding_text(record: dict[str, Any]) -> str:
    """The same "title. abstract" text the build embeds for a work."""
    snippet = record.get("snippet") or ""
    text = f"{record['title']}. {snippet}" if snippet else record["title"]
    return text[:EMBEDDING_TEXT_CHARS]


def title_keys(record: dict[str, Any]) -> set[str]:
    """Normalized keys a typed title could reach this record through."""
    return set(title_key_variants(record["title"])) | {record["normalized_title"]}
