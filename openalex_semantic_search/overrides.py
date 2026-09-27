"""Hand-verified corrections applied over an immutable generation.

A generation is built once and never rewritten, so a record that OpenAlex
publishes incorrectly cannot be repaired without a full rebuild. This layer
reads a small JSON file that lives outside the generation directory and
patches display metadata for named works at fetch time.

The overrides are deliberately narrow:

* Only card fields may be replaced. Identity (``row_id``, ``openalex_id``)
  and the embedding are never touched, so retrieval is unchanged.
* The sidecar files are never rewritten. The engine maps each corrected work
  to its row at startup (through the work id index) and uses the corrected
  year, citations and flags for ranking, sorting and filtering, so a
  corrected year both shows on the card and decides a year filter. A work it
  cannot map is logged and patches the card only.
* ``cited_by_count_unreliable: true`` marks a count that belongs to another
  work. The card keeps the number; ranking, sorting and ``min_citations``
  use zero (``effective_citations``).
* An unknown work id is ignored rather than rejected, so an override file may
  outlive the generation it was written against.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OVERRIDE_FORMAT_VERSION = 1

CITATION_UNRELIABLE = "cited_by_count_unreliable"

# Card fields a correction may replace. Identity, ranking inputs consumed from
# the fixed-width sidecars, and the normalized title used for exact lookup are
# excluded on purpose.
OVERRIDABLE_FIELDS = frozenset(
    {
        "title",
        "snippet",
        "authors",
        "publication_year",
        "publication_date",
        "doi",
        "venue",
        "work_type",
        "topic",
        "field",
        "is_oa",
        "oa_url",
        "landing_url",
        "cited_by_count",
        # Marks a count proven to belong to another work (an upstream merge).
        # The card keeps the published number; ranking, sorting and the
        # min_citations filter treat the work as uncited.
        CITATION_UNRELIABLE,
    }
)


def effective_citations(record: dict) -> int:
    """The citation count ranking and filters should use for ``record``."""
    if record.get(CITATION_UNRELIABLE):
        return 0
    return int(record.get("cited_by_count") or 0)


def work_id(openalex_id: str) -> str:
    """Return the bare ``W...`` identifier for a full OpenAlex URL."""
    return openalex_id.rsplit("/", 1)[-1] if openalex_id else ""


@dataclass(frozen=True, slots=True)
class MetadataOverrides:
    """Verified per-work corrections, keyed by bare OpenAlex work id."""

    entries: dict[str, dict[str, Any]]
    source: Path | None = None

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def empty(cls) -> "MetadataOverrides":
        return cls(entries={})

    @classmethod
    def load(cls, path: Path | None) -> "MetadataOverrides":
        if path is None:
            return cls.empty()
        if not path.exists():
            raise FileNotFoundError(f"Override file {path} does not exist")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object")
        version = int(payload.get("format_version", -1))
        if version != OVERRIDE_FORMAT_VERSION:
            raise ValueError(
                f"{path} declares format_version {version}; "
                f"this build reads {OVERRIDE_FORMAT_VERSION}"
            )
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, dict):
            raise ValueError(f"{path} must contain an 'entries' object")

        entries: dict[str, dict[str, Any]] = {}
        for key, value in raw_entries.items():
            identifier = work_id(str(key))
            if not identifier.startswith("W"):
                raise ValueError(f"{path}: {key!r} is not an OpenAlex work id")
            if not isinstance(value, dict):
                raise ValueError(f"{path}: entry {identifier} must be an object")
            fields = {
                name: field_value
                for name, field_value in value.items()
                # Documentation keys such as "note" or "verified_at" travel
                # with the entry; only known card fields are ever applied.
                if name in OVERRIDABLE_FIELDS
            }
            if CITATION_UNRELIABLE in fields and not isinstance(
                fields[CITATION_UNRELIABLE], bool
            ):
                raise ValueError(
                    f"{path}: entry {identifier} {CITATION_UNRELIABLE} must be true or false"
                )
            if not fields:
                raise ValueError(
                    f"{path}: entry {identifier} replaces no overridable field"
                )
            entries[identifier] = fields
        logger.info("loaded %d metadata overrides from %s", len(entries), path)
        return cls(entries=entries, source=path)

    def apply(self, record: dict) -> dict:
        """Return ``record`` with any verified correction merged in.

        The input mapping is never mutated, so a cached compact block stays
        byte-faithful to the generation on disk.
        """
        if not self.entries:
            return record
        fields = self.entries.get(work_id(str(record.get("openalex_id") or "")))
        if not fields:
            return record
        patched = dict(record)
        patched.update(fields)
        patched["overridden_fields"] = sorted(fields)
        return patched
