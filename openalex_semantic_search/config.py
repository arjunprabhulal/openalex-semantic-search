from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Stage:
    name: str
    records: int
    ivf_lists: int
    nprobe: int
    candidate_count: int = 1_000
    pq_subquantizers: int = 32
    pq_bits: int = 8


STAGES: dict[str, Stage] = {
    "10k": Stage("10k", 10_000, 128, 16),
    "100k": Stage("100k", 100_000, 512, 32),
    # Measured on the newest-first 1M rich corpus: nprobe=64 produced
    # recall@20 mean/min 0.91/0.75; 192 produced 0.97/0.95 with 1,000
    # candidates and negligible FAISS search overhead.
    "1m": Stage("1m", 1_000_000, 4_096, 192),
    "10m": Stage("10m", 10_000_000, 16_384, 96),
    "25m": Stage("25m", 25_000_000, 32_768, 128),
    "50m": Stage("50m", 50_000_000, 65_536, 160),
    "full": Stage("full", 340_800_000, 131_072, 192),
}

STAGE_ORDER = tuple(STAGES)
VECTOR_DIMENSION = 384
MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Design decision 2026-08-31: production is
# the FULL corpus - 340.8M usable works (510,372,821 raw snapshot records x
# 0.668 sampled usable ratio) - in the rich uncompressed layout at a measured
# 1,750 B/record (~596GB), as a single copy with no rollback generation and
# no backup, refreshed quarterly via in-place deltas only.
FULL_CORPUS_RECORDS = 340_800_000
# Design decision 2026-09-01: provision at least 1TB usable storage on the
# serving host and allow 800GB for the rich production generation.
# The remaining capacity is reserved for the OS, WAL/temp files, logs, and
# assembly/transfer headroom. The gate fails if production data alone exceeds
# this ceiling.
STORAGE_BUDGET_BYTES = 800_000_000_000
# Design decision 2026-09-02: production service-side SLO. Public-internet
# round-trip time is measured separately from the search service itself.
QUERY_BUDGET_MS = 200.0


# Ceiling on the probe count a filtered query may widen to. Probing is read
# time only: the trained index, its lists and the vectors are unchanged.
MAX_FILTER_NPROBE = 1_024


@dataclass(frozen=True, slots=True)
class SearchTuning:
    """Read-time retrieval settings, per request class.

    The probe count and candidate count were tuned on the 1M stage and carried
    to the full corpus unmeasured. Both are query-time parameters, so they are
    settable here without touching the index. ``None`` means "use what the
    generation manifest records", so a deploy that sets nothing serves exactly
    what the index was built with; change them only after running
    the sweep in ``benchmark/tune_search.py``.
    """

    # Unfiltered queries.
    nprobe: int | None = None
    candidates: int | None = None
    # Filtered queries that the first pass under-fills. None: 4x nprobe capped
    # at MAX_FILTER_NPROBE, and 2x candidates.
    filter_nprobe: int | None = None
    filter_candidates: int | None = None
    # Filters matching at most this many rows are scored exactly instead of
    # through the candidate index (about 100us per row on the serving host).
    exact_filter_rows: int = 10_000
    # Leading ranked rows whose metadata is read for duplicate collapse and
    # the title-only penalty. Constant, so every page of a query sees the same
    # ordering.
    rerank_window: int = 40

    _ENVIRONMENT = {
        "nprobe": "OPENALEX_SEARCH_NPROBE",
        "candidates": "OPENALEX_SEARCH_CANDIDATES",
        "filter_nprobe": "OPENALEX_SEARCH_FILTER_NPROBE",
        "filter_candidates": "OPENALEX_SEARCH_FILTER_CANDIDATES",
        "exact_filter_rows": "OPENALEX_SEARCH_EXACT_FILTER_ROWS",
        "rerank_window": "OPENALEX_SEARCH_RERANK_WINDOW",
    }

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "SearchTuning":
        environ = os.environ if environ is None else environ
        values: dict[str, int] = {}
        for name, variable in cls._ENVIRONMENT.items():
            raw = (environ.get(variable) or "").strip()
            if not raw:
                continue
            try:
                value = int(raw)
            except ValueError as error:
                raise ValueError(f"{variable} must be an integer, got {raw!r}") from error
            if value < (0 if name in ("exact_filter_rows", "rerank_window") else 1):
                raise ValueError(f"{variable} must be positive, got {value}")
            values[name] = value
        return cls(**values)


def get_stage(name: str) -> Stage:
    try:
        return STAGES[name.lower()]
    except KeyError as error:
        choices = ", ".join(STAGE_ORDER)
        raise ValueError(f"Unknown stage {name!r}; choose one of: {choices}") from error


def require_stage_confirmation(stage: Stage, confirmation: str | None) -> None:
    """Require an explicit token above 10K to prevent accidental large builds."""
    if stage.name == "10k":
        return
    expected = f"BUILD_{stage.name.upper()}"
    if confirmation != expected:
        raise ValueError(
            f"Stage {stage.name} requires --confirm {expected}. "
            "Promote only after the previous benchmark passes."
        )
