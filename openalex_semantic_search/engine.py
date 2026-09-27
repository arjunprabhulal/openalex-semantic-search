from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import logging
import math
from pathlib import Path
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

SORT_ORDERS = ("relevance", "most_cited", "newest", "oldest")

# Title and prefix rows retrieved per query, constant so pagination is
# stable. Only binds when a query has many same-title matches, which is
# rare; the cost is a bounded metadata verification inside title_search.
TITLE_CANDIDATE_LIMIT = 100

# Prefix rows kept per query, chosen by citations from up to PREFIX_SCAN_ROWS
# stored rows. Storage order is newest-updated first, so taking the first rows
# kept an arbitrary recent slice of a common opening and dropped the landmark.
PREFIX_CANDIDATE_LIMIT = 50
PREFIX_SCAN_ROWS = 50_000

# Ranking weights, in raw cosine units at REFERENCE_SPREAD. The boosts are
# rescaled by how widely a query's candidate scores spread, so a query whose
# neighbours sit within 0.02 of each other and one spread over 0.2 get the same
# relative pull from citations, recency and a title prefix.
REFERENCE_SPREAD = 0.10
CITATION_WEIGHT = 0.15
# log1p(100,000): a paper with 100k citations receives the full weight.
CITATION_SATURATION = math.log1p(100_000)
RECENCY_WEIGHT = 0.03
PREFIX_TITLE_BOOST = 0.15
# A record embedded from its title alone scores high against short queries on
# wording, not meaning; keep it from outranking a full record by that alone.
TITLE_ONLY_PENALTY = 0.03
# most_cited/newest/oldest order only rows within this fraction of the
# candidate score spread from the best one (plus exact titles); weaker matches
# are left out. Without a floor, "most cited" for a title surfaces famous
# unrelated works that merely landed in the candidate pool, and appending them
# after the sorted rows would restart the order part-way down the list.
SORT_RELEVANCE_FLOOR = 0.5
# Hyphenated spellings of the typed words probed when no title matched.
HYPHENATION_CANDIDATES = 8
HYPHENATION_PROBES = 48

# Ceiling on how far the candidate request is scaled up for a selective
# filter. Measured on the 315M index, the candidate search is nearly flat in
# k: 11ms at 1,000, 20ms at 64,000, 56ms at 256,000. Asking for far more
# candidates is therefore much cheaper than materializing eligible rows by
# hand, which costs a steady 100us per row.
MAX_FILTER_MULTIPLIER = 512

from .config import MAX_FILTER_NPROBE, VECTOR_DIMENSION, SearchTuning
from .compact_store import CompactMetadataStore, stable_text_hash
from .embeddings import Embedder
from .overrides import CITATION_UNRELIABLE, MetadataOverrides, effective_citations, work_id
from .quantization import int8_scores
from .query_keys import _fold_accents, hyphenation_candidates, title_key_variants
from .store import Filters, MetadataStore, normalize_title
from .supplement import SupplementRecords, embedding_text
from .supplement import title_keys as supplement_title_keys
from .title_prefix import MAX_PREFIX_WORDS, MIN_PREFIX_WORDS, TitlePrefixIndex
from .work_index import WorkIdIndex, normalize_work_id
from .vector_index import FaissPQIndex, NumpyFlatIndex


@dataclass(frozen=True, slots=True)
class SearchResult:
    row_id: int
    openalex_id: str
    work_id: str
    title: str
    snippet: str
    authors: list[str]
    publication_year: int
    doi: str
    cited_by_count: int
    topic: str
    field: str
    is_oa: bool
    oa_url: str
    work_type: str
    venue: str
    publication_date: str
    landing_url: str
    openalex_url: str
    view_url: str
    semantic_score: float
    ranking_score: float
    title_match: bool
    overridden_fields: list[str] | None = None
    # The published count belongs to another work (an upstream merge). It is
    # shown, but ranked, sorted and filtered as if the work had no citations.
    citation_count_unreliable: bool = False
    # A hand-verified record served from the supplement file because the
    # generation does not hold this work. row_id is then not a generation row.
    supplement: bool = False


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    corpus_records: int
    candidates: int
    filtered_records: int
    elapsed_ms: float
    results: list[SearchResult]
    offset: int = 0
    page_size: int = 0
    total_matches: int = 0
    has_more: bool = False
    sort: str = "relevance"
    timings_ms: dict[str, float] | None = None
    generation_id: str = ""
    snapshot_at: str = ""
    # total_matches counts ranked candidates, not corpus matches. The candidate
    # generator is bounded, so a broad query saturates it and the true number of
    # matching records is unknown and far larger. Consumers must not present a
    # saturated count as a result total.
    total_matches_capped: bool = False
    # Rank position after the last row this page examined. Equal to
    # offset + len(results) unless a row failed the final metadata recheck, in
    # which case offset + len(results) would hand the next page a row already
    # consumed here.
    cursor: int | None = None

    def next_offset(self) -> int:
        """Offset a client should request next; the current one when done."""
        if not self.has_more:
            return self.offset
        return self.cursor if self.cursor is not None else self.offset + len(self.results)

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "corpus_records": self.corpus_records,
            "candidates": self.candidates,
            "filtered_records": self.filtered_records,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "offset": self.offset,
            "page_size": self.page_size,
            "total_matches": self.total_matches,
            "total_matches_capped": self.total_matches_capped,
            "has_more": self.has_more,
            "next_offset": self.next_offset() if self.has_more else None,
            "sort": self.sort,
            "generation_id": self.generation_id,
            "snapshot_at": self.snapshot_at,
            "timings_ms": {
                name: round(value, 3)
                for name, value in (self.timings_ms or {}).items()
            },
            "results": [asdict(result) for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class _RankedCandidate:
    row_id: int
    semantic_score: float
    ranking_score: float
    cited_by_count: int
    publication_year: int
    title_match: bool
    exact_title_match: bool
    prefix_match: bool = False


class SearchEngine:
    def __init__(
        self,
        generation: Path,
        embedder: Embedder,
        *,
        selective_filter_threshold: int | None = None,
        overrides: MetadataOverrides | None = None,
        title_prefixes: TitlePrefixIndex | None = None,
        work_ids: WorkIdIndex | None = None,
        tuning: SearchTuning | None = None,
        supplement: SupplementRecords | None = None,
    ):
        self.generation = generation
        self.embedder = embedder
        self.overrides = overrides or MetadataOverrides.empty()
        self.manifest = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
        if int(self.manifest["dimension"]) != VECTOR_DIMENSION:
            raise RuntimeError("Generation vector dimension does not match the search service")
        self.record_count = int(self.manifest["records"])
        self.tuning = tuning or SearchTuning()
        self.candidate_count = int(self.tuning.candidates or self.manifest["candidate_count"])
        self.filter_candidate_count = int(
            self.tuning.filter_candidates or 2 * self.candidate_count
        )
        self.nprobe = int(self.tuning.nprobe or self.manifest.get("nprobe") or 1)
        self.filter_nprobe = int(
            self.tuning.filter_nprobe
            or max(self.nprobe, min(4 * self.nprobe, MAX_FILTER_NPROBE))
        )
        self.rerank_window = int(self.tuning.rerank_window)
        # Provenance travels on every search response: the only public route
        # into this service is POST /search, so a consumer has no other way to
        # learn how old the corpus is.
        self.generation_id = generation.name
        provenance = self.manifest.get("backfill_provenance")
        snapshot_meta = (
            provenance.get("snapshot_meta") if isinstance(provenance, dict) else None
        )
        self.snapshot_at = (
            str(snapshot_meta.get("manifest_date") or "")
            if isinstance(snapshot_meta, dict)
            else ""
        )
        backend = self.manifest["backend"]
        if backend == "faiss":
            self.index = FaissPQIndex.load(generation, self.nprobe)
            indexed_records = int(self.index.index.ntotal)
        elif backend == "numpy":
            self.index = NumpyFlatIndex.load(generation)
            indexed_records = len(self.index.vectors)
        else:
            raise RuntimeError(f"Unsupported generation backend {backend!r}")
        if indexed_records != self.record_count:
            raise RuntimeError(
                f"Candidate index has {indexed_records:,} rows; manifest promises "
                f"{self.record_count:,}"
            )
        self.codes = np.memmap(
            generation / "int8-vectors.bin",
            mode="r",
            dtype=np.int8,
            shape=(self.record_count, VECTOR_DIMENSION),
        )
        self.int8_scales = np.load(generation / "int8-scales.npy")
        if self.int8_scales.shape != (VECTOR_DIMENSION,):
            raise RuntimeError(
                f"INT8 scales have shape {self.int8_scales.shape}; expected "
                f"({VECTOR_DIMENSION},)"
            )
        metadata_backend = self.manifest.get("metadata_backend", "sqlite")
        if metadata_backend == "compact":
            self.store = CompactMetadataStore(generation)
        elif metadata_backend == "sqlite":
            self.store = MetadataStore(generation / "metadata.sqlite3", read_only=True)
        else:
            raise RuntimeError(f"Unsupported metadata backend {metadata_backend!r}")
        self.metadata_backend = metadata_backend
        self.selective_filter_threshold = (
            selective_filter_threshold
            if selective_filter_threshold is not None
            else self.tuning.exact_filter_rows
        )
        self.publication_years = self._load_optional_feature(
            "ranking-years.npy", np.int16
        )
        self.citation_counts = self._load_optional_feature(
            "ranking-citations.npy", np.uint32
        )
        self._embed_lock = threading.Lock()
        self.title_prefixes = (
            title_prefixes
            if title_prefixes is not None
            and self._index_matches_generation("title prefix", title_prefixes)
            else None
        )
        self.work_ids = (
            work_ids
            if work_ids is not None and self._index_matches_generation("work id", work_ids)
            else None
        )
        recorded_embedder = str(self.manifest.get("embedder") or "")
        self.embedder_matches_generation = (
            not recorded_embedder or recorded_embedder == getattr(embedder, "name", "")
        )
        if not self.embedder_matches_generation:
            # A warning, not a refusal: the name recorded at build time may be
            # spelled differently from the serving model id for the same model.
            logger.warning(
                "query embedder %r differs from the generation's embedder %r; "
                "semantic scores are meaningless if these are different models",
                getattr(embedder, "name", ""),
                recorded_embedder,
            )
        self._patched_rows, self._override_titles = self._resolve_override_rows()
        self._supplement_rows: dict[int, dict] = {}
        self._supplement_by_work: dict[str, int] = {}
        self._supplement_by_key: dict[str, list[int]] = {}
        self._supplement_ids: list[int] = []
        self._supplement_vectors = np.empty((0, VECTOR_DIMENSION), dtype=np.float32)
        if supplement is not None and len(supplement):
            try:
                self._load_supplement(supplement)
            except Exception:  # A supplement only adds results; serve without it.
                logger.exception("supplement records could not be prepared; serving without them")
                self._supplement_rows, self._supplement_by_work = {}, {}
                self._supplement_by_key, self._supplement_ids = {}, []
                self._supplement_vectors = np.empty((0, VECTOR_DIMENSION), dtype=np.float32)

    def _load_supplement(self, supplement: SupplementRecords) -> None:
        """Embed supplement records and give each an id past the last row.

        The ids sit at record_count and above, so they can never index the
        sidecars, the INT8 codes or the metadata store; every path that reads
        those checks ``_is_supplement`` first. A work the generation already
        holds is skipped: the generation copy (and any correction of it) wins.
        """
        kept: list[dict] = []
        for record in supplement.records:
            identifier = work_id(record["openalex_id"])
            if self._generation_has_work(identifier):
                logger.warning(
                    "supplement record %s is already in the generation; skipped", identifier
                )
                continue
            kept.append(record)
        if not kept:
            return
        with self._embed_lock:
            vectors = np.asarray(
                self.embedder.encode_documents([embedding_text(r) for r in kept]),
                dtype=np.float32,
            )
        if vectors.shape != (len(kept), VECTOR_DIMENSION):
            raise RuntimeError(f"supplement embeddings have shape {vectors.shape}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
        for position, record in enumerate(kept):
            row_id = self.record_count + position
            self._supplement_rows[row_id] = record
            self._supplement_by_work[work_id(record["openalex_id"])] = row_id
            for key in supplement_title_keys(record):
                self._supplement_by_key.setdefault(key, []).append(row_id)
        self._supplement_ids = sorted(self._supplement_rows)
        self._supplement_vectors = vectors
        logger.info("serving %d supplement records", len(kept))

    def _generation_has_work(self, identifier: str) -> bool:
        if self.work_ids is not None:
            rows = self.work_ids.candidate_rows(identifier, stable_text_hash)
        elif hasattr(self.store, "row_id_for_openalex_id"):
            row = self.store.row_id_for_openalex_id(f"https://openalex.org/{identifier}")
            rows = [row] if row is not None else []
        else:
            return False
        return any(
            normalize_work_id(str(item.get("openalex_id") or "")) == identifier
            for item in self.store.fetch(rows).values()
        )

    @property
    def supplement_count(self) -> int:
        return len(self._supplement_rows)

    def _is_supplement(self, row_id: int) -> bool:
        return row_id >= self.record_count

    def _fetch(self, ids) -> dict[int, dict]:
        """Store rows plus supplement records, for the ids given."""
        rows = [int(row_id) for row_id in ids]
        stored = self.store.fetch([row_id for row_id in rows if not self._is_supplement(row_id)])
        stored.update(
            (row_id, self._supplement_rows[row_id])
            for row_id in rows
            if row_id in self._supplement_rows
        )
        return stored

    def _supplement_candidates(
        self, query_vector: np.ndarray, keys: set[str], filters: Filters, pool_floor: float | None
    ) -> tuple[list[int], list[int], dict[int, float]]:
        """(exact-title ids, semantic ids, scores) from the supplement.

        A record joins semantically only when it scores at least as well as
        the weakest row the candidate index returned: it would have been in
        the pool had it been indexed, and no further.
        """
        if not self._supplement_rows:
            return [], [], {}
        scores = self._supplement_vectors @ np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(query_vector))
        if norm > 0:
            scores = scores / norm
        score_by_id = {
            row_id: float(scores[index]) for index, row_id in enumerate(self._supplement_ids)
        }
        eligible = {
            row_id
            for row_id, record in self._supplement_rows.items()
            if self._metadata_matches(record, filters)
        }
        exact = list(
            dict.fromkeys(
                row_id
                for key in keys
                for row_id in self._supplement_by_key.get(key, ())
                if row_id in eligible
            )
        )
        semantic = (
            [
                row_id
                for row_id in self._supplement_ids
                if row_id in eligible
                and row_id not in exact
                and score_by_id[row_id] >= pool_floor
            ]
            if pool_floor is not None
            else []
        )
        wanted = set(exact) | set(semantic)
        return exact, semantic, {k: v for k, v in score_by_id.items() if k in wanted}

    def _index_matches_generation(self, label: str, index) -> bool:
        """Refuse a lookup index built for another generation.

        Its row ids would name different works, and every hit would be pinned
        or boosted as a title match. A record-count mismatch is proof and
        disables the index; a differing generation name is only warned about,
        because the build may have read the same generation through another
        path.
        """
        manifest = getattr(index, "manifest", None) or {}
        records = manifest.get("records")
        if records is not None and int(records) != self.record_count:
            logger.warning(
                "%s index at %s was built for %s records; this generation has %s. "
                "Lookup disabled.",
                label,
                getattr(index, "directory", "?"),
                records,
                self.record_count,
            )
            return False
        built_for = manifest.get("generation")
        if built_for and built_for != self.generation_id:
            logger.warning(
                "%s index was built from generation %r; serving %r",
                label,
                built_for,
                self.generation_id,
            )
        return True

    def _resolve_override_rows(self) -> tuple[dict[int, dict], dict[int, str]]:
        """Map each corrected work to its row, with the corrected record.

        Corrections are keyed by OpenAlex id, while ranking, sorting and
        filtering read fixed-width sidecars by row. Resolving the rows once at
        startup lets those paths use the corrected year, citations and flags
        instead of patching only the card a reader sees.
        """
        if not len(self.overrides):
            return {}, {}
        found: dict[int, dict] = {}
        for identifier in self.overrides.entries:
            rows: list[int] = []
            if self.work_ids is not None:
                rows = self.work_ids.candidate_rows(identifier, stable_text_hash)
            elif hasattr(self.store, "row_id_for_openalex_id"):
                row = self.store.row_id_for_openalex_id(f"https://openalex.org/{identifier}")
                rows = [row] if row is not None else []
            for row_id, item in self.store.fetch(rows).items():
                if normalize_work_id(str(item.get("openalex_id") or "")) == identifier:
                    found[int(row_id)] = item
                    break
        missing = len(self.overrides) - len(found)
        if missing:
            logger.warning(
                "%d of %d corrections could not be mapped to a row (no work id "
                "index?); they patch the card only, not ranking or filters",
                missing,
                len(self.overrides),
            )
        return (
            {row: self.overrides.apply(item) for row, item in found.items()},
            {row: str(item["normalized_title"]) for row, item in found.items()},
        )

    def fetch_work(self, work_id: str) -> dict | None:
        """Return one work's rich metadata by OpenAlex id, or None.

        Retrieval never needed this: every other path reaches a row through the
        candidate index or a title hash. A caller holding an id from an earlier
        result does, and it is a direct lookup rather than a search, so no
        embedding or ranking runs.
        """
        normalized = normalize_work_id(work_id)
        supplement_row = self._supplement_by_work.get(normalized) if normalized else None
        if supplement_row is not None:
            return asdict(self._result(supplement_row, self._supplement_rows[supplement_row]))
        if self.work_ids is None:
            raise RuntimeError("This generation was served without a work id index")
        if not normalized:
            return None
        rows = self.work_ids.candidate_rows(normalized, stable_text_hash)
        if not rows:
            return None
        # Confirm the id rather than trusting the hash: a 64-bit collision is
        # vanishingly unlikely, but handing back the wrong paper is not a
        # failure mode worth accepting for a lookup by identifier.
        for row_id, item in self.store.fetch(rows).items():
            patched = self.overrides.apply(item)
            if normalize_work_id(str(patched.get("openalex_id") or "")) == normalized:
                # The same field set as a search result, so a client renders a
                # looked-up work with the code it uses for search hits.
                return asdict(self._result(int(row_id), patched))
        return None

    @staticmethod
    def _result(
        row_id: int,
        item: dict,
        *,
        semantic_score: float = 0.0,
        ranking_score: float = 0.0,
        title_match: bool = False,
    ) -> SearchResult:
        return SearchResult(
            row_id=row_id,
            openalex_id=item["openalex_id"],
            work_id=work_id(item["openalex_id"]),
            title=item["title"],
            snippet=item["snippet"],
            authors=item["authors"],
            publication_year=item["publication_year"],
            doi=item["doi"],
            cited_by_count=item["cited_by_count"],
            topic=item["topic"],
            field=item["field"],
            is_oa=item["is_oa"],
            oa_url=item["oa_url"],
            work_type=item["work_type"],
            venue=item["venue"],
            publication_date=item["publication_date"],
            landing_url=item["landing_url"],
            # The OpenAlex work page always resolves, even when the
            # publisher DOI recorded upstream does not.
            openalex_url=item["openalex_id"],
            view_url=item["doi"]
            or item["landing_url"]
            or item["oa_url"]
            or item["openalex_id"],
            semantic_score=semantic_score,
            ranking_score=ranking_score,
            title_match=title_match,
            overridden_fields=item.get("overridden_fields"),
            citation_count_unreliable=bool(item.get(CITATION_UNRELIABLE)),
            supplement=bool(item.get("supplement")),
        )

    def _load_optional_feature(self, name: str, dtype) -> np.ndarray | None:
        path = self.generation / name
        if not path.exists():
            return None
        values = np.load(path, mmap_mode="r")
        if values.dtype != np.dtype(dtype) or values.shape != (self.record_count,):
            raise RuntimeError(
                f"Invalid {name}: expected {self.record_count} values of dtype "
                f"{np.dtype(dtype)}, got shape {values.shape} dtype {values.dtype}"
            )
        return values

    def close(self) -> None:
        self.store.close()

    def _embed_query(self, query: str) -> np.ndarray:
        # SentenceTransformer model objects are not guaranteed to be safe under
        # concurrent forwards. Keep one loaded CPU model and serialize only the
        # short query-embedding step; FAISS and metadata work remain concurrent.
        with self._embed_lock:
            vector = np.asarray(self.embedder.encode_query(query), dtype=np.float32)
        if vector.shape != (VECTOR_DIMENSION,):
            raise RuntimeError(
                f"Query embedder returned {vector.shape}; expected ({VECTOR_DIMENSION},)"
            )
        return vector

    def _count(self, filters: Filters) -> int:
        """Rows matching ``filters``, with corrected rows judged by their correction."""
        count = self.store.count(filters)
        for row_id, record in self._patched_rows.items():
            stored = bool(self.store.filter_candidate_ids([row_id], filters))
            count += int(self._metadata_matches(record, filters)) - int(stored)
        return max(0, count)

    def _apply_row_filters(self, ids, filters: Filters) -> list[int]:
        """Filter candidate rows on the sidecars, except corrected rows.

        The sidecars hold the values OpenAlex published. For a corrected work
        they are the wrong values -- W2626778328 is stored as 2025 and is 2017
        -- so its correction decides instead, in both directions.
        """
        values = [int(value) for value in ids]
        kept = self.store.filter_candidate_ids(values, filters)
        if not filters.active or not self._patched_rows:
            return kept
        kept_set = set(kept)
        return [
            row_id
            for row_id in values
            if (
                self._metadata_matches(self._patched_rows[row_id], filters)
                if row_id in self._patched_rows
                else row_id in kept_set
            )
        ]

    def _exact_candidates(
        self, query_vector: np.ndarray, filters: Filters, keep: int
    ) -> list[int]:
        ids = [
            row_id
            for row_id in self.store.eligible_ids(filters)
            if row_id not in self._patched_rows
            or self._metadata_matches(self._patched_rows[row_id], filters)
        ]
        ids.extend(
            row_id
            for row_id, record in self._patched_rows.items()
            if row_id not in ids and self._metadata_matches(record, filters)
        )
        if not ids:
            return []
        scores = int8_scores(query_vector, self.codes[ids], self.int8_scales)
        limit = min(keep, len(ids))
        selected = np.argpartition(scores, -limit)[-limit:]
        return sorted(ids[index] for index in selected)

    def _semantic_candidates(
        self,
        query_vector: np.ndarray,
        filters: Filters,
    ) -> tuple[list[int], int]:
        filtered_count = self._count(filters) if filters.active else self.record_count
        if filtered_count == 0:
            return [], 0
        keep = self.filter_candidate_count if filters.active else self.candidate_count

        if filters.active and filtered_count <= self.selective_filter_threshold:
            try:
                return self._exact_candidates(query_vector, filters, keep), filtered_count
            except RuntimeError:
                # The store caps how many rows it will materialize; a count
                # adjusted by corrections can sit just past that cap.
                pass

        # Post-filtering keeps only the eligible share of what the candidate
        # index returned, so a fixed multiplier collapses as a filter gets
        # selective: at 8x, a filter matching 0.04% of the corpus left 3 of
        # 8,000 candidates and the caller saw 85 results out of 135,963
        # matching works. Scale the request by how rare eligible rows are, so
        # roughly candidate_count survive whatever the selectivity.
        multiplier = 1
        if filters.active:
            selectivity = filtered_count / self.record_count
            needed = math.ceil(1.0 / selectivity) if selectivity > 0 else MAX_FILTER_MULTIPLIER
            multiplier = min(MAX_FILTER_MULTIPLIER, max(8, needed))
        requested = min(self.record_count, self.candidate_count * multiplier)
        _, ids = self.index.search(query_vector, requested, nprobe=self.nprobe)
        candidates = self._apply_row_filters(ids, filters)
        if filters.active and len(candidates) < min(filtered_count, self.candidate_count):
            # A larger k cannot reach rows outside the probed lists, and a
            # filter whose rows sit in other clusters (field=Medicine on a CS
            # query) leaves the page nearly empty. Probe more lists, once.
            if self.filter_nprobe > self.nprobe:
                _, ids = self.index.search(query_vector, requested, nprobe=self.filter_nprobe)
                candidates = self._apply_row_filters(ids, filters)
        return sorted(candidates[:keep]), filtered_count

    @staticmethod
    def _ranking_score(
        metadata: dict,
        semantic_score: float,
        title_match: bool,
        exact_title_match: bool = False,
        *,
        prefix_match: bool = False,
        scale: float = 1.0,
    ) -> float:
        # Log-scaled and saturating at 100k citations, so the weight is spent
        # across the range real papers occupy rather than reserved for counts
        # no paper has. At corpus scale thousands of works cluster around any
        # concept and a 0-citation preprint otherwise outranks a landmark on a
        # 0.03 similarity edge.
        citation_boost = min(math.log1p(metadata["cited_by_count"]) / CITATION_SATURATION, 1.0)
        year = metadata["publication_year"]
        recency_boost = max(0.0, min((year - 1950) / 100.0, 1.0)) * RECENCY_WEIGHT
        # FTS phrase matches are useful recall signals, but only normalized
        # equality deserves the full exact-title boost. Otherwise a longer
        # title containing every query term can outrank the paper requested
        # verbatim (for example TimeSformer ahead of Attention Is All You Need).
        # A leading-prefix match is weaker evidence again, and is a bounded
        # boost rather than a pin: a semantically far better paper still wins.
        if exact_title_match:
            title_boost = 1.0
        elif prefix_match:
            title_boost = PREFIX_TITLE_BOOST * scale
        elif title_match:
            title_boost = 0.25
        else:
            title_boost = 0.0
        return (
            semantic_score
            + (citation_boost * CITATION_WEIGHT + recency_boost) * scale
            + title_boost
        )

    def _rank_features(
        self,
        ids: list[int],
        metadata: dict[int, dict] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._supplement_rows and any(self._is_supplement(row_id) for row_id in ids):
            stored = [row_id for row_id in ids if not self._is_supplement(row_id)]
            stored_years, stored_citations = (
                self._rank_features(stored, metadata)
                if stored
                else (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
            )
            years = np.zeros(len(ids), dtype=np.int64)
            citations = np.zeros(len(ids), dtype=np.int64)
            position = 0
            for index, row_id in enumerate(ids):
                if self._is_supplement(row_id):
                    record = self._supplement_rows[row_id]
                    years[index] = int(record["publication_year"])
                    citations[index] = effective_citations(record)
                else:
                    years[index] = stored_years[position]
                    citations[index] = stored_citations[position]
                    position += 1
            return years, citations
        if self.publication_years is not None and self.citation_counts is not None:
            selected = np.asarray(ids, dtype=np.int64)
            years = np.asarray(self.publication_years[selected], dtype=np.int64)
            citations = np.asarray(self.citation_counts[selected], dtype=np.int64)
        else:
            years = np.asarray(
                [metadata[row_id]["publication_year"] for row_id in ids], dtype=np.int64
            )
            citations = np.asarray(
                [metadata[row_id]["cited_by_count"] for row_id in ids], dtype=np.int64
            )
        if self._patched_rows:
            # Rank and sort on the corrected values the card will show.
            for index, row_id in enumerate(ids):
                record = self._patched_rows.get(row_id)
                if record is not None:
                    years[index] = int(record["publication_year"])
                    citations[index] = effective_citations(record)
        return years, citations

    def _citations_of(self, ids: list[int]) -> np.ndarray:
        if self.publication_years is not None and self.citation_counts is not None:
            _, citations = self._rank_features(ids)
            return citations
        return self._rank_features(ids, self._fetch(ids))[1]

    @staticmethod
    def _ranking_score_values(
        publication_year: int,
        cited_by_count: int,
        semantic_score: float,
        title_match: bool,
        exact_title_match: bool,
        *,
        prefix_match: bool = False,
        scale: float = 1.0,
    ) -> float:
        return SearchEngine._ranking_score(
            {
                "publication_year": publication_year,
                "cited_by_count": cited_by_count,
            },
            semantic_score,
            title_match,
            exact_title_match,
            prefix_match=prefix_match,
            scale=scale,
        )

    @staticmethod
    def _metadata_matches(item: dict, filters: Filters) -> bool:
        return not (
            # Year 0 means "unknown"; a year filter must not admit it.
            (
                (filters.year_min is not None or filters.year_max is not None)
                and item["publication_year"] <= 0
            )
            or (filters.year_min is not None and item["publication_year"] < filters.year_min)
            or (filters.year_max is not None and item["publication_year"] > filters.year_max)
            or (
                filters.min_citations is not None
                and effective_citations(item) < filters.min_citations
            )
            or (filters.open_access_only and not item["is_oa"])
            # Truthiness, not "is not None": the stores ignore an empty
            # topic or field, so checking identity here rejected every row for
            # a caller that sent topic="".
            or (bool(filters.topic) and item["topic"] != filters.topic)
            or (bool(filters.field) and item["field"] != filters.field)
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
        sort: str = "relevance",
        filters: Filters = Filters(),
    ) -> SearchResponse:
        started = time.perf_counter()
        cleaned_query = " ".join(query.split())
        if not cleaned_query:
            raise ValueError("query must not be empty")
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")
        if not 0 <= offset <= 10_000:
            raise ValueError("offset must be between 0 and 10000")
        if sort not in SORT_ORDERS:
            raise ValueError(f"sort must be one of: {', '.join(SORT_ORDERS)}")
        if (
            filters.year_min is not None
            and filters.year_max is not None
            and filters.year_min > filters.year_max
        ):
            raise ValueError("year_min must not be greater than year_max")

        # The stores ignore an empty topic or field, but Filters.active does
        # not, so topic="" used to send a plain search down the filtered path
        # and a full sidecar scan.
        filters = replace(filters, topic=filters.topic or None, field=filters.field or None)

        query_vector = self._embed_query(cleaned_query)
        embedded_at = time.perf_counter()
        semantic_ids, filtered_count = self._semantic_candidates(query_vector, filters)
        candidates_at = time.perf_counter()
        semantic_scores = (
            int8_scores(query_vector, self.codes[semantic_ids], self.int8_scales)
            if semantic_ids
            else np.empty(0, dtype=np.float32)
        )
        score_by_id = {
            row_id: float(semantic_scores[index])
            for index, row_id in enumerate(semantic_ids)
        }
        refined_at = time.perf_counter()

        title_ids, prefix_ids, title_keys = self._title_candidates(cleaned_query, filters)
        supplement_exact, supplement_semantic, supplement_scores = self._supplement_candidates(
            query_vector,
            title_keys,
            filters,
            min(semantic_scores.tolist()) if len(semantic_scores) else None,
        )
        score_by_id.update(supplement_scores)
        title_ids = [*title_ids, *supplement_exact]
        all_ids = list(
            dict.fromkeys([*title_ids, *prefix_ids, *semantic_ids, *supplement_semantic])
        )
        title_at = time.perf_counter()
        metadata = None
        if self.publication_years is None or self.citation_counts is None:
            metadata = self._fetch(all_ids)
        years, citations = self._rank_features(all_ids, metadata)
        # Any row the candidate index did not return needs its score computed
        # here, whether it arrived from exact-title lookup or the prefix index.
        missing_score_ids = [row_id for row_id in all_ids if row_id not in score_by_id]
        if missing_score_ids:
            missing_scores = int8_scores(
                query_vector,
                self.codes[missing_score_ids],
                self.int8_scales,
            )
            score_by_id.update(
                (row_id, float(missing_scores[index]))
                for index, row_id in enumerate(missing_score_ids)
            )
        # Normalise the pull of citations, recency and a title prefix to how
        # widely this query's candidates spread in similarity.
        pool_scores = [score_by_id[row_id] for row_id in (semantic_ids or all_ids)]
        best_score = max(pool_scores) if pool_scores else 0.0
        spread = best_score - min(pool_scores) if len(pool_scores) > 1 else 0.0
        scale = (
            min(2.0, max(0.25, spread / REFERENCE_SPREAD)) if spread > 0 else 1.0
        )
        relevance_floor = best_score - SORT_RELEVANCE_FLOOR * spread

        ranked: list[_RankedCandidate] = []
        title_id_set = set(title_ids) | set(prefix_ids)
        exact_title_ids = set(title_ids)
        prefix_id_set = set(prefix_ids) - exact_title_ids
        for index, row_id in enumerate(all_ids):
            title_match = row_id in title_id_set
            exact_title_match = (
                row_id in exact_title_ids
                if self.metadata_backend == "compact"
                or metadata is None
                or self._is_supplement(row_id)
                else metadata[row_id]["normalized_title"] in title_keys
            )
            prefix_match = row_id in prefix_id_set and not exact_title_match
            semantic_score = score_by_id[row_id]
            publication_year = int(years[index])
            cited_by_count = int(citations[index])
            ranking_score = self._ranking_score_values(
                publication_year,
                cited_by_count,
                semantic_score,
                title_match,
                exact_title_match,
                prefix_match=prefix_match,
                scale=scale,
            )
            ranked.append(
                _RankedCandidate(
                    row_id=row_id,
                    semantic_score=semantic_score,
                    ranking_score=ranking_score,
                    cited_by_count=cited_by_count,
                    publication_year=publication_year,
                    title_match=title_match,
                    exact_title_match=exact_title_match,
                    prefix_match=prefix_match,
                )
            )
        if sort != "relevance":
            ranked = [
                candidate
                for candidate in ranked
                if candidate.exact_title_match or candidate.semantic_score >= relevance_floor
            ]
        ranked = self._order(ranked, sort)

        records: dict[int, dict] = {}

        def _rich(batch: list[_RankedCandidate]) -> list[tuple[_RankedCandidate, dict]]:
            wanted_rows = [c.row_id for c in batch if c.row_id not in records]
            records.update(
                (row, self.overrides.apply(item))
                for row, item in self._fetch(wanted_rows).items()
            )
            rows: list[tuple[_RankedCandidate, dict]] = []
            for candidate in batch:
                item = records.get(candidate.row_id)
                if item is None:
                    raise RuntimeError(f"Metadata row {candidate.row_id} is missing")
                rows.append((candidate, item))
            return rows

        ranked = self._rerank_window(
            ranked,
            _rich,
            sort=sort,
            multi_word=len(cleaned_query.split()) > 1,
            scale=scale,
        )
        ranked_at = time.perf_counter()

        # Every candidate source above is already filtered on the sidecars
        # (and corrected rows on their correction), so ranked is the filtered
        # pool and its length the total. Compact topic/field filters compare
        # 64-bit hashes, so recheck rich metadata before returning a row, which
        # keeps even a hash collision from leaking a false result. A rejected
        # row is replaced from further down, never left as a hole.
        start = min(offset, len(ranked))
        cursor = start
        rejected = 0
        page_rows: list[tuple[_RankedCandidate, dict]] = []
        while cursor < len(ranked) and len(page_rows) < limit:
            batch = ranked[cursor : cursor + (limit - len(page_rows))]
            for row in _rich(batch):
                if filters.active and not self._metadata_matches(row[1], filters):
                    rejected += 1
                    continue
                page_rows.append(row)
            cursor += len(batch)
        selected_total = len(ranked) - rejected
        page = [
            self._result(
                candidate.row_id,
                item,
                semantic_score=candidate.semantic_score,
                ranking_score=candidate.ranking_score,
                title_match=candidate.title_match,
            )
            for candidate, item in page_rows
        ]
        metadata_at = time.perf_counter()
        return SearchResponse(
            query=cleaned_query,
            corpus_records=self.record_count,
            candidates=len(semantic_ids),
            filtered_records=filtered_count,
            elapsed_ms=(time.perf_counter() - started) * 1_000,
            results=page,
            # Clamped to the ranked pool: an offset past its end is answered
            # with an empty last page at the pool's end, not an echo of a
            # position that does not exist.
            offset=start,
            page_size=limit,
            total_matches=selected_total,
            # Judged against rows actually left, and only with a non-empty
            # page, so a client can never be sent back to the same offset.
            has_more=bool(page) and cursor < len(ranked),
            sort=sort,
            generation_id=self.generation_id,
            snapshot_at=self.snapshot_at,
            # True whenever more records match than were ranked: the pool is
            # bounded, so total_matches is then a floor, not a count.
            total_matches_capped=filtered_count > selected_total,
            cursor=cursor,
            timings_ms={
                "query_embedding": (embedded_at - started) * 1_000,
                "candidate_search": (candidates_at - embedded_at) * 1_000,
                "int8_refinement": (refined_at - candidates_at) * 1_000,
                "title_lookup": (title_at - refined_at) * 1_000,
                "ranking": (ranked_at - title_at) * 1_000,
                "metadata_fetch": (metadata_at - ranked_at) * 1_000,
            },
        )

    def _title_candidates(
        self, query: str, filters: Filters
    ) -> tuple[list[int], list[int], set[str]]:
        """Exact-title and prefix rows for ``query`` and its spelling variants.

        Returns (exact rows, prefix rows, the keys that were looked up). The
        tables are the unchanged build-time ones; only the keys are varied.
        """
        keys = title_key_variants(query)
        # Fixed, and deliberately independent of both offset and limit.
        # Retrieving more title rows on a later page inserts them into the
        # pinned tier and reshuffles everything above, so an offset-scaled
        # limit makes page 2 overlap page 1. A constant keeps the candidate set
        # identical for every page of a query.
        title_ids = self._exact_title_rows(keys, filters)
        prefix_keys = list(keys)
        if not title_ids and keys:
            probe = (
                (lambda text: self.title_prefixes.contains(text, stable_text_hash))
                if self.title_prefixes is not None
                else None
            )
            # Bounded: at most HYPHENATION_PROBES binary searches of the
            # prefix table, and only for title-length queries.
            decoded = hyphenation_candidates(
                keys[0],
                prefix_exists=probe,
                max_probes=HYPHENATION_PROBES,
                max_words=16,
            )[:HYPHENATION_CANDIDATES]
            exact_keys = [key for key in decoded if self.store.title_key_exists(key)]
            if exact_keys:
                title_ids = self._exact_title_rows(exact_keys, filters)
                keys.extend(exact_keys)
            if probe is not None:
                prefix_keys.extend(key for key in decoded if probe(key))
        prefix_ids: list[int] = []
        if not title_ids and self.title_prefixes is not None:
            # A reader who types part of a title gets nothing from the exact
            # index, and a short query scores poorly against a title-plus-
            # abstract embedding, so semantic retrieval does not rescue it
            # either. Treat a leading-prefix match as a weaker title signal.
            prefix_ids = self._prefix_rows(prefix_keys, filters)
        return title_ids, prefix_ids, set(keys)

    def _exact_title_rows(self, keys: list[str], filters: Filters) -> list[int]:
        rows: list[int] = []
        for key in keys:
            found = self.store.title_search(key, filters, limit=TITLE_CANDIDATE_LIMIT)
            if filters.active and self._patched_rows:
                found = [
                    row_id
                    for row_id in found
                    if row_id not in self._patched_rows
                    or self._metadata_matches(self._patched_rows[row_id], filters)
                ]
                found.extend(
                    row_id
                    for row_id, title in self._override_titles.items()
                    if title == key
                    and row_id not in found
                    and self._metadata_matches(self._patched_rows[row_id], filters)
                )
            rows.extend(found)
        rows = list(dict.fromkeys(rows))
        if len(rows) > TITLE_CANDIDATE_LIMIT:
            citations = self._citations_of(rows)
            order = np.lexsort((np.asarray(rows), -citations))[:TITLE_CANDIDATE_LIMIT]
            rows = [rows[index] for index in order]
        return rows

    def _prefix_rows(self, keys: list[str], filters: Filters) -> list[int]:
        """Rows whose stored title begins with one of ``keys``, most cited first.

        Each hit is confirmed against the stored normalized title before use,
        so a hash collision or a stale index cannot inject an unrelated row.
        """
        usable = [key for key in keys if MIN_PREFIX_WORDS <= len(key.split()) <= MAX_PREFIX_WORDS]
        rows: list[int] = []
        for key in usable:
            rows.extend(
                self.title_prefixes.lookup(key, stable_text_hash, max_rows=PREFIX_SCAN_ROWS)
            )
        rows = self._apply_row_filters(list(dict.fromkeys(rows)), filters)
        if not rows:
            return []
        citations = self._citations_of(rows)
        ordered = np.asarray(rows, dtype=np.int64)[np.lexsort((np.asarray(rows), -citations))]
        starts = tuple(f"{key} " for key in usable)
        kept: list[int] = []
        for begin in range(0, min(len(ordered), 4 * PREFIX_CANDIDATE_LIMIT), PREFIX_CANDIDATE_LIMIT):
            chunk = ordered[begin : begin + PREFIX_CANDIDATE_LIMIT].tolist()
            stored = self.store.fetch(chunk)
            kept.extend(
                row_id
                for row_id in chunk
                if row_id in stored and stored[row_id]["normalized_title"].startswith(starts)
            )
            if len(kept) >= PREFIX_CANDIDATE_LIMIT:
                break
        return kept[:PREFIX_CANDIDATE_LIMIT]

    @staticmethod
    def _order(rows: list[_RankedCandidate], sort: str) -> list[_RankedCandidate]:
        if sort == "relevance":
            # An exact title is an explicit lookup signal, so under relevance
            # those rows lead, and same-title works resolve by citations.
            # Only under relevance: a reader who asks for newest or most cited
            # has said what order they want.
            exact = [row for row in rows if row.exact_title_match]
            exact.sort(key=lambda r: (-r.cited_by_count, -r.ranking_score, r.row_id))
            rest = [row for row in rows if not row.exact_title_match]
            rest.sort(key=lambda r: (-r.ranking_score, -r.cited_by_count, r.row_id))
            return [*exact, *rest]
        # The whole merged pool, title rows included, by the requested key.
        if sort == "most_cited":
            key = lambda r: (-r.cited_by_count, -r.ranking_score, r.row_id)
        elif sort == "newest":
            key = lambda r: (r.publication_year <= 0, -r.publication_year, -r.ranking_score, r.row_id)
        else:
            key = lambda r: (r.publication_year <= 0, r.publication_year, -r.ranking_score, r.row_id)
        return sorted(rows, key=key)

    def _prefer_duplicate(self, candidate: _RankedCandidate, kept: _RankedCandidate) -> bool:
        """Whether ``candidate`` should replace ``kept``, a record of the same work.

        A supplement record is hand-verified, so it replaces a generation copy
        of the same work and is never replaced by one.
        """
        candidate_verified = self._is_supplement(candidate.row_id)
        kept_verified = self._is_supplement(kept.row_id)
        if candidate_verified != kept_verified:
            return candidate_verified
        return candidate.cited_by_count > kept.cited_by_count

    def _rerank_window(
        self,
        ranked: list[_RankedCandidate],
        rich,
        *,
        sort: str,
        multi_word: bool,
        scale: float,
    ) -> list[_RankedCandidate]:
        """Collapse duplicate works and demote title-only records near the top.

        Both need card metadata, which costs a block read per row, so they
        apply to a fixed leading window rather than the whole pool. The window
        does not depend on offset or limit, so every page of a query sees the
        same order.
        """
        window = ranked[: self.rerank_window]
        if len(window) < 2:
            return ranked
        items = {candidate.row_id: item for candidate, item in rich(window)}
        if sort == "relevance" and multi_word:
            window = [
                replace(
                    candidate,
                    ranking_score=candidate.ranking_score - TITLE_ONLY_PENALTY * scale,
                )
                if not str(items[candidate.row_id].get("snippet") or "").strip()
                else candidate
                for candidate in window
            ]
        kept: list[_RankedCandidate] = []
        by_title: dict[str, list[int]] = {}
        for candidate in window:
            item = items[candidate.row_id]
            title = normalize_title(str(item.get("title") or ""))
            duplicate_of = next(
                (
                    position
                    for position in by_title.get(title, ())
                    if _same_work(items[kept[position].row_id], item)
                ),
                None,
            ) if title else None
            if duplicate_of is None:
                by_title.setdefault(title, []).append(len(kept))
                kept.append(candidate)
            elif self._prefer_duplicate(candidate, kept[duplicate_of]):
                # Keep the verified or most-cited version, in the better-ranked slot.
                kept[duplicate_of] = candidate
        return [*self._order(kept, sort), *ranked[len(window) :]]


def _normalized_doi(value: str) -> str:
    text = (value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _first_author_surname(item: dict) -> str:
    authors = item.get("authors") or ()
    if not authors:
        return ""
    words = normalize_title(_fold_accents(str(authors[0]))).split()
    return words[-1] if words else ""


def _same_work(first: dict, second: dict) -> bool:
    """Two records with the same normalized title that are one work.

    OpenAlex keeps preprint, proceedings and reprint versions under separate
    ids. They are the same work when they share a DOI, or a first author; with
    no author to compare, publication within a year of each other. Different
    first authors always stay separate: a shared generic title is common.
    """
    first_doi = _normalized_doi(str(first.get("doi") or ""))
    if first_doi and first_doi == _normalized_doi(str(second.get("doi") or "")):
        return True
    first_author = _first_author_surname(first)
    second_author = _first_author_surname(second)
    if first_author and second_author:
        return first_author == second_author
    first_year = int(first.get("publication_year") or 0)
    second_year = int(second.get("publication_year") or 0)
    return first_year > 0 and second_year > 0 and abs(first_year - second_year) <= 1
