from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, Security
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .compact_store import FilterScanBudgetExceeded
from .config import SearchTuning
from .embeddings import SentenceTransformerEmbedder
from .engine import SearchEngine
from .overrides import MetadataOverrides
from .progress import read_index_progress
from .supplement import SupplementRecords
from .title_prefix import TitlePrefixIndex
from .work_index import WorkIdIndex
from .store import Filters

# Client UIs typically offer these minimum-citation steps. Rounding any other
# value down to one keeps the filter cache keyed on a handful of values, so
# arbitrary numbers cannot each force a fresh full scan.
MIN_CITATION_BUCKETS = (0, 10, 50, 100, 500, 1000, 5000, 10000)
MAX_INT32 = 2_147_483_647


def bucket_min_citations(value: int | None) -> int | None:
    if value is None:
        return None
    return max(bucket for bucket in MIN_CITATION_BUCKETS if bucket <= value)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class SearchRequest(BaseModel):
    # A lookup by identifier rides the search route rather than a new path,
    # because only POST /search is publicly proxied. query stays required so
    # the contract is unchanged for every existing caller.
    query: str = Field(min_length=1, max_length=2_000)
    work_id: str | None = Field(default=None, max_length=256)
    limit: int = Field(default=10, ge=1, le=50)
    offset: int = Field(default=0, ge=0, le=10_000)
    sort: str = Field(default="relevance", pattern="^(relevance|most_cited|newest|oldest)$")
    year_min: int | None = Field(default=None, ge=0, le=2100)
    year_max: int | None = Field(default=None, ge=0, le=2100)
    min_citations: int | None = Field(default=None, ge=0, le=MAX_INT32)
    open_access_only: bool = False
    topic: str | None = Field(default=None, max_length=200)
    field: str | None = Field(default=None, max_length=200)


def create_app(
    *,
    generation: Path | None = None,
    api_keys: frozenset[str] | None = None,
    engine: SearchEngine | None = None,
    progress_artifacts: Path | None = None,
    overrides: Path | None = None,
    title_prefixes: Path | None = None,
    work_ids: Path | None = None,
    supplement: Path | None = None,
):
    owned_engine = engine is None
    resolved_generation = generation
    if owned_engine and resolved_generation is None:
        configured_index = os.environ.get("OPENALEX_SEARCH_INDEX")
        if not configured_index:
            raise RuntimeError("OPENALEX_SEARCH_INDEX must name a built generation")
        resolved_generation = Path(configured_index)
    resolved_keys = api_keys or frozenset(
        key.strip()
        for key in os.environ.get("OPENALEX_SEARCH_API_KEYS", "").split(",")
        if key.strip()
    )
    if not resolved_keys:
        raise RuntimeError("OPENALEX_SEARCH_API_KEYS must contain at least one application key")
    resolved_overrides = overrides
    if resolved_overrides is None:
        configured_overrides = os.environ.get("OPENALEX_SEARCH_OVERRIDES")
        if configured_overrides:
            resolved_overrides = Path(configured_overrides)
    # Fail fast at import time rather than serving a generation whose
    # corrections silently did not load.
    loaded_overrides = MetadataOverrides.load(resolved_overrides)

    resolved_title_prefixes = title_prefixes
    if resolved_title_prefixes is None:
        configured_prefixes = os.environ.get("OPENALEX_SEARCH_TITLE_PREFIXES")
        if configured_prefixes:
            resolved_title_prefixes = Path(configured_prefixes)
    loaded_title_prefixes = TitlePrefixIndex.load(resolved_title_prefixes)

    resolved_work_ids = work_ids
    if resolved_work_ids is None:
        configured_work_ids = os.environ.get("OPENALEX_SEARCH_WORK_IDS")
        if configured_work_ids:
            resolved_work_ids = Path(configured_work_ids)
    loaded_work_ids = WorkIdIndex.load(resolved_work_ids)

    resolved_supplement = supplement
    if resolved_supplement is None:
        configured_supplement = os.environ.get("OPENALEX_SEARCH_SUPPLEMENT")
        if configured_supplement:
            resolved_supplement = Path(configured_supplement)
    # Unlike corrections, a bad supplement is logged and skipped: it only adds
    # results, so serving without it is the behaviour before it existed.
    loaded_supplement = SupplementRecords.load_or_empty(resolved_supplement)
    # Read at startup so a malformed value fails the deploy, not a request.
    tuning = SearchTuning.from_environment()

    resolved_progress_artifacts = progress_artifacts
    if resolved_progress_artifacts is None:
        configured_progress = os.environ.get("OPENALEX_SEARCH_BACKFILL_ARTIFACTS")
        if configured_progress:
            resolved_progress_artifacts = Path(configured_progress)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal engine
        if engine is None:
            embedder = SentenceTransformerEmbedder(
                os.getenv("OPENALEX_SEARCH_MODEL", "BAAI/bge-small-en-v1.5"),
                device=os.getenv("OPENALEX_SEARCH_DEVICE", "cpu"),
            )
            if resolved_generation is None:  # Narrowing guard for type checkers.
                raise RuntimeError("No generation configured")
            engine = SearchEngine(
                resolved_generation,
                embedder,
                overrides=loaded_overrides,
                title_prefixes=loaded_title_prefixes,
                work_ids=loaded_work_ids,
                tuning=tuning,
                supplement=loaded_supplement,
            )
        app.state.engine = engine
        yield
        if owned_engine and engine is not None:
            engine.close()

    app = FastAPI(
        title="OpenAlex Search Lab",
        docs_url=None,
        redoc_url=None,
        # The schema maps every route and bound; publish it only on request.
        openapi_url="/openapi.json" if _env_flag("OPENALEX_SEARCH_EXPOSE_OPENAPI") else None,
        lifespan=lifespan,
    )

    @app.exception_handler(FilterScanBudgetExceeded)
    async def scan_budget_exceeded(_request, error: FilterScanBudgetExceeded):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many new filter combinations; retry shortly"},
            headers={"Retry-After": str(error.retry_after), "Cache-Control": "no-store"},
        )
    key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_key(provided: str | None = Security(key_header)) -> None:
        if provided and any(hmac.compare_digest(provided, valid) for valid in resolved_keys):
            return
        raise HTTPException(status_code=401, detail="Invalid or missing application key")

    @app.get("/healthz")
    def healthz() -> dict:
        # Liveness only: counts, generation and tuning are behind the key.
        app.state.engine  # Raises (500) until the engine is loaded.
        return {"ok": True}

    @app.get("/healthz/details")
    def healthz_details(_: None = Security(require_key)) -> dict:
        current = app.state.engine
        return {
            "status": "ok",
            "stage": current.manifest["stage"],
            "records": current.record_count,
            "backend": current.manifest["backend"],
            "generation_id": current.generation_id,
            "snapshot_at": current.snapshot_at,
            "overrides": len(current.overrides),
            "title_prefixes": (
                len(current.title_prefixes) if current.title_prefixes else 0
            ),
            "work_ids": len(current.work_ids) if current.work_ids else 0,
            "supplement_records": getattr(current, "supplement_count", 0),
            "embedder_matches_generation": getattr(
                current, "embedder_matches_generation", None
            ),
            # The effective read-time settings, so a deploy can be checked.
            "search_tuning": {
                name: getattr(current, name)
                for name in (
                    "nprobe",
                    "candidate_count",
                    "filter_nprobe",
                    "filter_candidate_count",
                    "selective_filter_threshold",
                    "rerank_window",
                )
                if hasattr(current, name)
            },
        }

    @app.get("/index-progress")
    def index_progress(response: Response, _: None = Security(require_key)) -> dict:
        response.headers["Cache-Control"] = "no-store"
        return read_index_progress(
            resolved_progress_artifacts,
            indexed_records=app.state.engine.record_count,
        )

    @app.post("/search")
    def search(request: SearchRequest, _: None = Security(require_key)) -> dict:
        filters = Filters(
            year_min=request.year_min,
            year_max=request.year_max,
            min_citations=bucket_min_citations(request.min_citations),
            open_access_only=request.open_access_only,
            topic=request.topic,
            field=request.field,
        )
        if request.work_id:
            engine = app.state.engine
            try:
                work = engine.fetch_work(request.work_id)
            except RuntimeError as error:
                # No work id index, and not a supplement record.
                raise HTTPException(
                    status_code=501, detail="This generation has no work id index"
                ) from error
            if work is None:
                raise HTTPException(status_code=404, detail="Unknown work id")
            return {
                "query": request.query,
                "corpus_records": engine.record_count,
                "generation_id": engine.generation_id,
                "snapshot_at": engine.snapshot_at,
                "offset": 0,
                "page_size": 1,
                "total_matches": 1,
                "has_more": False,
                "next_offset": None,
                "results": [work],
            }
        try:
            return app.state.engine.search(
                request.query,
                limit=request.limit,
                offset=request.offset,
                sort=request.sort,
                filters=filters,
            ).as_dict()
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    if not _env_flag("OPENALEX_SEARCH_UI"):
        return app

    static = Path(__file__).with_name("static")
    # Revalidate on every load so a redeploy is never hidden by browser cache.
    no_cache = {"Cache-Control": "no-cache"}

    @app.get("/")
    def index():
        return FileResponse(static / "index.html", headers=no_cache)

    @app.get("/app.css")
    def css():
        return FileResponse(static / "app.css", media_type="text/css", headers=no_cache)

    @app.get("/app.js")
    def javascript():
        return FileResponse(static / "app.js", media_type="text/javascript", headers=no_cache)

    return app


def app_from_environment():
    return create_app()
