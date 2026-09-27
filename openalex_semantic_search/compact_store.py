from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import gzip
from hashlib import blake2b
import json
import logging
import mmap
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Callable, Sequence
import zlib

import numpy as np

from .store import Filters, normalize_title


logger = logging.getLogger(__name__)

COMPACT_FORMAT_VERSION = 1
DEFAULT_BLOCK_ROWS = 256
# Rows a filter may materialize for an exact scan. The scan itself is cheap;
# what follows it is not, because the engine then gathers one 384-byte vector
# per row scattered across a 121GB file. Measured on this hardware that costs
# a flat 100us per row -- 1s at 10,000 rows, 25s at 250,000 -- so the exact
# path only pays for itself on genuinely narrow filters. Anything wider goes
# to the candidate index, which is nearly flat in k.
SELECTIVE_FILTER_MAX_ROWS = 10_000
# Filter results (a count, plus the row ids when there are at most
# SELECTIVE_FILTER_MAX_ROWS of them) are cached per distinct Filters value.
# Each entry holds at most 80KB of ids; the byte bound keeps the whole cache
# small even if every entry is a full id list.
FILTER_CACHE_MAX_ENTRIES = 256
FILTER_CACHE_MAX_ID_BYTES = 32 * 1024 * 1024
# A full filter scan walks every row's sidecars (~300M rows). Uncached scans
# are rationed process-wide so varied filter values cannot queue scan after
# scan; over the budget a request fails fast and the client retries.
DEFAULT_FILTER_SCANS = 2
DEFAULT_FILTER_SCAN_WINDOW_SECONDS = 10.0
_HASH_PERSON = b"openalex-v1"


class FilterScanBudgetExceeded(Exception):
    """An uncached full filter scan was refused; retry after ``retry_after`` s.

    Deliberately not a RuntimeError: the engine treats RuntimeError from the
    store as "too many rows to materialize" and falls back, which would hide
    this refusal instead of surfacing it as HTTP 429.
    """

    def __init__(self, retry_after: float):
        super().__init__("Filter scan budget exhausted; retry shortly")
        self.retry_after = max(1, int(retry_after + 0.999))


class FilterScanBudget:
    """At most ``scans`` uncached full scans per ``window_seconds`` (sliding)."""

    def __init__(
        self,
        scans: int = DEFAULT_FILTER_SCANS,
        window_seconds: float = DEFAULT_FILTER_SCAN_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        if scans < 1 or window_seconds <= 0:
            raise ValueError("A filter scan budget needs at least one scan per positive window")
        self.scans = scans
        self.window_seconds = window_seconds
        self._clock = clock
        self._starts: deque[float] = deque()
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls, environ=os.environ) -> "FilterScanBudget":
        scans = int(environ.get("OPENALEX_SEARCH_FILTER_SCANS", DEFAULT_FILTER_SCANS))
        window = float(
            environ.get(
                "OPENALEX_SEARCH_FILTER_SCAN_WINDOW_SECONDS",
                DEFAULT_FILTER_SCAN_WINDOW_SECONDS,
            )
        )
        return cls(scans, window)

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            while self._starts and now - self._starts[0] >= self.window_seconds:
                self._starts.popleft()
            if len(self._starts) >= self.scans:
                raise FilterScanBudgetExceeded(
                    self.window_seconds - (now - self._starts[0])
                )
            self._starts.append(now)


def _filter_components(filters: Filters) -> list[Filters]:
    """One single-dimension Filters per active dimension of ``filters``."""
    parts: list[Filters] = []
    if filters.year_min is not None or filters.year_max is not None:
        parts.append(Filters(year_min=filters.year_min, year_max=filters.year_max))
    if filters.min_citations is not None:
        parts.append(Filters(min_citations=filters.min_citations))
    if filters.open_access_only:
        parts.append(Filters(open_access_only=True))
    if filters.topic:
        parts.append(Filters(topic=filters.topic))
    if filters.field:
        parts.append(Filters(field=filters.field))
    return parts
_PART_MARKER = "part-complete.json"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_marker(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def stable_text_hash(value: str) -> int:
    if not value:
        return 0
    digest = blake2b(
        value.encode("utf-8"), digest_size=8, person=_HASH_PERSON
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


@dataclass(frozen=True, slots=True)
class _ShardResult:
    shard_index: int
    start_row: int
    records: int
    part_directory: str
    year_counts: dict[int, int]
    open_access_count: int


def _open_sidecar(path: Path, dtype: np.dtype, count: int) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(count,))


def _compact_shard(
    shard_index: int,
    start_row: int,
    expected_records: int,
    metadata_path: str,
    parts_root: str,
    block_rows: int,
) -> _ShardResult:
    """Convert one gzip metadata shard into independent compact parts."""
    output = Path(parts_root) / f"shard-{shard_index:05d}"
    output.mkdir(parents=True, exist_ok=False)
    years = _open_sidecar(output / "years.npy", np.dtype(np.int16), expected_records)
    citations = _open_sidecar(output / "citations.npy", np.dtype(np.uint32), expected_records)
    open_access = _open_sidecar(output / "open-access.npy", np.dtype(np.uint8), expected_records)
    topic_hashes = _open_sidecar(
        output / "topic-hashes.npy", np.dtype(np.uint64), expected_records
    )
    field_hashes = _open_sidecar(
        output / "field-hashes.npy", np.dtype(np.uint64), expected_records
    )
    title_hashes = _open_sidecar(
        output / "title-hashes.npy", np.dtype(np.uint64), expected_records
    )
    offsets = [0]
    row_starts: list[int] = []
    block: list[bytes] = []
    row = 0
    year_counts: dict[int, int] = {}
    open_access_count = 0
    with open(output / "metadata.blocks", "wb") as blocks:
        with gzip.open(metadata_path, "rb") as lines:
            for line in lines:
                if row >= expected_records:
                    raise ValueError(
                        f"{metadata_path} contains more than {expected_records:,} records"
                    )
                record = json.loads(line)
                year = max(0, min(32767, int(record.get("publication_year") or 0)))
                years[row] = year
                citation_count = max(0, int(record.get("cited_by_count") or 0))
                citations[row] = min(citation_count, np.iinfo(np.uint32).max)
                is_open_access = bool(record.get("is_oa"))
                open_access[row] = is_open_access
                open_access_count += int(is_open_access)
                topic_hashes[row] = stable_text_hash(str(record.get("topic") or ""))
                field_hashes[row] = stable_text_hash(str(record.get("field") or ""))
                title_hashes[row] = stable_text_hash(
                    normalize_title(str(record.get("title") or ""))
                )
                year_counts[year] = year_counts.get(year, 0) + 1
                if not line.endswith(b"\n"):
                    line += b"\n"
                block.append(line)
                row += 1
                if len(block) == block_rows:
                    row_starts.append(start_row + row - len(block))
                    blocks.write(zlib.compress(b"".join(block), level=1))
                    offsets.append(blocks.tell())
                    block = []
            if block:
                row_starts.append(start_row + row - len(block))
                blocks.write(zlib.compress(b"".join(block), level=1))
                offsets.append(blocks.tell())
    if row != expected_records:
        raise ValueError(
            f"{metadata_path} contains {row:,} records; expected {expected_records:,}"
        )
    for values in (years, citations, open_access, topic_hashes, field_hashes, title_hashes):
        values.flush()
    np.save(output / "block-offsets.npy", np.asarray(offsets, dtype=np.uint64))
    np.save(output / "block-row-starts.npy", np.asarray(row_starts, dtype=np.uint64))
    result = _ShardResult(
        shard_index=shard_index,
        start_row=start_row,
        records=row,
        part_directory=str(output),
        year_counts=year_counts,
        open_access_count=open_access_count,
    )
    component_names = (
        "years.npy",
        "citations.npy",
        "open-access.npy",
        "topic-hashes.npy",
        "field-hashes.npy",
        "title-hashes.npy",
        "block-offsets.npy",
        "block-row-starts.npy",
        "metadata.blocks",
    )
    _write_marker(
        output / _PART_MARKER,
        {
            "format_version": COMPACT_FORMAT_VERSION,
            "shard_index": result.shard_index,
            "start_row": result.start_row,
            "records": result.records,
            "year_counts": result.year_counts,
            "open_access_count": result.open_access_count,
            "files": {
                name: {
                    "bytes": (output / name).stat().st_size,
                    "sha256": _sha256(output / name),
                }
                for name in component_names
            },
        },
    )
    return result


def _load_completed_part(
    output: Path,
    *,
    shard_index: int,
    start_row: int,
    expected_records: int,
) -> _ShardResult | None:
    marker = output / _PART_MARKER
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        int(data.get("format_version", -1)) != COMPACT_FORMAT_VERSION
        or
        int(data.get("shard_index", -1)) != shard_index
        or int(data.get("start_row", -1)) != start_row
        or int(data.get("records", -1)) != expected_records
    ):
        return None
    required = (
        "years.npy",
        "citations.npy",
        "open-access.npy",
        "topic-hashes.npy",
        "field-hashes.npy",
        "title-hashes.npy",
        "block-offsets.npy",
        "block-row-starts.npy",
        "metadata.blocks",
    )
    if not all((output / name).is_file() for name in required):
        return None
    try:
        file_markers = data["files"]
        if set(file_markers) != set(required):
            return None
        for name in required:
            path = output / name
            marker = file_markers[name]
            if int(marker["bytes"]) != path.stat().st_size:
                return None
            if str(marker["sha256"]) != _sha256(path):
                return None
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return _ShardResult(
        shard_index=shard_index,
        start_row=start_row,
        records=expected_records,
        part_directory=str(output),
        year_counts={int(year): int(count) for year, count in data["year_counts"].items()},
        open_access_count=int(data["open_access_count"]),
    )


def _copy_sidecar_parts(
    results: list[_ShardResult],
    directory: Path,
    *,
    part_name: str,
    final_name: str,
    dtype: np.dtype,
    total: int,
) -> None:
    output = np.lib.format.open_memmap(
        directory / final_name, mode="w+", dtype=dtype, shape=(total,)
    )
    for result in results:
        part = np.load(Path(result.part_directory) / part_name, mmap_mode="r")
        output[result.start_row : result.start_row + result.records] = part
    output.flush()


def _build_selective_year_index(
    directory: Path,
    *,
    year_counts: dict[int, int],
    total: int,
) -> None:
    """Materialize row ids only for years small enough for exact scanning."""
    eligible_years = np.asarray(
        sorted(
            year
            for year, count in year_counts.items()
            if year > 0 and count <= SELECTIVE_FILTER_MAX_ROWS
        ),
        dtype=np.int16,
    )
    if len(eligible_years) == 0:
        np.save(directory / "selective-year-values.npy", eligible_years)
        np.save(directory / "selective-year-offsets.npy", np.zeros(1, dtype=np.uint64))
        np.save(directory / "selective-year-rows.npy", np.empty(0, dtype=np.uint32))
        return

    years = np.load(directory / "ranking-years.npy", mmap_mode="r")
    selected_year_parts: list[np.ndarray] = []
    selected_row_parts: list[np.ndarray] = []
    chunk_rows = 5_000_000
    for start in range(0, total, chunk_rows):
        stop = min(start + chunk_rows, total)
        values = np.asarray(years[start:stop])
        keep = np.isin(values, eligible_years, assume_unique=False)
        local_rows = np.flatnonzero(keep)
        if len(local_rows):
            selected_year_parts.append(values[local_rows])
            selected_row_parts.append((local_rows + start).astype(np.uint32))

    if selected_row_parts:
        selected_years = np.concatenate(selected_year_parts)
        selected_rows = np.concatenate(selected_row_parts)
        order = np.argsort(selected_years, kind="stable")
        selected_years = selected_years[order]
        selected_rows = selected_rows[order]
        counts = np.searchsorted(
            selected_years,
            eligible_years,
            side="right",
        ) - np.searchsorted(selected_years, eligible_years, side="left")
    else:
        selected_rows = np.empty(0, dtype=np.uint32)
        counts = np.zeros(len(eligible_years), dtype=np.int64)
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.uint64), np.cumsum(counts, dtype=np.uint64))
    )
    np.save(directory / "selective-year-values.npy", eligible_years)
    np.save(directory / "selective-year-offsets.npy", offsets)
    np.save(directory / "selective-year-rows.npy", selected_rows)


def build_compact_metadata(
    shards: Sequence[Path],
    directory: Path,
    *,
    shard_rows: Sequence[int],
    total: int,
    workers: int = 8,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    resume: bool = False,
) -> dict:
    """Build a compact, random-access metadata layer in parallel by shard."""
    if len(shards) != len(shard_rows):
        raise ValueError("shard row counts do not align with metadata shards")
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    parts = directory / ".compact-parts"
    parts.mkdir(exist_ok=resume)
    starts: list[int] = []
    row = 0
    for count in shard_rows:
        starts.append(row)
        row += int(count)
    if row != total:
        raise ValueError(f"shard rows total {row:,}; expected {total:,}")

    results: list[_ShardResult] = []
    pending: list[int] = []
    for index, expected_records in enumerate(shard_rows):
        output = parts / f"shard-{index:05d}"
        completed = _load_completed_part(
            output,
            shard_index=index,
            start_row=starts[index],
            expected_records=int(expected_records),
        )
        if completed is not None:
            results.append(completed)
            logger.info(
                "compact metadata: shard %d/%d reused from checkpoint",
                index + 1,
                len(shards),
            )
            continue
        if output.exists():
            shutil.rmtree(output)
        pending.append(index)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=context) as pool:
        futures = {
            pool.submit(
                _compact_shard,
                index,
                starts[index],
                int(shard_rows[index]),
                str(shard.with_name(shard.name.replace(".int8.npy", ".meta.jsonl.gz"))),
                str(parts),
                block_rows,
            ): index
            for index in pending
            for shard in (shards[index],)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            logger.info(
                "compact metadata: shard %d/%d converted (%s rows)",
                result.shard_index + 1,
                len(shards),
                f"{result.records:,}",
            )
    results.sort(key=lambda value: value.shard_index)

    global_offsets = [0]
    global_row_starts: list[int] = []
    with open(directory / "metadata.blocks", "wb") as target:
        for result in results:
            part_directory = Path(result.part_directory)
            relative_offsets = np.load(part_directory / "block-offsets.npy")
            row_starts = np.load(part_directory / "block-row-starts.npy")
            base = target.tell()
            with open(part_directory / "metadata.blocks", "rb") as source:
                shutil.copyfileobj(source, target, length=8 << 20)
            global_offsets.extend(int(base + value) for value in relative_offsets[1:])
            global_row_starts.extend(int(value) for value in row_starts)
    np.save(directory / "metadata-offsets.npy", np.asarray(global_offsets, dtype=np.uint64))
    np.save(
        directory / "metadata-row-starts.npy",
        np.asarray(global_row_starts, dtype=np.uint64),
    )

    sidecars = (
        ("years.npy", "ranking-years.npy", np.dtype(np.int16)),
        ("citations.npy", "ranking-citations.npy", np.dtype(np.uint32)),
        ("open-access.npy", "filter-open-access.npy", np.dtype(np.uint8)),
        ("topic-hashes.npy", "filter-topic-hashes.npy", np.dtype(np.uint64)),
        ("field-hashes.npy", "filter-field-hashes.npy", np.dtype(np.uint64)),
        ("title-hashes.npy", "title-hashes-unsorted.npy", np.dtype(np.uint64)),
    )
    for part_name, final_name, dtype in sidecars:
        _copy_sidecar_parts(
            results,
            directory,
            part_name=part_name,
            final_name=final_name,
            dtype=dtype,
            total=total,
        )

    logger.info("sorting %s normalized-title hashes", f"{total:,}")
    unsorted_hashes = np.load(directory / "title-hashes-unsorted.npy", mmap_mode="r")
    order = np.argsort(unsorted_hashes, kind="stable")
    sorted_hashes = np.lib.format.open_memmap(
        directory / "title-hashes.npy", mode="w+", dtype=np.uint64, shape=(total,)
    )
    sorted_rows = np.lib.format.open_memmap(
        directory / "title-rows.npy", mode="w+", dtype=np.uint32, shape=(total,)
    )
    copy_rows = 5_000_000
    for offset in range(0, total, copy_rows):
        selected = order[offset : offset + copy_rows]
        sorted_hashes[offset : offset + len(selected)] = unsorted_hashes[selected]
        sorted_rows[offset : offset + len(selected)] = selected.astype(np.uint32)
    sorted_hashes.flush()
    sorted_rows.flush()
    del order, sorted_hashes, sorted_rows, unsorted_hashes
    (directory / "title-hashes-unsorted.npy").unlink()

    year_counts: dict[int, int] = {}
    for result in results:
        for year, count in result.year_counts.items():
            year_counts[year] = year_counts.get(year, 0) + count
    _build_selective_year_index(directory, year_counts=year_counts, total=total)
    component_names = [
        "metadata.blocks",
        "metadata-offsets.npy",
        "metadata-row-starts.npy",
        "ranking-years.npy",
        "ranking-citations.npy",
        "filter-open-access.npy",
        "filter-topic-hashes.npy",
        "filter-field-hashes.npy",
        "title-hashes.npy",
        "title-rows.npy",
        "selective-year-values.npy",
        "selective-year-offsets.npy",
        "selective-year-rows.npy",
    ]
    metadata_bytes = sum((directory / name).stat().st_size for name in component_names)
    manifest = {
        "format_version": COMPACT_FORMAT_VERSION,
        "records": total,
        "block_rows": block_rows,
        "blocks": len(global_row_starts),
        "hash": "blake2b-64/openalex-v1",
        "year_counts": {str(year): count for year, count in sorted(year_counts.items())},
        "open_access_count": sum(result.open_access_count for result in results),
        "selective_filter_max_rows": SELECTIVE_FILTER_MAX_ROWS,
        "files": component_names,
        "checksums": {
            name: _sha256(directory / name) for name in component_names
        },
        "bytes": metadata_bytes,
    }
    (directory / "compact-metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The manifest is the marker-last commit for the compact layer. If the
    # process stops before this point, per-shard parts remain resumable. If it
    # stops after the manifest, the next resume can safely remove stale parts.
    shutil.rmtree(parts)
    return manifest


def validated_compact_manifest(directory: Path, *, expected_records: int) -> dict | None:
    """Return a completed compact manifest, or None for an incomplete checkpoint."""
    path = directory / "compact-metadata.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if int(manifest.get("format_version", -1)) != COMPACT_FORMAT_VERSION:
            return None
        if int(manifest.get("records", -1)) != expected_records:
            return None
        files = [directory / str(name) for name in manifest["files"]]
        if not files or not all(file.is_file() for file in files):
            return None
        if sum(file.stat().st_size for file in files) != int(manifest["bytes"]):
            return None
        checksums = manifest["checksums"]
        if set(checksums) != {file.name for file in files}:
            return None
        if any(_sha256(file) != str(checksums[file.name]) for file in files):
            return None
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return manifest


class CompactMetadataStore:
    """Read-only metadata and filter layer for the full-corpus generation."""

    def __init__(self, directory: Path, *, scan_budget: FilterScanBudget | None = None):
        self.directory = directory
        self.scan_budget = scan_budget or FilterScanBudget.from_environment()
        self.manifest = json.loads(
            (directory / "compact-metadata.json").read_text(encoding="utf-8")
        )
        if int(self.manifest.get("format_version", -1)) != COMPACT_FORMAT_VERSION:
            raise RuntimeError("Unsupported compact metadata format")
        self.record_count = int(self.manifest["records"])
        self.offsets = np.load(directory / "metadata-offsets.npy", mmap_mode="r")
        self.row_starts = np.load(directory / "metadata-row-starts.npy", mmap_mode="r")
        self.years = np.load(directory / "ranking-years.npy", mmap_mode="r")
        self.citations = np.load(directory / "ranking-citations.npy", mmap_mode="r")
        self.open_access = np.load(directory / "filter-open-access.npy", mmap_mode="r")
        self.topic_hashes = np.load(directory / "filter-topic-hashes.npy", mmap_mode="r")
        self.field_hashes = np.load(directory / "filter-field-hashes.npy", mmap_mode="r")
        self.title_hashes = np.load(directory / "title-hashes.npy", mmap_mode="r")
        self.title_rows = np.load(directory / "title-rows.npy", mmap_mode="r")
        self.selective_year_values = np.load(
            directory / "selective-year-values.npy", mmap_mode="r"
        )
        self.selective_year_offsets = np.load(
            directory / "selective-year-offsets.npy", mmap_mode="r"
        )
        self.selective_year_rows = np.load(
            directory / "selective-year-rows.npy", mmap_mode="r"
        )
        expected = (self.record_count,)
        for name, values in (
            ("years", self.years),
            ("citations", self.citations),
            ("open_access", self.open_access),
            ("topic_hashes", self.topic_hashes),
            ("field_hashes", self.field_hashes),
            ("title_hashes", self.title_hashes),
            ("title_rows", self.title_rows),
        ):
            if values.shape != expected:
                raise RuntimeError(f"Compact metadata {name} has shape {values.shape}; expected {expected}")
        if len(self.offsets) != len(self.row_starts) + 1:
            raise RuntimeError("Compact metadata block offsets do not align with row starts")
        if len(self.row_starts) == 0 or int(self.row_starts[0]) != 0:
            raise RuntimeError("Compact metadata must start at row zero")
        if np.any(np.diff(self.offsets) <= 0) or np.any(np.diff(self.row_starts) <= 0):
            raise RuntimeError("Compact metadata offsets and row starts must be increasing")
        if len(self.selective_year_offsets) != len(self.selective_year_values) + 1:
            raise RuntimeError("Selective-year offsets do not align with year values")
        if int(self.selective_year_offsets[-1]) != len(self.selective_year_rows):
            raise RuntimeError("Selective-year row count does not match its offsets")
        self._year_counts = {
            int(year): int(count)
            for year, count in self.manifest.get("year_counts", {}).items()
        }
        self._selective_year_lookup = {
            int(year): index for index, year in enumerate(self.selective_year_values)
        }
        self._blocks_file = open(directory / "metadata.blocks", "rb")
        self._blocks = mmap.mmap(self._blocks_file.fileno(), 0, access=mmap.ACCESS_READ)
        self._block_cache: OrderedDict[int, list[bytes]] = OrderedDict()
        self._filter_cache: OrderedDict[Filters, tuple[int, np.ndarray | None]] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._closed = False

    def close(self) -> None:
        with self._cache_lock:
            if self._closed:
                return
            self._closed = True
            self._blocks.close()
            self._blocks_file.close()

    def _read_block(self, block_index: int) -> list[bytes]:
        with self._cache_lock:
            if self._closed:
                raise RuntimeError("Compact metadata store is closed")
            cached = self._block_cache.get(block_index)
            if cached is not None:
                self._block_cache.move_to_end(block_index)
                return cached
            start = int(self.offsets[block_index])
            stop = int(self.offsets[block_index + 1])
            rows = zlib.decompress(self._blocks[start:stop]).splitlines()
            self._block_cache[block_index] = rows
            self._block_cache.move_to_end(block_index)
            while len(self._block_cache) > 256:
                self._block_cache.popitem(last=False)
            return rows

    @staticmethod
    def _record(record: dict, row_id: int) -> dict:
        return {
            "row_id": row_id,
            "openalex_id": str(record.get("openalex_id") or ""),
            "title": str(record.get("title") or ""),
            "normalized_title": normalize_title(str(record.get("title") or "")),
            "snippet": str(record.get("snippet") or ""),
            "authors": list(record.get("authors") or ()),
            "publication_year": int(record.get("publication_year") or 0),
            "doi": str(record.get("doi") or ""),
            "cited_by_count": int(record.get("cited_by_count") or 0),
            "topic": str(record.get("topic") or ""),
            "field": str(record.get("field") or ""),
            "is_oa": bool(record.get("is_oa")),
            "oa_url": str(record.get("oa_url") or ""),
            "work_type": str(record.get("work_type") or ""),
            "venue": str(record.get("venue") or ""),
            "publication_date": str(record.get("publication_date") or ""),
            "landing_url": str(record.get("landing_url") or ""),
        }

    def fetch(self, ids: Sequence[int]) -> dict[int, dict]:
        requested: dict[int, list[tuple[int, int]]] = {}
        for row_id in dict.fromkeys(int(value) for value in ids):
            if not 0 <= row_id < self.record_count:
                continue
            block = int(np.searchsorted(self.row_starts, row_id, side="right") - 1)
            local = row_id - int(self.row_starts[block])
            requested.setdefault(block, []).append((row_id, local))
        output: dict[int, dict] = {}
        for block, positions in requested.items():
            rows = self._read_block(block)
            for row_id, local in positions:
                if not 0 <= local < len(rows):
                    raise RuntimeError(f"Metadata row {row_id} is outside compact block {block}")
                output[row_id] = self._record(json.loads(rows[local]), row_id)
        return output

    def _filter_mask(self, ids: np.ndarray, filters: Filters) -> np.ndarray:
        mask = np.ones(len(ids), dtype=bool)
        if filters.year_min is not None or filters.year_max is not None:
            # Year 0 means "unknown"; a year filter must not admit it.
            mask &= self.years[ids] > 0
        if filters.year_min is not None:
            mask &= self.years[ids] >= filters.year_min
        if filters.year_max is not None:
            mask &= self.years[ids] <= filters.year_max
        if filters.min_citations is not None:
            mask &= self.citations[ids] >= filters.min_citations
        if filters.open_access_only:
            mask &= self.open_access[ids] == 1
        if filters.topic:
            mask &= self.topic_hashes[ids] == stable_text_hash(filters.topic)
        if filters.field:
            mask &= self.field_hashes[ids] == stable_text_hash(filters.field)
        return mask

    def filter_candidate_ids(self, ids: Sequence[int], filters: Filters) -> list[int]:
        if len(ids) == 0:
            return []
        values = np.asarray(ids, dtype=np.int64)
        if not filters.active:
            return values.tolist()
        return values[self._filter_mask(values, filters)].tolist()

    def _cached_filter(self, filters: Filters) -> tuple[int, np.ndarray | None] | None:
        with self._cache_lock:
            cached = self._filter_cache.get(filters)
            if cached is not None:
                self._filter_cache.move_to_end(filters)
            return cached

    def _remember_filter(self, filters: Filters, result: tuple[int, np.ndarray | None]) -> None:
        with self._cache_lock:
            self._filter_cache[filters] = result
            self._filter_cache.move_to_end(filters)
            id_bytes = sum(
                ids.nbytes for _, ids in self._filter_cache.values() if ids is not None
            )
            while len(self._filter_cache) > 1 and (
                len(self._filter_cache) > FILTER_CACHE_MAX_ENTRIES
                or id_bytes > FILTER_CACHE_MAX_ID_BYTES
            ):
                _, (_, evicted) = self._filter_cache.popitem(last=False)
                if evicted is not None:
                    id_bytes -= evicted.nbytes

    def _known_component(self, part: Filters) -> tuple[int, np.ndarray | None] | None:
        """A single-dimension result that costs no full scan, if one is known."""
        cached = self._cached_filter(part)
        if cached is not None:
            return cached
        if part.year_min is not None or part.year_max is not None:
            return self.count(part), self._year_candidate_ids(part)
        if part == Filters(open_access_only=True) and "open_access_count" in self.manifest:
            return int(self.manifest["open_access_count"]), None
        return None

    def _from_components(self, filters: Filters) -> tuple[int, np.ndarray | None] | None:
        """Answer a combined filter from its cheap parts, without a full scan.

        Any part matching nothing empties the whole filter; any part whose
        rows are materialized bounds the filter to masking those rows.
        """
        parts = _filter_components(filters)
        if len(parts) < 2:
            return None
        known = [self._known_component(part) for part in parts]
        if any(item is not None and item[0] == 0 for item in known):
            return 0, np.empty(0, dtype=np.int64)
        bounded = [item[1] for item in known if item is not None and item[1] is not None]
        if not bounded:
            return None
        rows = min(bounded, key=len)
        matched = rows[self._filter_mask(rows, filters)]
        return int(len(matched)), matched

    def _scan_filters(self, filters: Filters) -> tuple[int, np.ndarray | None]:
        cached = self._cached_filter(filters)
        if cached is not None:
            return cached
        derived = self._from_components(filters)
        if derived is not None:
            self._remember_filter(filters, derived)
            return derived
        # Raises FilterScanBudgetExceeded instead of starting another scan.
        self.scan_budget.acquire()
        parts = _filter_components(filters)
        # The same pass records each part of a combined filter, so a later
        # combination sharing a part can be answered or bounded from it.
        targets = [filters, *parts] if len(parts) > 1 else [filters]
        counts = [0] * len(targets)
        collected: list[list[np.ndarray]] = [[] for _ in targets]
        keep_ids = [True] * len(targets)
        chunk_rows = 5_000_000
        for start in range(0, self.record_count, chunk_rows):
            stop = min(start + chunk_rows, self.record_count)
            ids = np.arange(start, stop, dtype=np.int64)
            if len(targets) == 1:
                masks = [self._filter_mask(ids, filters)]
            else:
                part_masks = [self._filter_mask(ids, part) for part in parts]
                masks = [np.logical_and.reduce(part_masks), *part_masks]
            for index, mask in enumerate(masks):
                matched = np.flatnonzero(mask).astype(np.int64)
                matched += start
                counts[index] += len(matched)
                if keep_ids[index] and counts[index] <= SELECTIVE_FILTER_MAX_ROWS:
                    collected[index].append(matched)
                else:
                    keep_ids[index] = False
                    collected[index] = []
        results = [
            (
                count,
                np.concatenate(chunks) if keep and chunks else (
                    np.empty(0, dtype=np.int64) if count == 0 else None
                ),
            )
            for count, chunks, keep in zip(counts, collected, keep_ids)
        ]
        for target, result in zip(targets[1:], results[1:]):
            self._remember_filter(target, result)
        self._remember_filter(filters, results[0])
        return results[0]

    def _year_candidate_ids(self, filters: Filters) -> np.ndarray | None:
        if filters.year_min is None and filters.year_max is None:
            return None
        lower = max(1, filters.year_min if filters.year_min is not None else 0)
        upper = filters.year_max if filters.year_max is not None else 32767
        matching_years = [
            year
            for year, count in self._year_counts.items()
            if count > 0 and lower <= year <= upper
        ]
        total = sum(self._year_counts[year] for year in matching_years)
        if total > SELECTIVE_FILTER_MAX_ROWS:
            return None
        parts: list[np.ndarray] = []
        for year in matching_years:
            lookup = self._selective_year_lookup.get(year)
            if lookup is None:
                return None
            start = int(self.selective_year_offsets[lookup])
            stop = int(self.selective_year_offsets[lookup + 1])
            parts.append(np.asarray(self.selective_year_rows[start:stop], dtype=np.int64))
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.sort(np.concatenate(parts))

    def count(self, filters: Filters = Filters()) -> int:
        if not filters.active:
            return self.record_count
        only_year_range = not any(
            (
                filters.min_citations is not None,
                filters.open_access_only,
                filters.topic is not None,
                filters.field is not None,
            )
        )
        if only_year_range:
            lower = max(1, filters.year_min if filters.year_min is not None else 0)
            upper = filters.year_max if filters.year_max is not None else 32767
            return sum(
                count
                for year, count in self._year_counts.items()
                if lower <= year <= upper
            )
        year_candidates = self._year_candidate_ids(filters)
        if year_candidates is not None:
            return int(np.count_nonzero(self._filter_mask(year_candidates, filters)))
        only_open_access = filters == Filters(open_access_only=True)
        if only_open_access:
            return int(self.manifest.get("open_access_count", 0))
        return self._scan_filters(filters)[0]

    def eligible_ids(self, filters: Filters) -> list[int]:
        year_candidates = self._year_candidate_ids(filters)
        if year_candidates is not None:
            return year_candidates[self._filter_mask(year_candidates, filters)].tolist()
        count, ids = self._scan_filters(filters)
        if ids is None:
            raise RuntimeError(
                f"Filter matches {count:,} rows; selective materialization "
                f"is capped at {SELECTIVE_FILTER_MAX_ROWS:,}"
            )
        return ids.tolist()

    def _title_span(self, normalized: str) -> tuple[int, int]:
        value = np.uint64(stable_text_hash(normalized))
        start = int(np.searchsorted(self.title_hashes, value, side="left"))
        if start >= len(self.title_hashes) or self.title_hashes[start] != value:
            return start, start
        return start, int(np.searchsorted(self.title_hashes, value, side="right"))

    def title_key_exists(self, normalized: str) -> bool:
        """Whether any row's normalized title hashes to ``normalized``.

        One binary search and no metadata read, so a caller can probe several
        spellings cheaply; rows it then uses still go through title_search,
        which verifies the stored title.
        """
        if not normalized:
            return False
        start, stop = self._title_span(normalized)
        return stop > start

    def title_search(self, query: str, filters: Filters, limit: int = 20) -> list[int]:
        normalized = normalize_title(query)
        if not normalized:
            return []
        start, stop = self._title_span(normalized)
        if start == stop:
            return []
        # Bound pathological duplicate titles; semantic retrieval still supplies
        # the broader candidate set.
        rows = np.asarray(self.title_rows[start : min(stop, start + 100_000)], dtype=np.int64)
        if filters.active:
            rows = rows[self._filter_mask(rows, filters)]
        # Rank by citations from the sidecar before reading any metadata, then
        # verify in that order and stop once the limit is met. Verifying every
        # row first decompressed a block per row: a generic title such as
        # "Introduction" read around 100,000 blocks and held a request thread
        # for minutes. Storage order is not citation order, so ranking first is
        # also what keeps the most-cited works when the cap binds.
        citations = np.asarray(self.citations[rows], dtype=np.int64)
        rows = rows[np.lexsort((rows, -citations))]
        exact: list[int] = []
        step = max(limit, 32)
        budget = max(4 * limit, 256)
        for begin in range(0, min(len(rows), budget), step):
            chunk = rows[begin : begin + step].tolist()
            metadata = self.fetch(chunk)
            exact.extend(
                row_id
                for row_id in chunk
                if metadata[row_id]["normalized_title"] == normalized
            )
            if len(exact) >= limit:
                break
        return exact[:limit]

    def rarest_year(self) -> tuple[int, int] | None:
        counts = {year: count for year, count in self._year_counts.items() if year > 0 and count > 0}
        if not counts:
            return None
        year = min(counts, key=lambda value: (counts[value], value))
        return year, counts[year]
