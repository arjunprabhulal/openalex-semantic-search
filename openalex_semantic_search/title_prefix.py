"""Leading-word-prefix lookup for titles.

``CompactMetadataStore.title_search`` matches a normalized title exactly, by
hash. That is the right primary signal, but it fails the moment a reader types
part of a title: at full corpus scale "Attention Is All You" does not match
"Attention Is All You Need", and the semantic path does not rescue it either,
because a short query scores poorly against an embedding built from a title
plus a long abstract. The paper falls outside the candidate set entirely.

This builds a second sorted hash index over the leading word prefixes of every
title, in the same shape as ``title-hashes.npy``/``title-rows.npy``:

    title-prefix-hashes.npy   uint64, sorted
    title-prefix-rows.npy     uint32, parallel row ids

It is produced by a standalone pass over an existing generation's metadata
blocks and written to a directory of the caller's choosing, so the immutable
generation and its ``generation-files.sha256`` are never touched.

Prefixes shorter than ``MIN_PREFIX_WORDS`` are not indexed: one or two words
match far too many titles to be a useful lookup signal, and semantic retrieval
already covers that case. The full title is not indexed either, because the
exact index already answers it.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np

from .store import normalize_title

logger = logging.getLogger(__name__)

TITLE_PREFIX_FORMAT_VERSION = 1
HASHES_FILENAME = "title-prefix-hashes.npy"
ROWS_FILENAME = "title-prefix-rows.npy"
MANIFEST_FILENAME = "title-prefix.json"

# A two-word prefix matches an unusable number of titles; beyond a dozen words a
# reader is effectively typing the whole title, which the exact index handles.
MIN_PREFIX_WORDS = 3
MAX_PREFIX_WORDS = 12

# Bound on rows returned for one prefix. A common opening ("a study of the")
# can front thousands of titles; ranking still runs over the merged candidate
# set, so a cap costs recall only on prefixes that were never selective.
MAX_ROWS_PER_PREFIX = 10_000


def prefix_hashes(title: str, hasher) -> list[int]:
    """Hashes of each indexable leading word prefix of ``title``."""
    words = normalize_title(title).split()
    if len(words) <= MIN_PREFIX_WORDS:
        # Too short to have a prefix worth indexing that is not the title.
        return []
    stop = min(len(words) - 1, MAX_PREFIX_WORDS)
    return [hasher(" ".join(words[:count])) for count in range(MIN_PREFIX_WORDS, stop + 1)]


@dataclass(frozen=True, slots=True)
class TitlePrefixIndex:
    """Sorted prefix-hash index loaded alongside a generation."""

    hashes: np.ndarray
    rows: np.ndarray
    directory: Path
    # The build manifest, kept so the engine can refuse an index that was
    # built for a different generation (its row ids would name other works).
    manifest: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.hashes))

    @classmethod
    def load(cls, directory: Path | None) -> "TitlePrefixIndex | None":
        if directory is None:
            return None
        manifest_path = directory / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"No title prefix index in {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = int(manifest.get("format_version", -1))
        if version != TITLE_PREFIX_FORMAT_VERSION:
            raise ValueError(
                f"{manifest_path} declares format_version {version}; "
                f"this build reads {TITLE_PREFIX_FORMAT_VERSION}"
            )
        hashes = np.load(directory / HASHES_FILENAME, mmap_mode="r")
        rows = np.load(directory / ROWS_FILENAME, mmap_mode="r")
        if hashes.shape != rows.shape:
            raise RuntimeError("Title prefix hash and row sidecars disagree in length")
        logger.info("loaded %d title prefixes from %s", len(hashes), directory)
        return cls(hashes=hashes, rows=rows, directory=directory, manifest=manifest)

    def _span(self, query: str, hasher) -> tuple[int, int]:
        words = normalize_title(query).split()
        if not MIN_PREFIX_WORDS <= len(words) <= MAX_PREFIX_WORDS:
            return 0, 0
        value = np.uint64(hasher(" ".join(words)))
        start = int(np.searchsorted(self.hashes, value, side="left"))
        if start >= len(self.hashes) or self.hashes[start] != value:
            return start, start
        return start, int(np.searchsorted(self.hashes, value, side="right"))

    def contains(self, query: str, hasher) -> bool:
        """Whether any title begins with ``query``; one binary search."""
        start, stop = self._span(query, hasher)
        return stop > start

    def lookup(
        self, query: str, hasher, *, max_rows: int = MAX_ROWS_PER_PREFIX
    ) -> list[int]:
        """Row ids whose title begins with ``query``.

        Returns an empty list when the query is too short to be indexed, so a
        caller can fall back to semantic retrieval without special-casing.
        Rows come back in storage order, which is not citation order; a caller
        that keeps only some of them should rank them first.
        """
        start, stop = self._span(query, hasher)
        if start == stop:
            return []
        stop = min(stop, start + max_rows)
        return np.asarray(self.rows[start:stop], dtype=np.int64).tolist()


def _scan_blocks(
    generation: str,
    spill: str,
    worker_index: int,
    first_block: int,
    stop_block: int,
    shift: int,
) -> int:
    """Emit prefix hashes for a range of compact blocks.

    Walks blocks in order and reads each exactly once. ``CompactMetadataStore
    .fetch`` is deliberately not used: it is built for pulling a handful of rows
    per query and costs an ``np.searchsorted`` over a memmap per row, which
    measures 180 records/second against 21,700 for decoding the same rows
    straight from their block. That difference is four hours versus five weeks
    at corpus scale.
    """
    import json as _json
    import zlib

    from .compact_store import CompactMetadataStore, stable_text_hash

    store = CompactMetadataStore(Path(generation))
    handles: dict[int, tuple] = {}
    emitted = 0
    try:
        for block in range(first_block, stop_block):
            start_row = int(store.row_starts[block])
            begin = int(store.offsets[block])
            end = int(store.offsets[block + 1])
            lines = zlib.decompress(store._blocks[begin:end]).splitlines()
            pending: dict[int, list[tuple[int, int]]] = {}
            for local, line in enumerate(lines):
                if not line:
                    continue
                title = _json.loads(line).get("title") or ""
                row_id = start_row + local
                for value in prefix_hashes(str(title), stable_text_hash):
                    pending.setdefault(value >> shift, []).append((value, row_id))
            for bucket, pairs in pending.items():
                if bucket not in handles:
                    handles[bucket] = (
                        open(Path(spill) / f"{bucket:04d}.{worker_index:03d}.h", "ab"),
                        open(Path(spill) / f"{bucket:04d}.{worker_index:03d}.r", "ab"),
                    )
                hfile, rfile = handles[bucket]
                hfile.write(np.asarray([pair[0] for pair in pairs], dtype=np.uint64).tobytes())
                rfile.write(np.asarray([pair[1] for pair in pairs], dtype=np.uint32).tobytes())
                emitted += len(pairs)
    finally:
        store.close()
        for hfile, rfile in handles.values():
            hfile.close()
            rfile.close()
    return emitted


def build_title_prefix_index(
    generation: Path,
    output: Path,
    *,
    hasher,
    store_factory,
    batch_rows: int = 200_000,
    buckets: int = 256,
    workers: int = 1,
    progress_every: int = 10_000_000,
) -> dict:
    """Scan a generation's titles and write a sorted prefix index to ``output``.

    ``generation`` is only read. ``store_factory(generation)`` supplies a store
    exposing ``record_count``, ``fetch`` and ``close``.

    Memory is bounded by bucketing rather than sorting everything at once. At
    full corpus scale this index holds billions of entries, and a single
    in-memory argsort over them would need tens of gigabytes on a host that is
    also serving the index. Because the hashes are uniformly distributed,
    partitioning on their high bits and sorting each bucket independently
    yields a globally sorted array when the buckets are concatenated in order,
    with peak memory set by one bucket instead of the whole index.
    """
    output.mkdir(parents=True, exist_ok=True)
    spill = output / ".buckets"
    spill.mkdir(exist_ok=True)
    shift = 64 - max(1, buckets.bit_length() - 1)

    if workers > 1:
        probe = store_factory(generation)
        try:
            total_records = int(probe.record_count)
            total_blocks = int(len(probe.row_starts))
        finally:
            probe.close()
        span = (total_blocks + workers - 1) // workers
        emitted = 0
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {
                pool.submit(
                    _scan_blocks,
                    str(generation),
                    str(spill),
                    index,
                    index * span,
                    min((index + 1) * span, total_blocks),
                    shift,
                ): index
                for index in range(workers)
                if index * span < total_blocks
            }
            done = 0
            for future in as_completed(futures):
                emitted += int(future.result())
                done += 1
                logger.info("title prefixes: %d/%d workers done", done, len(futures))
    else:
        store = store_factory(generation)
        handles: dict[int, tuple] = {}
        total_records = 0
        emitted = 0
        try:
            total_records = int(store.record_count)
            for start in range(0, total_records, batch_rows):
                ids = list(range(start, min(start + batch_rows, total_records)))
                records = store.fetch(ids)
                pending: dict[int, list[tuple[int, int]]] = {}
                for row_id in ids:
                    record = records.get(row_id)
                    if record is None:
                        continue
                    for value in prefix_hashes(str(record.get("title") or ""), hasher):
                        pending.setdefault(value >> shift, []).append((value, row_id))
                for bucket, pairs in pending.items():
                    if bucket not in handles:
                        handles[bucket] = (
                            open(spill / f"{bucket:04d}.000.h", "wb"),
                            open(spill / f"{bucket:04d}.000.r", "wb"),
                        )
                    hfile, rfile = handles[bucket]
                    hfile.write(np.asarray([p[0] for p in pairs], dtype=np.uint64).tobytes())
                    rfile.write(np.asarray([p[1] for p in pairs], dtype=np.uint32).tobytes())
                    emitted += len(pairs)
                if progress_every and start and start % progress_every < batch_rows:
                    logger.info(
                        "title prefixes: %d/%d records scanned, %d prefixes",
                        start, total_records, emitted,
                    )
        finally:
            store.close()
            for hfile, rfile in handles.values():
                hfile.close()
                rfile.close()

    hashes_out = np.lib.format.open_memmap(
        output / HASHES_FILENAME, mode="w+", dtype=np.uint64, shape=(emitted,)
    )
    rows_out = np.lib.format.open_memmap(
        output / ROWS_FILENAME, mode="w+", dtype=np.uint32, shape=(emitted,)
    )
    cursor = 0
    # Ascending bucket order is ascending hash order, so each sorted bucket can
    # be appended without any merge step.
    bucket_ids = sorted({int(path.name.split(".")[0]) for path in spill.glob("*.h")})
    for bucket in bucket_ids:
        parts_h = sorted(spill.glob(f"{bucket:04d}.*.h"))
        chunk_h = np.concatenate(
            [np.fromfile(path, dtype=np.uint64) for path in parts_h]
        ) if parts_h else np.empty(0, dtype=np.uint64)
        parts_r = sorted(spill.glob(f"{bucket:04d}.*.r"))
        chunk_r = np.concatenate(
            [np.fromfile(path, dtype=np.uint32) for path in parts_r]
        ) if parts_r else np.empty(0, dtype=np.uint32)
        order = np.argsort(chunk_h, kind="stable")
        stop = cursor + len(order)
        hashes_out[cursor:stop] = chunk_h[order]
        rows_out[cursor:stop] = chunk_r[order]
        cursor = stop
        for path in parts_h + parts_r:
            path.unlink()
    hashes_out.flush()
    rows_out.flush()
    del hashes_out, rows_out
    spill.rmdir()

    if cursor != emitted:
        raise RuntimeError(f"Wrote {cursor:,} prefixes; counted {emitted:,}")

    manifest = {
        "format_version": TITLE_PREFIX_FORMAT_VERSION,
        "generation": generation.name,
        "records": total_records,
        "prefixes": emitted,
        "buckets": buckets,
        "workers": workers,
        "min_prefix_words": MIN_PREFIX_WORDS,
        "max_prefix_words": MAX_PREFIX_WORDS,
    }
    (output / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("wrote %d title prefixes for %d records", emitted, total_records)
    return manifest
