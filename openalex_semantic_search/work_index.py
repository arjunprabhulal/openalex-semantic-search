"""Lookup of a single work by its OpenAlex id.

The compact store indexes titles, not identifiers, so a full generation can
answer "which work is called this" but not "which work is this". Retrieval
never needed the latter: every path reaches a row through the candidate index
or a title hash. A caller holding an id from an earlier result does need it,
and searching for the id as query text cannot work, because an OpenAlex id
appears in neither an embedding nor a title.

This is the same sorted-hash shape as ``title-hashes.npy``, one entry per
record rather than several:

    work-id-hashes.npy   uint64, sorted
    work-id-rows.npy     uint32, parallel row ids

Built by a standalone pass over an existing generation and written wherever
the caller asks, so the immutable generation is untouched and no reassembly is
needed. Hash collisions are resolved by comparing the stored id, so a
collision costs a metadata read rather than returning the wrong paper.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

WORK_INDEX_FORMAT_VERSION = 1
HASHES_FILENAME = "work-id-hashes.npy"
ROWS_FILENAME = "work-id-rows.npy"
MANIFEST_FILENAME = "work-id-index.json"


def normalize_work_id(value: str) -> str:
    """Return the bare ``W...`` identifier for any accepted spelling."""
    text = (value or "").strip().split("?")[0].split("#")[0].rstrip("/")
    if not text:
        return ""
    tail = text.rsplit("/", 1)[-1]
    if tail.lower().startswith("w") and tail[1:].isdigit():
        return "W" + tail[1:]
    return ""


def _scan_blocks(
    generation: str,
    spill: str,
    worker_index: int,
    first_block: int,
    stop_block: int,
    shift: int,
) -> int:
    """Emit one id hash per record for a range of compact blocks."""
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
                work_id = normalize_work_id(str(_json.loads(line).get("openalex_id") or ""))
                if not work_id:
                    continue
                value = stable_text_hash(work_id)
                pending.setdefault(value >> shift, []).append((value, start_row + local))
            for bucket, pairs in pending.items():
                if bucket not in handles:
                    handles[bucket] = (
                        open(Path(spill) / f"{bucket:04d}.{worker_index:03d}.h", "ab"),
                        open(Path(spill) / f"{bucket:04d}.{worker_index:03d}.r", "ab"),
                    )
                hfile, rfile = handles[bucket]
                hfile.write(np.asarray([p[0] for p in pairs], dtype=np.uint64).tobytes())
                rfile.write(np.asarray([p[1] for p in pairs], dtype=np.uint32).tobytes())
                emitted += len(pairs)
    finally:
        store.close()
        for hfile, rfile in handles.values():
            hfile.close()
            rfile.close()
    return emitted


@dataclass(frozen=True, slots=True)
class WorkIdIndex:
    """Sorted id-hash index loaded alongside a generation."""

    hashes: np.ndarray
    rows: np.ndarray
    directory: Path
    # Build manifest, checked by the engine against the served generation.
    manifest: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.hashes))

    @classmethod
    def load(cls, directory: Path | None) -> "WorkIdIndex | None":
        if directory is None:
            return None
        manifest_path = directory / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"No work id index in {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = int(manifest.get("format_version", -1))
        if version != WORK_INDEX_FORMAT_VERSION:
            raise ValueError(
                f"{manifest_path} declares format_version {version}; "
                f"this build reads {WORK_INDEX_FORMAT_VERSION}"
            )
        hashes = np.load(directory / HASHES_FILENAME, mmap_mode="r")
        rows = np.load(directory / ROWS_FILENAME, mmap_mode="r")
        if hashes.shape != rows.shape:
            raise RuntimeError("Work id hash and row sidecars disagree in length")
        logger.info("loaded %d work ids from %s", len(hashes), directory)
        return cls(hashes=hashes, rows=rows, directory=directory, manifest=manifest)

    def candidate_rows(self, work_id: str, hasher) -> list[int]:
        """Rows whose id hashes to ``work_id``; usually one, never many.

        The caller must confirm the id on the returned row. A 64-bit collision
        is vanishingly unlikely but returning the wrong paper is not a failure
        mode worth accepting for a lookup by identifier.
        """
        normalized = normalize_work_id(work_id)
        if not normalized:
            return []
        value = np.uint64(hasher(normalized))
        start = int(np.searchsorted(self.hashes, value, side="left"))
        stop = int(np.searchsorted(self.hashes, value, side="right"))
        if start == stop:
            return []
        return np.asarray(self.rows[start:stop], dtype=np.int64).tolist()


def build_work_id_index(
    generation: Path,
    output: Path,
    *,
    store_factory,
    buckets: int = 256,
    workers: int = 1,
) -> dict:
    """Scan a generation's ids and write a sorted lookup index to ``output``."""
    output.mkdir(parents=True, exist_ok=True)
    spill = output / ".buckets"
    spill.mkdir(exist_ok=True)
    shift = 64 - max(1, buckets.bit_length() - 1)

    probe = store_factory(generation)
    try:
        total_records = int(probe.record_count)
        total_blocks = int(len(probe.row_starts))
    finally:
        probe.close()

    span = (total_blocks + workers - 1) // workers
    emitted = 0
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=context) as pool:
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
            logger.info("work ids: %d/%d workers done", done, len(futures))

    hashes_out = np.lib.format.open_memmap(
        output / HASHES_FILENAME, mode="w+", dtype=np.uint64, shape=(emitted,)
    )
    rows_out = np.lib.format.open_memmap(
        output / ROWS_FILENAME, mode="w+", dtype=np.uint32, shape=(emitted,)
    )
    cursor = 0
    # Ascending bucket order is ascending hash order, so sorted buckets
    # concatenate without a merge step.
    for bucket in sorted({int(p.name.split(".")[0]) for p in spill.glob("*.h")}):
        parts_h = sorted(spill.glob(f"{bucket:04d}.*.h"))
        parts_r = sorted(spill.glob(f"{bucket:04d}.*.r"))
        chunk_h = np.concatenate([np.fromfile(p, dtype=np.uint64) for p in parts_h])
        chunk_r = np.concatenate([np.fromfile(p, dtype=np.uint32) for p in parts_r])
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
        raise RuntimeError(f"Wrote {cursor:,} ids; counted {emitted:,}")

    manifest = {
        "format_version": WORK_INDEX_FORMAT_VERSION,
        "generation": generation.name,
        "records": total_records,
        "ids": emitted,
        "buckets": buckets,
        "workers": workers,
    }
    (output / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("wrote %d work ids for %d records", emitted, total_records)
    return manifest
