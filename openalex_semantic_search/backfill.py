"""GPU/remote backfill worker: stream the snapshot, embed, emit shard artifacts.

Runs on a rented GPU box (or anywhere) using only public inputs. Emits, per
shard of `shard_size` usable works, written atomically (tmp + rename):

- shard-NNNNN.int8.npy      INT8-quantized normalized embeddings
- shard-NNNNN.ids.npy       int64 numeric OpenAlex work ids, row-aligned
- shard-NNNNN.meta.jsonl.gz parsed card/filter metadata, row-aligned

Plus, at the artifact root: int8-scales.npy (calibrated on the first shard and
frozen), float32-truth-sample.npy + float32-truth-ids.npy (strided exact
vectors for benchmark ground truth and quantization-fidelity checks),
checkpoint.json (resume position), and backfill-manifest.json (provenance:
model, snapshot, config, per-file SHA-256).

Restarts are safe: completed shards are detected, the dedupe set is rebuilt
from shard id files, and streaming resumes from the checkpointed
(file_index, line_index) position.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
from queue import Empty, Full, Queue
import subprocess
from threading import Event, Semaphore
import time
from typing import Callable, Iterable, TextIO

import numpy as np

from .embeddings import Embedder
from .quantization import INT8_LEVELS, quantize_normalized
from .records import Paper, iter_s3_manifest_papers, parse_work

logger = logging.getLogger(__name__)

TRUTH_STRIDE_DEFAULT = 2_048  # every Nth record keeps its float32 vector
DEDUPE_MERGE_INTERVAL = 50_000_000
_SHARD_PREFETCH_DEPTH = 1


@dataclass(slots=True)
class _PreparedShard:
    papers: list[Paper]
    position: tuple[int, int]


@dataclass(frozen=True, slots=True)
class _ShardProducerFinished:
    source_exhausted: bool


@dataclass(frozen=True, slots=True)
class _ShardProducerFailure:
    error: BaseException


def _code_provenance() -> dict:
    """Best-effort source identity for reproducible rented-GPU artifacts."""
    source_root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={source_root}",
                "-C",
                str(source_root),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={source_root}",
                "-C",
                str(source_root),
                "status",
                "--porcelain",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"git_commit": revision, "git_dirty": bool(status.strip())}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def _atomic_write(path: Path, writer) -> None:
    """Publish one artifact durably with a same-directory atomic rename."""
    # Prefix (not suffix) the temp marker: np.save appends ".npy" to names
    # that lack it, which would break the rename.
    tmp = path.with_name(".tmp-" + path.name)
    writer(tmp)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    tmp.replace(path)
    # Persist the directory entry as well as the file contents. This matters
    # on rented/ephemeral builders where a host failure can follow a rename.
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _temporary_path(path: Path) -> Path:
    return path.with_name(".tmp-" + path.name)


def _publish_temporary(path: Path) -> None:
    """Durably publish a same-directory temporary artifact.

    Unlike :func:`_atomic_write`, this supports files populated incrementally
    through a memmap or gzip stream. The shard's ``.done`` marker is still the
    group commit: a crash between individual renames leaves uncommitted files
    which resume removes before rebuilding the shard.
    """
    temporary = _temporary_path(path)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _numeric_id(paper: Paper) -> int:
    return int(paper.openalex_id.rsplit("W", 1)[1])


def _backfill_config(
    embedder: Embedder,
    *,
    truth_stride: int,
    include_expansion: bool,
) -> dict:
    """Return corpus/quality settings which must not change on resume.

    A finite record limit is deliberately absent: increasing that limit is the
    supported way to continue from one validated round to the next. Shard size
    and embedding subchunk are also absent because they are safe to change at
    a committed marker boundary and are recorded per shard instead.
    """
    model = getattr(embedder, "model", None)
    effective_max_tokens = int(getattr(model, "max_seq_length", 0)) or None
    return {
        "format_version": 1,
        "embedder": {
            "name": str(embedder.name),
            "dimension": int(embedder.dimension),
            "dtype": str(getattr(embedder, "dtype", "float32")),
            "max_tokens": effective_max_tokens,
            "batch_size": getattr(embedder, "batch_size", None),
        },
        "truth_stride": truth_stride,
        "include_expansion": include_expansion,
    }


def _pin_backfill_config(output: Path, config: dict) -> None:
    """Create or verify the immutable, resume-critical build configuration."""
    path = output / "backfill-config-pin.json"
    if path.exists():
        pinned = json.loads(path.read_text(encoding="utf-8"))
        if pinned != config:
            changed = sorted(
                key for key in set(pinned) | set(config) if pinned.get(key) != config.get(key)
            )
            raise RuntimeError(
                "Backfill configuration changed since this artifact set started "
                f"({', '.join(changed)}). Resume with the pinned settings or use "
                "a fresh artifact directory."
            )
        return
    if any(output.glob("shard-*.done")):
        raise RuntimeError(
            "Existing shard markers have no backfill-config-pin.json; refusing "
            "to guess the artifact-shaping settings for a resume"
        )
    _atomic_write(
        path,
        lambda temporary: temporary.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        ),
    )


class SeenIds:
    """Memory-bounded dedupe: completed shards live in one sorted int64 array
    (8 bytes/id — ~2.7GB at 340.8M). New IDs stay in a set until a large,
    bounded merge interval; merging after every 1M shard would repeatedly copy
    the corpus-sized array and become quadratic over a full backfill."""

    def __init__(self, merged: np.ndarray | None = None):
        self.merged = merged if merged is not None else np.empty(0, dtype=np.int64)
        self.pending: set[int] = set()

    def __contains__(self, key: int) -> bool:
        if key in self.pending:
            return True
        position = int(np.searchsorted(self.merged, key))
        return position < len(self.merged) and int(self.merged[position]) == key

    def add(self, key: int) -> None:
        self.pending.add(key)

    def merge_pending(self) -> None:
        if self.pending:
            incoming = np.sort(np.fromiter(self.pending, dtype=np.int64))
            # O(n) merge of two sorted arrays instead of a full re-sort.
            positions = np.searchsorted(self.merged, incoming)
            self.merged = np.insert(self.merged, positions, incoming)
            self.pending = set()

    def merge_if_due(self, interval: int = DEDUPE_MERGE_INTERVAL) -> None:
        if len(self.pending) >= interval:
            self.merge_pending()


def _resume_state(output: Path) -> tuple[dict, SeenIds]:
    """Per-shard .done markers are the single source of truth. A crash between
    shard-file renames and the marker write leaves files without a marker;
    those orphans are deleted and the shard is redone — never overwritten in
    place with different records."""
    markers = sorted(output.glob("shard-*.done"))
    for index, marker in enumerate(markers):
        if int(marker.name[len("shard-"):-len(".done")]) != index:
            raise RuntimeError(f"Non-contiguous shard markers at {marker.name}")
    next_shard = len(markers)
    for temporary in output.glob(".tmp-*"):
        if temporary.is_file():
            logger.warning("removing temporary artifact from interrupted shard: %s", temporary.name)
            temporary.unlink()
    scales_path = output / "int8-scales.npy"
    if not markers:
        # Scales are calibrated from all rows of shard zero. Without its marker
        # they are uncommitted, even if the atomic file rename completed.
        scales_path.unlink(missing_ok=True)
    elif not scales_path.exists():
        raise RuntimeError("Committed shards exist but int8-scales.npy is missing")
    for orphan in output.glob("shard-*"):
        if orphan.suffix == ".done":
            continue
        shard_no = int(orphan.name.split("-")[1].split(".")[0])
        if shard_no >= next_shard:
            logger.warning("removing orphan artifact from interrupted shard: %s", orphan.name)
            orphan.unlink()
    if markers:
        last = json.loads(markers[-1].read_text(encoding="utf-8"))
        state = {
            "next_shard": next_shard,
            "file_index": last["file_index"],
            "line_index": last["line_index"],
            "records_done": last["records_done"],
        }
    else:
        state = {"next_shard": 0, "file_index": 0, "line_index": 0, "records_done": 0}
    ids: list[np.ndarray] = [
        np.load(output / f"shard-{index:05d}.ids.npy") for index in range(next_shard)
    ]
    merged = np.sort(np.concatenate(ids)) if ids else None
    return state, SeenIds(merged)


EMBED_SUBCHUNK_DEFAULT = 65_536


def _close_memmap(array: np.memmap | None) -> None:
    if array is None:
        return
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and mapping.closed:
        return
    array.flush()
    if mapping is not None:
        mapping.close()


def _write_embedded_chunk(
    matrix: np.ndarray,
    papers: list[Paper],
    offset: int,
    *,
    scales: np.ndarray | None,
    vectors: np.memmap,
    float_scratch: np.memmap | None,
    ids: np.memmap,
    truth: np.memmap,
    truth_ids: np.memmap,
    metadata: TextIO,
    truth_stride: int,
    calibration_maxima: np.ndarray | None,
) -> tuple[int, int]:
    """CPU half of the two-buffer shard pipeline.

    There is exactly one writer thread, so gzip order and all row-aligned
    memmaps stay deterministic. While this function consumes one matrix, the
    caller can submit the next bounded embedding batch to the GPU.
    """
    stop = offset + len(papers)
    numeric_ids = np.fromiter((_numeric_id(paper) for paper in papers), dtype=np.int64)
    ids[offset:stop] = numeric_ids

    if float_scratch is not None:
        float_scratch[offset:stop] = matrix
        if calibration_maxima is None:  # pragma: no cover - caller invariant
            raise RuntimeError("calibration maxima are required with Float32 scratch")
        np.maximum(calibration_maxima, np.max(np.abs(matrix), axis=0), out=calibration_maxima)
        clipped = 0
        values = 0
    else:
        if scales is None:  # pragma: no cover - caller invariant
            raise RuntimeError("INT8 scales are required without Float32 scratch")
        clipped = int(np.count_nonzero(np.abs(matrix / scales) > INT8_LEVELS))
        values = int(matrix.size)
        vectors[offset:stop] = quantize_normalized(matrix, scales)

    first_pick = (-offset) % truth_stride
    if first_pick < len(papers):
        local_picks = np.arange(first_pick, len(papers), truth_stride)
        truth_offset = (offset + first_pick) // truth_stride
        truth_stop = truth_offset + len(local_picks)
        truth[truth_offset:truth_stop] = matrix[local_picks]
        truth_ids[truth_offset:truth_stop] = numeric_ids[local_picks]

    for paper in papers:
        row = asdict(paper)
        row.pop("embedding_text", None)
        metadata.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return clipped, values


def _flush_shard(
    output: Path,
    shard: int,
    papers: list[Paper],
    embedder: Embedder,
    scales: np.ndarray | None,
    truth_stride: int,
    global_offset: int,
    embed_subchunk: int = EMBED_SUBCHUNK_DEFAULT,
) -> tuple[np.ndarray, float]:
    """Embed and publish one shard with bounded resident memory.

    At most two Float32 embedding subchunks are live: one being produced by
    the GPU and one being quantized/written by a CPU thread. For the first
    shard only, exact whole-shard calibration requires a temporary Float32
    memmap. This preserves the previous calibration result without holding the
    whole matrix in RAM; subsequent shards write INT8 directly.
    """
    started = time.perf_counter()
    if not papers:
        raise ValueError("cannot flush an empty shard")
    if embed_subchunk <= 0:
        raise ValueError("embed_subchunk must be positive")

    dimension = int(embedder.dimension)
    record_count = len(papers)
    truth_count = (record_count + truth_stride - 1) // truth_stride
    final_paths = {
        "truth": output / f"shard-{shard:05d}.truth.npy",
        "truth_ids": output / f"shard-{shard:05d}.tids.npy",
        "vectors": output / f"shard-{shard:05d}.int8.npy",
        "ids": output / f"shard-{shard:05d}.ids.npy",
        "metadata": output / f"shard-{shard:05d}.meta.jsonl.gz",
    }
    temporary_paths = {name: _temporary_path(path) for name, path in final_paths.items()}
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    calibrating = scales is None
    scratch_path = output / f".tmp-shard-{shard:05d}.float32.npy"
    scratch_path.unlink(missing_ok=True)
    vectors: np.memmap | None = None
    ids: np.memmap | None = None
    truth: np.memmap | None = None
    truth_ids: np.memmap | None = None
    float_scratch: np.memmap | None = None
    metadata: TextIO | None = None
    memmaps: list[np.memmap | None] = []
    calibration_maxima = np.zeros(dimension, dtype=np.float32) if calibrating else None
    clipped = 0
    values = 0
    try:
        vectors = np.lib.format.open_memmap(
            temporary_paths["vectors"],
            mode="w+",
            dtype=np.int8,
            shape=(record_count, dimension),
        )
        memmaps.append(vectors)
        ids = np.lib.format.open_memmap(
            temporary_paths["ids"], mode="w+", dtype=np.int64, shape=(record_count,)
        )
        memmaps.append(ids)
        truth = np.lib.format.open_memmap(
            temporary_paths["truth"],
            mode="w+",
            dtype=np.float32,
            shape=(truth_count, dimension),
        )
        memmaps.append(truth)
        truth_ids = np.lib.format.open_memmap(
            temporary_paths["truth_ids"],
            mode="w+",
            dtype=np.int64,
            shape=(truth_count,),
        )
        memmaps.append(truth_ids)
        metadata = gzip.open(temporary_paths["metadata"], "wt", encoding="utf-8")
        if calibrating:
            float_scratch = np.lib.format.open_memmap(
                scratch_path,
                mode="w+",
                dtype=np.float32,
                shape=(record_count, dimension),
            )
            memmaps.append(float_scratch)

        pending: Future[tuple[int, int]] | None = None
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="backfill-writer") as writer:
            for offset in range(0, record_count, embed_subchunk):
                chunk = papers[offset : offset + embed_subchunk]
                matrix = np.asarray(
                    embedder.encode_documents([paper.embedding_text for paper in chunk]),
                    dtype=np.float32,
                )
                expected_shape = (len(chunk), dimension)
                if matrix.shape != expected_shape:
                    raise RuntimeError(
                        f"Embedder returned shape {matrix.shape}; expected {expected_shape}"
                    )
                if not np.all(np.isfinite(matrix)):
                    raise RuntimeError("Embedder returned non-finite vectors")

                # Waiting here bounds the queue at one matrix. The preceding
                # write ran while the GPU produced the matrix above.
                if pending is not None:
                    chunk_clipped, chunk_values = pending.result()
                    clipped += chunk_clipped
                    values += chunk_values
                pending = writer.submit(
                    _write_embedded_chunk,
                    matrix,
                    chunk,
                    offset,
                    scales=scales,
                    vectors=vectors,
                    float_scratch=float_scratch,
                    ids=ids,
                    truth=truth,
                    truth_ids=truth_ids,
                    metadata=metadata,
                    truth_stride=truth_stride,
                    calibration_maxima=calibration_maxima,
                )
                if record_count > embed_subchunk:
                    logger.info(
                        "shard %d: embedded %s/%s",
                        shard,
                        f"{min(offset + embed_subchunk, record_count):,}",
                        f"{record_count:,}",
                    )
            if pending is not None:
                chunk_clipped, chunk_values = pending.result()
                clipped += chunk_clipped
                values += chunk_values

        if calibrating:
            if calibration_maxima is None or float_scratch is None:  # pragma: no cover
                raise RuntimeError("first-shard calibration state is incomplete")
            calibration_maxima[calibration_maxima < np.finfo(np.float32).eps] = 1.0
            scales = calibration_maxima / INT8_LEVELS
            _atomic_write(output / "int8-scales.npy", lambda p: np.save(p, scales))
            logger.info(
                "INT8 scales calibrated on all %s rows of shard %d and frozen",
                f"{record_count:,}",
                shard,
            )

            # Scale fitting needs the entire first shard, but quantization is
            # still bounded: scan the temporary Float32 memmap one subchunk at
            # a time, then delete it before committing the shard.
            for offset in range(0, record_count, embed_subchunk):
                stop = min(offset + embed_subchunk, record_count)
                matrix = np.asarray(float_scratch[offset:stop])
                clipped += int(np.count_nonzero(np.abs(matrix / scales) > INT8_LEVELS))
                values += int(matrix.size)
                vectors[offset:stop] = quantize_normalized(matrix, scales)
            del matrix

        metadata.close()
        for array in memmaps:
            _close_memmap(array)
        memmaps = []
        scratch_path.unlink(missing_ok=True)
        for name in ("truth", "truth_ids", "vectors", "ids", "metadata"):
            _publish_temporary(final_paths[name])
    finally:
        if metadata is not None and not metadata.closed:
            metadata.close()
        for array in memmaps:
            _close_memmap(array)
        scratch_path.unlink(missing_ok=True)
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    # Clip rate monitors whether the frozen first-shard calibration still
    # represents this shard's distribution (values landing outside +/-127).
    clip_rate = clipped / max(values, 1)
    rate = len(papers) / max(time.perf_counter() - started, 1e-9)
    logger.info(
        "shard %d written: %s works (%.0f works/s, clip rate %.2e, corpus position %s)",
        shard,
        f"{len(papers):,}",
        rate,
        clip_rate,
        f"{global_offset + len(papers):,}",
    )
    return scales, clip_rate


def _finalize_truth(output: Path, shard_count: int) -> None:
    """Regenerate the merged truth sample deterministically from the per-shard
    truth artifacts (which live under the crash-atomic marker protocol), so a
    redone shard can never leave duplicate rows in the global sample."""
    vectors = [np.load(output / f"shard-{i:05d}.truth.npy") for i in range(shard_count)]
    ids = [np.load(output / f"shard-{i:05d}.tids.npy") for i in range(shard_count)]
    _atomic_write(
        output / "float32-truth-sample.npy",
        lambda p: np.save(p, np.concatenate(vectors, axis=0)),
    )
    _atomic_write(
        output / "float32-truth-ids.npy", lambda p: np.save(p, np.concatenate(ids))
    )


def _put_shard_message(
    messages: Queue[object], message: object, stop: Event
) -> bool:
    """Write to the one-shard queue without making cancellation deadlock."""
    while not stop.is_set():
        try:
            messages.put(message, timeout=0.1)
            return True
        except Full:
            continue
    return False


def _acquire_shard_slot(slots: Semaphore, stop: Event) -> bool:
    """Wait for permission to build one more shard, remaining cancellable."""
    while not stop.is_set():
        if slots.acquire(timeout=0.1):
            # Cancellation can race with the consumer releasing this permit.
            # Do not advance the source once the consumer has failed/stopped.
            if stop.is_set():
                slots.release()
                return False
            return True
    return False


def _produce_backfill_shards(
    *,
    messages: Queue[object],
    slots: Semaphore,
    stop: Event,
    seen: SeenIds,
    seed_papers: Iterable[Paper],
    works_source,
    initial_position: tuple[int, int],
    records_done: int,
    shard_size: int,
    limit: int | None,
    include_expansion: bool,
) -> bool | None:
    """Parse and deduplicate source rows into strict-order prepared shards.

    The semaphore is acquired *before* reading the first row of a new shard.
    The consumer releases it as soon as it takes that shard for embedding.
    Consequently, while one shard embeds, the producer may hold at most one
    additional shard (partially built or queued).

    Returns whether the source was exhausted, or ``None`` when cancelled.
    """
    papers: list[Paper] = []
    position = initial_position
    accepted = records_done
    slot_held = False
    source_exhausted = False
    source_iterator = iter(works_source)

    def ensure_slot() -> bool:
        nonlocal slot_held
        if slot_held:
            return True
        slot_held = _acquire_shard_slot(slots, stop)
        return slot_held

    def emit_shard() -> bool:
        nonlocal papers, slot_held
        if not papers:
            return True
        if not _put_shard_message(messages, _PreparedShard(papers, position), stop):
            return False
        papers = []
        # The permit now belongs to the queued shard. Only the consumer may
        # release it, when that exact shard is taken for embedding.
        slot_held = False
        return True

    def accept(paper: Paper) -> bool:
        nonlocal accepted
        key = _numeric_id(paper)
        if key in seen:
            return True
        seen.add(key)
        seen.merge_if_due(DEDUPE_MERGE_INTERVAL)
        papers.append(paper)
        accepted += 1
        reached_limit = limit is not None and accepted >= limit
        if len(papers) >= shard_size or reached_limit:
            if not emit_shard():
                return False
        return not reached_limit

    try:
        for paper in seed_papers:
            if not ensure_slot() or not accept(paper):
                return False if stop.is_set() else source_exhausted

        while limit is None or accepted < limit:
            # Do not even request the next raw row until the sole ahead slot
            # is free. This bounds both parsed Paper memory and source advance.
            if not ensure_slot():
                return None
            try:
                file_index, line_index, raw_or_paper = next(source_iterator)
            except StopIteration:
                source_exhausted = True
                break
            position = (file_index, line_index + 1)
            if isinstance(raw_or_paper, Paper) or raw_or_paper is None:
                paper = raw_or_paper
            else:
                # Explicit test/local sources retain the historical raw-dict
                # contract. Production S3 ingestion arrives pre-parsed from
                # worker processes so gzip/JSON/abstract reconstruction is no
                # longer serialized behind this producer thread.
                paper = parse_work(raw_or_paper, include_expansion=include_expansion)
            if paper is not None and not accept(paper):
                return False if stop.is_set() else source_exhausted

        if papers and not emit_shard():
            return None
        if slot_held:
            # Exhaustion after only rejected/duplicate rows leaves no prepared
            # shard for the consumer to release.
            slots.release()
            slot_held = False
        return source_exhausted
    finally:
        close_source = getattr(source_iterator, "close", None)
        if callable(close_source):
            close_source()


def _run_shard_producer(
    *,
    messages: Queue[object],
    slots: Semaphore,
    stop: Event,
    seen: SeenIds,
    seed_papers: Iterable[Paper],
    works_source,
    initial_position: tuple[int, int],
    records_done: int,
    shard_size: int,
    limit: int | None,
    include_expansion: bool,
) -> None:
    """Publish producer completion/failure through the ordered message queue."""
    try:
        source_exhausted = _produce_backfill_shards(
            messages=messages,
            slots=slots,
            stop=stop,
            seen=seen,
            seed_papers=seed_papers,
            works_source=works_source,
            initial_position=initial_position,
            records_done=records_done,
            shard_size=shard_size,
            limit=limit,
            include_expansion=include_expansion,
        )
    except BaseException as error:
        _put_shard_message(messages, _ShardProducerFailure(error), stop)
        raise
    else:
        if source_exhausted is not None:
            _put_shard_message(
                messages,
                _ShardProducerFinished(source_exhausted),
                stop,
            )


def run_backfill(
    output: Path,
    embedder: Embedder,
    *,
    shard_size: int = 1_000_000,
    limit: int | None = None,
    truth_stride: int = TRUTH_STRIDE_DEFAULT,
    include_expansion: bool = False,
    works_source=None,
    seed_works: Iterable[dict] | None = None,
    snapshot_meta: dict | None = None,
    snapshot_probe: Callable[[], dict] | None = None,
    clip_fail_rate: float = 0.01,
    embed_subchunk: int = EMBED_SUBCHUNK_DEFAULT,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if truth_stride <= 0:
        raise ValueError("truth_stride must be positive")
    if embed_subchunk <= 0:
        raise ValueError("embed_subchunk must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if seed_works is not None and limit is None:
        raise ValueError(
            "A benchmark seed overlay requires a finite --limit; an exhaustive "
            "full-corpus backfill must contain only pinned snapshot records"
        )

    _pin_backfill_config(
        output,
        _backfill_config(
            embedder,
            truth_stride=truth_stride,
            include_expansion=include_expansion,
        ),
    )

    seed_values = list(seed_works or ())
    seed_payload = json.dumps(
        seed_values, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    seed_papers = [paper for value in seed_values if (paper := parse_work(value)) is not None]
    seed_pin = {
        "sha256": hashlib.sha256(seed_payload).hexdigest(),
        "openalex_ids": [paper.openalex_id for paper in seed_papers],
    }
    seed_pin_path = output / "benchmark-seeds-pin.json"
    if seed_pin_path.exists():
        pinned_seeds = json.loads(seed_pin_path.read_text(encoding="utf-8"))
        if not seed_values or pinned_seeds != seed_pin:
            raise RuntimeError(
                "Benchmark seed overlay changed since this backfill started; "
                "resume with the same seeds or use a fresh artifact directory"
            )
    elif seed_values:
        _atomic_write(
            seed_pin_path,
            lambda p: p.write_text(
                json.dumps(seed_pin, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            ),
        )

    if works_source is None and snapshot_meta is None:
        raise RuntimeError(
            "A live S3 backfill requires snapshot metadata so it cannot mix releases"
        )

    # Pin the snapshot identity on first run; a resume against a different
    # OpenAlex release must refuse rather than silently mix two corpora.
    pin_path = output / "snapshot-pin.json"
    pinned_snapshot = snapshot_meta
    if snapshot_meta is not None:
        if pin_path.exists():
            pinned = json.loads(pin_path.read_text(encoding="utf-8"))
            if pinned.get("manifest_etag") != snapshot_meta.get("manifest_etag"):
                raise RuntimeError(
                    "Snapshot changed since this backfill started "
                    f"(pinned etag {pinned.get('manifest_etag')}, current "
                    f"{snapshot_meta.get('manifest_etag')}). Finish against the pinned "
                    "release or start a fresh artifact directory."
                )
            pinned_snapshot = pinned
        else:
            _atomic_write(
                pin_path,
                lambda p: p.write_text(
                    json.dumps(snapshot_meta, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                ),
            )

    def assert_snapshot_unchanged() -> None:
        if snapshot_probe is None or pinned_snapshot is None:
            return
        current = snapshot_probe()
        if current.get("manifest_etag") != pinned_snapshot.get("manifest_etag"):
            raise RuntimeError(
                "OpenAlex snapshot changed while the backfill was running "
                f"(pinned etag {pinned_snapshot.get('manifest_etag')}, current "
                f"{current.get('manifest_etag')}). The in-flight shard was not committed."
            )

    state, seen = _resume_state(output)
    scales = (
        np.load(output / "int8-scales.npy") if (output / "int8-scales.npy").exists() else None
    )
    if state["records_done"]:
        logger.info(
            "resuming: %s works already embedded across %d shards",
            f"{state['records_done']:,}",
            state["next_shard"],
        )

    def commit_shard(papers: list[Paper], position: tuple[int, int]) -> None:
        nonlocal scales, done
        shard = state["next_shard"]
        # Check on both sides of expensive inference. If a quarterly release
        # lands during the shard, its files remain uncommitted and are redone
        # in a fresh artifact directory.
        assert_snapshot_unchanged()
        scales, clip_rate = _flush_shard(
            output,
            shard,
            papers,
            embedder,
            scales,
            truth_stride,
            done,
            embed_subchunk,
        )
        assert_snapshot_unchanged()
        if clip_rate > clip_fail_rate:
            # Raised BEFORE the marker: the shard stays uncommitted and is
            # redone after recalibration rather than shipping degraded vectors.
            raise RuntimeError(
                f"Shard {shard} clip rate {clip_rate:.4f} exceeds the "
                f"{clip_fail_rate:.4f} threshold: the frozen INT8 calibration no "
                "longer represents the corpus. Recalibrate before continuing."
            )
        done += len(papers)
        # The marker is written LAST: a crash before this line leaves orphan
        # shard files that resume deletes and redoes — never mixed shards.
        _atomic_write(
            output / f"shard-{shard:05d}.done",
            lambda p: p.write_text(
                json.dumps(
                    {
                        "records": len(papers),
                        "shard_size": shard_size,
                        "embed_subchunk": embed_subchunk,
                        "file_index": position[0],
                        "line_index": position[1],
                        "records_done": done,
                        "clip_rate": clip_rate,
                        "snapshot_etag": (
                            pinned_snapshot.get("manifest_etag")
                            if pinned_snapshot is not None
                            else None
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            ),
        )
        state.update(next_shard=shard + 1, file_index=position[0], line_index=position[1])

    done = state["records_done"]
    source_exhausted = False
    limit_reached = limit is not None and done >= limit

    if not limit_reached:
        if works_source is None:
            works_source = iter_s3_manifest_papers(
                skip_files=state["file_index"],
                skip_lines=state["line_index"],
                include_expansion=include_expansion,
            )
        messages: Queue[object] = Queue(maxsize=_SHARD_PREFETCH_DEPTH)
        slots = Semaphore(_SHARD_PREFETCH_DEPTH)
        producer_stop = Event()
        producer_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="backfill-ingest",
        )
        producer_future = producer_executor.submit(
            _run_shard_producer,
            messages=messages,
            slots=slots,
            stop=producer_stop,
            seen=seen,
            seed_papers=seed_papers,
            works_source=works_source,
            initial_position=(state["file_index"], state["line_index"]),
            records_done=done,
            shard_size=shard_size,
            limit=limit,
            include_expansion=include_expansion,
        )
        try:
            while True:
                try:
                    message = messages.get(timeout=0.1)
                except Empty:
                    if producer_future.done():
                        producer_future.result()
                        raise RuntimeError(
                            "backfill ingestion stopped without a completion message"
                        )
                    continue
                if isinstance(message, _PreparedShard):
                    # The producer may now build exactly one next shard while
                    # this shard embeds and writes.
                    slots.release()
                    commit_shard(message.papers, message.position)
                    continue
                if isinstance(message, _ShardProducerFinished):
                    source_exhausted = message.source_exhausted
                    break
                if isinstance(message, _ShardProducerFailure):
                    producer_future.result()
                    raise message.error  # pragma: no cover - defensive
                raise RuntimeError("backfill ingestion returned an invalid message")
            producer_future.result()
        finally:
            producer_stop.set()
            # Release an acquire blocked before the next source row. Queue puts
            # are separately stop-aware.
            slots.release()
            producer_executor.shutdown(wait=True, cancel_futures=True)

    # No further dedupe lookups occur after ingestion. Resume reconstructs the
    # set from committed shard ID artifacts, so release the potentially large
    # in-memory structures instead of copying the final sub-50M remainder.
    seen.pending.clear()
    seen.merged = np.empty(0, dtype=np.int64)

    _finalize_truth(output, state["next_shard"])
    marker_states = [
        json.loads(marker.read_text(encoding="utf-8"))
        for marker in sorted(output.glob("shard-*.done"))
    ]
    max_clip = max((marker.get("clip_rate", 0.0) for marker in marker_states), default=0.0)
    shard_config_history: list[dict] = []
    legacy_marker_defaults_applied = 0
    for shard, marker in enumerate(marker_states):
        if "shard_size" not in marker or "embed_subchunk" not in marker:
            # Markers emitted before format v2 did not record these two
            # performance-only settings. A manually verified config pin may
            # adopt such an artifact set without rewriting already-synced
            # commit markers. The historical code used a fixed 65,536-row
            # embedding subchunk; a full marker's record count is the best
            # available shard-size evidence.
            legacy_marker_defaults_applied += 1
        shard_config = {
            "shard_size": int(marker.get("shard_size", marker["records"])),
            "embed_subchunk": int(marker.get("embed_subchunk", EMBED_SUBCHUNK_DEFAULT)),
        }
        if shard_config_history and all(
            shard_config_history[-1][key] == value for key, value in shard_config.items()
        ):
            shard_config_history[-1]["last_shard"] = shard
        else:
            shard_config_history.append(
                {
                    "first_shard": shard,
                    "last_shard": shard,
                    **shard_config,
                }
            )

    logger.info("computing artifact checksums")
    checksums = {}
    for artifact in sorted(output.iterdir()):
        if artifact.name.startswith(".tmp-") or artifact.name == "backfill-manifest.json":
            continue
        digest = hashlib.sha256()
        with open(artifact, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        checksums[artifact.name] = digest.hexdigest()

    manifest = {
        "format_version": 2,
        "kind": "backfill-artifacts",
        "records": done,
        "shards": state["next_shard"],
        "shard_size": shard_size,
        "embed_subchunk": embed_subchunk,
        "shard_config_history": shard_config_history,
        "legacy_marker_defaults_applied": legacy_marker_defaults_applied,
        "embedder": embedder.name,
        "embedder_config": {
            "batch_size": getattr(embedder, "batch_size", None),
            "dtype": getattr(embedder, "dtype", "float32"),
            "max_seq_length": int(getattr(getattr(embedder, "model", None), "max_seq_length", 0))
            or None,
        },
        "snapshot_prefix": "s3://openalex/data/jsonl/works/",
        "snapshot_meta": pinned_snapshot,
        "snapshot_iteration_order": "updated_date_desc",
        "code_provenance": _code_provenance(),
        "include_expansion": include_expansion,
        "source_exhausted": source_exhausted,
        "requested_limit": limit,
        "benchmark_seed_overlay": seed_pin if seed_values else None,
        "truth_stride": truth_stride,
        "max_clip_rate": max_clip,
        "clip_fail_rate": clip_fail_rate,
        "created_unix": int(time.time()),
        "checksums": checksums,
    }
    _atomic_write(
        output / "backfill-manifest.json",
        lambda p: p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n"),
    )
    logger.info("backfill complete: %s works in %d shards", f"{done:,}", state["next_shard"])
    return manifest
