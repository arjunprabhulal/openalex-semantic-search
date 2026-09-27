"""Assemble a servable generation from backfill shard artifacts, streaming.

Unlike build_generation, nothing corpus-sized is held in memory: INT8 shards
are concatenated file-to-file, the FAISS index is trained on a bounded
dequantized sample and fed shard by shard, and metadata streams into SQLite in
batches with one FTS build at the end. Ground truth ships as the backfill's
strided float32 sample (float32-truth-sample.npy + float32-truth-ids.npy);
the benchmark falls back to an exact dequantized-INT8 scan when the full
float32 truth matrix is absent.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import gzip
import logging
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .config import Stage, VECTOR_DIMENSION
from .compact_store import (
    DEFAULT_BLOCK_ROWS,
    build_compact_metadata,
    validated_compact_manifest,
)
from .records import Paper
from .store import MetadataStore
from .vector_index import FaissPQIndex, NumpyFlatIndex

logger = logging.getLogger(__name__)

INSERT_BATCH = 5_000
FAISS_ADD_BATCH = 100_000
GENERATION_CHECKSUMS = "generation-files.sha256"
ASSEMBLY_PIN = "assembly-config-pin.json"
FAISS_TRAINED = "ivfpq-trained.faiss"
FAISS_PARTS = ".faiss-parts"
FAISS_PART_VERSION = 1


def _shard_paths(artifacts: Path) -> list[Path]:
    shards = sorted(artifacts.glob("shard-*.int8.npy"))
    if not shards:
        raise ValueError(f"No shard-*.int8.npy artifacts found in {artifacts}")
    return shards


def _verify_checksums(artifacts: Path) -> None:
    manifest_path = artifacts / "backfill-manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"{manifest_path} is missing; refuse to assemble unverified artifacts")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest.get("checksums", {}).items():
        digest = hashlib.sha256()
        with open(artifacts / name, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch for {name}; transfer is corrupt")
    logger.info("all %d artifact checksums verified", len(manifest.get("checksums", {})))


def _dequant_sample(shards: list[Path], scales: np.ndarray, target: int) -> np.ndarray:
    """Gather an evenly-strided dequantized training sample across shards."""
    collected: list[np.ndarray] = []
    total = 0
    for shard in shards:
        codes = np.load(shard, mmap_mode="r")
        total += len(codes)
    stride = max(1, total // max(target, 1))
    for shard in shards:
        codes = np.load(shard, mmap_mode="r")
        picks = codes[::stride]
        collected.append(np.asarray(picks, dtype=np.float32) * scales)
    sample = np.concatenate(collected, axis=0)
    logger.info("training sample: %s vectors (stride %d)", f"{len(sample):,}", stride)
    return np.ascontiguousarray(sample[:target])


def _write_truth_row_ids(
    artifacts: Path,
    directory: Path,
    shards: list[Path],
) -> None:
    truth_ids_path = artifacts / "float32-truth-ids.npy"
    if not truth_ids_path.exists():
        return
    truth_ids = np.asarray(np.load(truth_ids_path), dtype=np.int64)
    if len(np.unique(truth_ids)) != len(truth_ids):
        raise ValueError("Float32 truth ids contain duplicates")
    order = np.argsort(truth_ids)
    sorted_truth_ids = truth_ids[order]
    missing = np.iinfo(np.uint32).max
    truth_rows = np.full(len(truth_ids), missing, dtype=np.uint32)
    row_start = 0
    for shard in shards:
        ids_path = shard.with_name(shard.name.replace(".int8.npy", ".ids.npy"))
        ids = np.asarray(np.load(ids_path, mmap_mode="r"), dtype=np.int64)
        positions = np.searchsorted(sorted_truth_ids, ids)
        valid = positions < len(sorted_truth_ids)
        matched = np.zeros(len(ids), dtype=bool)
        matched[valid] = sorted_truth_ids[positions[valid]] == ids[valid]
        if np.any(matched):
            original_truth_positions = order[positions[matched]]
            truth_rows[original_truth_positions] = (
                row_start + np.flatnonzero(matched)
            ).astype(np.uint32)
        row_start += len(ids)
    if np.any(truth_rows == missing):
        raise ValueError("Some Float32 truth ids are absent from the assembled shards")
    np.save(directory / "float32-truth-row-ids.npy", truth_rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_generation_checksums(directory: Path) -> None:
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != GENERATION_CHECKSUMS
    )
    lines = [f"{_sha256(path)}  {path.relative_to(directory)}\n" for path in files]
    (directory / GENERATION_CHECKSUMS).write_text("".join(lines), encoding="utf-8")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _effective_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _new_faiss_index(faiss_module, stage: Stage, list_count: int):
    if list_count >= 4_096:
        quantizer = faiss_module.IndexHNSWFlat(
            VECTOR_DIMENSION,
            32,
            faiss_module.METRIC_INNER_PRODUCT,
        )
        quantizer.hnsw.efConstruction = 80
        quantizer.hnsw.efSearch = 128
    else:
        quantizer = faiss_module.IndexFlatIP(VECTOR_DIMENSION)
    return faiss_module.IndexIVFPQ(
        quantizer,
        VECTOR_DIMENSION,
        list_count,
        stage.pq_subquantizers,
        stage.pq_bits,
        faiss_module.METRIC_INNER_PRODUCT,
    )


def _valid_file_marker(marker_path: Path, file_path: Path, expected: dict) -> bool:
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if any(marker.get(key) != value for key, value in expected.items()):
        return False
    return file_path.is_file() and marker.get("sha256") == _sha256(file_path)


def _build_faiss_shard_part(
    trained_path: str,
    shard_path: str,
    scales_path: str,
    start_row: int,
    output_path: str,
    threads: int,
) -> dict:
    from .vector_index import FaissPQIndex as _FaissPQIndex

    faiss_module = _FaissPQIndex._faiss()
    faiss_module.omp_set_num_threads(max(1, threads))
    index = faiss_module.read_index(trained_path)
    codes = np.load(shard_path, mmap_mode="r")
    scales = np.load(scales_path).astype(np.float32)
    if codes.ndim != 2 or codes.shape[1] != VECTOR_DIMENSION:
        raise ValueError(f"{shard_path} has invalid shape {codes.shape}")
    row = start_row
    for offset in range(0, len(codes), FAISS_ADD_BATCH):
        block = np.asarray(
            codes[offset : offset + FAISS_ADD_BATCH],
            dtype=np.float32,
        ) * scales
        ids = np.arange(row, row + len(block), dtype=np.int64)
        index.add_with_ids(np.ascontiguousarray(block), ids)
        row += len(block)
    output = Path(output_path)
    temporary = output.with_name(f".{output.name}.tmp")
    faiss_module.write_index(index, str(temporary))
    temporary.replace(output)
    result = {
        "format_version": FAISS_PART_VERSION,
        "start_row": start_row,
        "records": len(codes),
        "sha256": _sha256(output),
    }
    _atomic_json(output.with_suffix(".done"), result)
    return result


def _build_resumable_faiss(
    shards: list[Path],
    scales: np.ndarray,
    directory: Path,
    *,
    stage: Stage,
    shard_rows: list[int],
    total: int,
) -> tuple[int, float]:
    started = time.perf_counter()
    faiss_module = FaissPQIndex._faiss()
    effective_cpus = _effective_cpus()
    faiss_module.omp_set_num_threads(max(1, min(32, effective_cpus)))
    list_count = min(stage.ivf_lists, max(1, total // 40))
    header = directory / FaissPQIndex.FILENAME
    lists = directory / FaissPQIndex.IVF_DATA_FILENAME
    merged_marker = directory / f"{FaissPQIndex.FILENAME}.done"
    try:
        marker = json.loads(merged_marker.read_text(encoding="utf-8"))
        merged_valid = (
            int(marker.get("records", -1)) == total
            and header.is_file()
            and lists.is_file()
            and marker.get("header_sha256") == _sha256(header)
            and marker.get("lists_sha256") == _sha256(lists)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        merged_valid = False
    if merged_valid:
        loaded = FaissPQIndex.load(directory, stage.nprobe)
        if int(loaded.index.ntotal) != total:
            raise ValueError("Checkpointed candidate index row count is invalid")
        # The merged on-disk lists are self-contained. Clean any staging files
        # left by a crash after the marker-last commit.
        shutil.rmtree(directory / FAISS_PARTS, ignore_errors=True)
        (directory / FAISS_TRAINED).unlink(missing_ok=True)
        (directory / f"{FAISS_TRAINED}.done").unlink(missing_ok=True)
        logger.info("reusing checksum-verified merged FAISS index")
        return header.stat().st_size + lists.stat().st_size, time.perf_counter() - started

    trained_path = directory / FAISS_TRAINED
    trained_marker = directory / f"{FAISS_TRAINED}.done"
    trained_expected = {
        "format_version": FAISS_PART_VERSION,
        "records": 0,
        "list_count": list_count,
    }
    if _valid_file_marker(trained_marker, trained_path, trained_expected):
        logger.info("reusing checksum-verified trained FAISS index")
    else:
        trained_path.unlink(missing_ok=True)
        trained_marker.unlink(missing_ok=True)
        index = _new_faiss_index(faiss_module, stage, list_count)
        train_target = min(total, max(50_000, list_count * 40))
        index.train(_dequant_sample(shards, scales, train_target))
        temporary = trained_path.with_name(f".{trained_path.name}.tmp")
        faiss_module.write_index(index, str(temporary))
        temporary.replace(trained_path)
        _atomic_json(
            trained_marker,
            {**trained_expected, "sha256": _sha256(trained_path)},
        )
        logger.info("trained FAISS index checkpoint committed")

    parts = directory / FAISS_PARTS
    parts.mkdir(exist_ok=True)
    starts: list[int] = []
    row = 0
    for count in shard_rows:
        starts.append(row)
        row += count
    missing: list[int] = []
    for index, count in enumerate(shard_rows):
        part = parts / f"shard-{index:05d}.faiss"
        marker = part.with_suffix(".done")
        expected = {
            "format_version": FAISS_PART_VERSION,
            "start_row": starts[index],
            "records": count,
        }
        if _valid_file_marker(marker, part, expected):
            logger.info("candidate part %d/%d reused from checkpoint", index + 1, len(shards))
        else:
            part.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            missing.append(index)

    worker_count = min(2, max(1, effective_cpus // 8), max(1, len(missing)))
    worker_threads = max(1, effective_cpus // worker_count)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as pool:
        futures = {
            pool.submit(
                _build_faiss_shard_part,
                str(trained_path),
                str(shards[index]),
                str(directory / "int8-scales.npy"),
                starts[index],
                str(parts / f"shard-{index:05d}.faiss"),
                worker_threads,
            ): index
            for index in missing
        }
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            logger.info(
                "candidate part %d/%d committed (%s records)",
                index + 1,
                len(shards),
                f"{result['records']:,}",
            )

    part_paths = [parts / f"shard-{index:05d}.faiss" for index in range(len(shards))]
    from faiss.contrib.ondisk import merge_ondisk

    header.unlink(missing_ok=True)
    lists.unlink(missing_ok=True)
    merged_marker.unlink(missing_ok=True)
    trained = faiss_module.read_index(str(trained_path))
    merge_ondisk(trained, [str(path) for path in part_paths], str(lists))
    trained.nprobe = min(stage.nprobe, list_count)
    header_temporary = header.with_name(f".{header.name}.tmp")
    faiss_module.write_index(trained, str(header_temporary))
    header_temporary.replace(header)
    _atomic_json(
        merged_marker,
        {
            "format_version": FAISS_PART_VERSION,
            "records": total,
            "header_sha256": _sha256(header),
            "lists_sha256": _sha256(lists),
        },
    )
    logger.info("merged on-disk FAISS index checkpoint committed")
    loaded = FaissPQIndex.load(directory, stage.nprobe)
    if int(loaded.index.ntotal) != total:
        raise ValueError(
            f"Merged candidate index has {int(loaded.index.ntotal):,} rows; expected {total:,}"
        )
    bytes_used = header.stat().st_size + lists.stat().st_size
    shutil.rmtree(parts)
    trained_path.unlink(missing_ok=True)
    trained_marker.unlink(missing_ok=True)
    return bytes_used, time.perf_counter() - started


def assemble_generation(
    artifacts: Path,
    *,
    stage: Stage,
    output: Path,
    backend: str = "faiss",
    metadata_backend: str = "sqlite",
    metadata_workers: int = 8,
    metadata_block_rows: int = DEFAULT_BLOCK_ROWS,
    resume: bool = False,
    embedder_name: str,
    allow_partial: bool = False,
) -> dict:
    started = time.perf_counter()
    _verify_checksums(artifacts)
    shards = _shard_paths(artifacts)
    scales = np.load(artifacts / "int8-scales.npy").astype(np.float32)

    # Completeness gate: the artifact set must be exactly what the backfill
    # manifest promised — a partial transfer must never become a generation.
    backfill_manifest = json.loads(
        (artifacts / "backfill-manifest.json").read_text(encoding="utf-8")
    )
    if len(shards) != backfill_manifest["shards"]:
        raise ValueError(
            f"Artifact set has {len(shards)} shards; manifest promises "
            f"{backfill_manifest['shards']} — refusing to assemble a partial set"
        )
    shard_rows = [
        len(np.load(shard.with_name(shard.name.replace(".int8.npy", ".ids.npy")), mmap_mode="r"))
        for shard in shards
    ]
    id_total = sum(shard_rows)
    if id_total != backfill_manifest["records"]:
        raise ValueError(
            f"Artifacts contain {id_total:,} records; manifest promises "
            f"{backfill_manifest['records']:,} — refusing to assemble a partial set"
        )
    source_exhausted = bool(backfill_manifest.get("source_exhausted", False))
    if stage.name == "full" and not source_exhausted:
        raise ValueError(
            "The full stage requires an unbounded backfill that exhausted the pinned "
            "snapshot; a limited or interrupted run cannot publish as full"
        )
    deviation = abs(id_total - stage.records) / stage.records
    if stage.name != "full" and id_total != stage.records and not allow_partial:
        raise ValueError(
            f"Artifacts hold {id_total:,} records but stage {stage.name} requires "
            f"exactly {stage.records:,} ({deviation:.0%} off). A truncated backfill must "
            "not publish under this stage name; pass allow_partial=True "
            "(--allow-partial-stage) only if this is deliberate."
        )
    if stage.name != "full" and id_total != stage.records:
        logger.warning(
            "assembling %s records under stage %s (nominal %s) by explicit override",
            f"{id_total:,}", stage.name, f"{stage.records:,}",
        )

    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing generation {output}")
    temporary = parent / f".{output.name}.building"
    if temporary.exists():
        if not resume:
            shutil.rmtree(temporary)
    temporary.mkdir(exist_ok=True)
    assembly_pin = {
        "format_version": 1,
        "artifacts_manifest_sha256": _sha256(artifacts / "backfill-manifest.json"),
        "records": id_total,
        "stage": {
            "name": stage.name,
            "ivf_lists": stage.ivf_lists,
            "nprobe": stage.nprobe,
            "candidate_count": stage.candidate_count,
            "pq_subquantizers": stage.pq_subquantizers,
            "pq_bits": stage.pq_bits,
        },
        "backend": backend,
        "metadata_backend": metadata_backend,
        "metadata_block_rows": metadata_block_rows,
    }
    assembly_pin_path = temporary / ASSEMBLY_PIN
    if assembly_pin_path.exists():
        existing_pin = json.loads(assembly_pin_path.read_text(encoding="utf-8"))
        if existing_pin != assembly_pin:
            raise RuntimeError("Assembly resume configuration does not match its frozen pin")
    else:
        _atomic_json(assembly_pin_path, assembly_pin)
    try:
        # INT8 refinement layer: concatenate shards file-to-file.
        total = id_total
        int8_path = temporary / "int8-vectors.bin"
        int8_marker = temporary / "int8-layer.done"
        expected_int8_bytes = total * VECTOR_DIMENSION
        valid_int8 = False
        try:
            marker = json.loads(int8_marker.read_text(encoding="utf-8"))
            valid_int8 = (
                int(marker.get("records", -1)) == total
                and int(marker.get("bytes", -1)) == expected_int8_bytes
                and int8_path.stat().st_size == expected_int8_bytes
                and marker.get("sha256") == _sha256(int8_path)
                and marker.get("scales_sha256") == _sha256(artifacts / "int8-scales.npy")
                and marker.get("output_scales_sha256")
                == _sha256(temporary / "int8-scales.npy")
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            valid_int8 = False
        if valid_int8:
            logger.info("reusing checkpointed INT8 refinement layer")
        else:
            int8_marker.unlink(missing_ok=True)
            digest = hashlib.sha256()
            with open(int8_path, "wb") as out:
                written = 0
                for shard in shards:
                    codes = np.load(shard, mmap_mode="r")
                    if codes.shape[1] != VECTOR_DIMENSION:
                        raise ValueError(f"{shard} has dimension {codes.shape[1]}")
                    for offset in range(0, len(codes), FAISS_ADD_BATCH):
                        block = np.ascontiguousarray(
                            codes[offset : offset + FAISS_ADD_BATCH]
                        ).tobytes()
                        out.write(block)
                        digest.update(block)
                    written += len(codes)
                if written != total:
                    raise ValueError(f"INT8 rows ({written}) do not match ids ({total})")
            np.save(temporary / "int8-scales.npy", scales)
            _atomic_json(
                int8_marker,
                {
                    "records": total,
                    "bytes": expected_int8_bytes,
                    "sha256": digest.hexdigest(),
                    "scales_sha256": _sha256(artifacts / "int8-scales.npy"),
                    "output_scales_sha256": _sha256(temporary / "int8-scales.npy"),
                },
            )
            logger.info("INT8 layer checkpoint committed: %s vectors", f"{total:,}")

        # Candidate index: train once on a bounded sample, add shard by shard.
        index_started = time.perf_counter()
        if backend == "faiss":
            if resume:
                candidate_index_bytes, index_seconds = _build_resumable_faiss(
                    shards,
                    scales,
                    temporary,
                    stage=stage,
                    shard_rows=shard_rows,
                    total=total,
                )
            else:
                faiss_module = FaissPQIndex._faiss()
                faiss_threads = max(1, min(32, _effective_cpus()))
                faiss_module.omp_set_num_threads(faiss_threads)
                logger.info("FAISS using %d effective CPU threads", faiss_threads)
                list_count = min(stage.ivf_lists, max(1, total // 40))
                index = _new_faiss_index(faiss_module, stage, list_count)
                train_target = min(total, max(50_000, list_count * 40))
                index.train(_dequant_sample(shards, scales, train_target))
                row = 0
                for shard in shards:
                    codes = np.load(shard, mmap_mode="r")
                    for offset in range(0, len(codes), FAISS_ADD_BATCH):
                        block = np.asarray(
                            codes[offset : offset + FAISS_ADD_BATCH], dtype=np.float32
                        ) * scales
                        ids = np.arange(row, row + len(block), dtype=np.int64)
                        index.add_with_ids(np.ascontiguousarray(block), ids)
                        row += len(block)
                    logger.info(
                        "candidate index: %s/%s vectors added", f"{row:,}", f"{total:,}"
                    )
                index.nprobe = min(stage.nprobe, list_count)
                faiss_module.write_index(index, str(temporary / FaissPQIndex.FILENAME))
                candidate_index_bytes = (temporary / FaissPQIndex.FILENAME).stat().st_size
                index_seconds = time.perf_counter() - index_started
        elif backend == "numpy":
            blocks = [
                np.asarray(np.load(shard, mmap_mode="r"), dtype=np.float32) * scales
                for shard in shards
            ]
            NumpyFlatIndex.build(np.concatenate(blocks, axis=0), temporary)
            candidate_index_bytes = (temporary / NumpyFlatIndex.FILENAME).stat().st_size
            index_seconds = time.perf_counter() - index_started
        else:
            raise ValueError("backend must be 'faiss' or 'numpy'")

        # Metadata: SQLite remains available for small benchmark stages. The
        # full corpus uses compressed random-access blocks plus fixed-width
        # ranking/filter sidecars so rich metadata is fetched only for a page.
        metadata_started = time.perf_counter()
        compact_manifest = None
        if metadata_backend == "compact":
            compact_manifest = (
                validated_compact_manifest(temporary, expected_records=total)
                if resume
                else None
            )
            if compact_manifest is None:
                compact_manifest = build_compact_metadata(
                    shards,
                    temporary,
                    shard_rows=shard_rows,
                    total=total,
                    workers=metadata_workers,
                    block_rows=metadata_block_rows,
                    resume=resume,
                )
            else:
                logger.info("reusing checkpointed compact metadata")
                shutil.rmtree(temporary / ".compact-parts", ignore_errors=True)
            metadata_bytes = int(compact_manifest["bytes"])
        elif metadata_backend == "sqlite":
            store = MetadataStore(temporary / "metadata.sqlite3")
            try:
                store.create_schema()
                row = 0
                for shard in shards:
                    meta_path = shard.with_name(
                        shard.name.replace(".int8.npy", ".meta.jsonl.gz")
                    )
                    batch: list[Paper] = []
                    with gzip.open(meta_path, "rt", encoding="utf-8") as lines:
                        for line in lines:
                            record = json.loads(line)
                            record["authors"] = tuple(record.get("authors") or ())
                            batch.append(Paper(embedding_text="", **record))
                            if len(batch) == INSERT_BATCH:
                                store.insert_paper_batch(batch, row)
                                row += len(batch)
                                batch = []
                    if batch:
                        store.insert_paper_batch(batch, row)
                        row += len(batch)
                    logger.info(
                        "metadata: %s/%s rows inserted", f"{row:,}", f"{total:,}"
                    )
                if row != total:
                    raise ValueError(f"metadata rows ({row}) do not match vectors ({total})")
                logger.info("building title FTS")
                store.build_fts()
            finally:
                store.close()
            metadata_bytes = sum(
                item.stat().st_size
                for item in temporary.glob("metadata.sqlite3*")
                if item.is_file()
            )
        else:
            raise ValueError("metadata_backend must be 'sqlite' or 'compact'")
        metadata_seconds = time.perf_counter() - metadata_started

        # Sampled ground truth travels with the generation.
        for name in ("float32-truth-sample.npy", "float32-truth-ids.npy"):
            if (artifacts / name).exists():
                shutil.copy2(artifacts / name, temporary / name)
        _write_truth_row_ids(artifacts, temporary, shards)

        int8_bytes = int8_path.stat().st_size + (temporary / "int8-scales.npy").stat().st_size
        production_bytes = int8_bytes + metadata_bytes + candidate_index_bytes
        manifest = {
            "format_version": 1,
            "stage": stage.name,
            "records": total,
            "dimension": VECTOR_DIMENSION,
            "embedder": embedder_name,
            "backend": backend,
            "metadata_backend": metadata_backend,
            "compact_metadata": compact_manifest,
            "resumable_assembly": resume,
            "candidate_count": min(stage.candidate_count, total),
            "nprobe": stage.nprobe,
            "assembled_from": str(artifacts),
            "backfill_created_unix": backfill_manifest.get("created_unix"),
            "backfill_provenance": {
                k: backfill_manifest.get(k)
                for k in (
                    "embedder_config",
                    "code_provenance",
                    "benchmark_seed_overlay",
                    "snapshot_meta",
                    "snapshot_iteration_order",
                    "source_exhausted",
                    "requested_limit",
                    "max_clip_rate",
                    "clip_fail_rate",
                    "shard_size",
                    "embed_subchunk",
                )
            },
            "stage_records_expected": stage.records,
            "truth": "sampled",
            "build_seconds": time.perf_counter() - started,
            "embed_seconds": 0.0,
            "embed_records_per_second": 0.0,
            "index_seconds": index_seconds,
            "metadata_seconds": metadata_seconds,
            "component_bytes": {
                "benchmark_float32_truth": (temporary / "float32-truth-sample.npy").stat().st_size
                if (temporary / "float32-truth-sample.npy").exists()
                else 0,
                "candidate_index": candidate_index_bytes,
                "int8_refinement": int8_bytes,
                "metadata_and_title_fts": metadata_bytes,
            },
            "measured_production_bytes": production_bytes,
            "measured_production_bytes_per_record": production_bytes / total,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Stabilize manifest size before the one expensive full-generation hash
        # pass. The checksum file excludes itself and covers the final manifest.
        for _ in range(4):
            existing = [path for path in temporary.rglob("*") if path.is_file()]
            checksum_bytes = sum(
                64 + 2 + len(str(path.relative_to(temporary)).encode("utf-8")) + 1
                for path in existing
                if path.name != GENERATION_CHECKSUMS
            )
            generation_bytes = sum(path.stat().st_size for path in existing) + checksum_bytes
            manifest["generation_bytes"] = generation_bytes
            manifest["benchmark_bytes_per_record"] = generation_bytes / total
            serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            current = (temporary / "manifest.json").read_text(encoding="utf-8")
            if current == serialized:
                break
            (temporary / "manifest.json").write_text(serialized, encoding="utf-8")
        _write_generation_checksums(temporary)
        generation_bytes = sum(
            item.stat().st_size for item in temporary.rglob("*") if item.is_file()
        )
        temporary.replace(output)
        logger.info(
            "generation assembled at %s (%.1fs, %.2f GB)",
            output,
            manifest["build_seconds"],
            generation_bytes / 1e9,
        )
        return manifest
    except Exception:
        if not resume:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
