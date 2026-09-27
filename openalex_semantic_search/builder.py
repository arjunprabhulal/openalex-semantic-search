from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import time
from typing import Iterable

import numpy as np

from .config import Stage, VECTOR_DIMENSION
from .embeddings import Embedder
from .quantization import fit_int8_scales, quantize_normalized
from .records import Paper
from .store import MetadataStore
from .vector_index import FaissPQIndex, NumpyFlatIndex

logger = logging.getLogger(__name__)


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build_generation(
    papers: Iterable[Paper],
    *,
    stage: Stage,
    output: Path,
    embedder: Embedder,
    backend: str = "faiss",
    batch_size: int = 128,
) -> dict:
    """Build a new immutable benchmark generation and publish it atomically."""
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{output.name}.building"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()

    selected: list[Paper] = []
    vectors: list[np.ndarray] = []
    started = time.perf_counter()
    try:
        logger.info("collecting %s usable works for stage %s", f"{stage.records:,}", stage.name)
        for paper in papers:
            selected.append(paper)
            if len(selected) % 2_500 == 0:
                logger.info("collected %s/%s works", f"{len(selected):,}", f"{stage.records:,}")
            if len(selected) == stage.records:
                break
        if len(selected) != stage.records:
            raise ValueError(
                f"Stage {stage.name} requires exactly {stage.records:,} usable works; "
                f"the source produced {len(selected):,}"
            )

        logger.info("embedding %s works with %s", f"{len(selected):,}", embedder.name)
        embed_started = time.perf_counter()
        # Feed the embedder large chunks: sentence-transformers length-sorts
        # within each call, so big chunks minimize padding waste, and a
        # multi-worker embedder shards each chunk across processes.
        chunk_size = max(batch_size, 8_192)
        for offset in range(0, len(selected), chunk_size):
            batch = selected[offset : offset + chunk_size]
            encoded = embedder.encode_documents([paper.embedding_text for paper in batch])
            if encoded.shape != (len(batch), VECTOR_DIMENSION):
                expected = (len(batch), VECTOR_DIMENSION)
                raise RuntimeError(f"Embedder returned {encoded.shape}; expected {expected}")
            vectors.append(np.asarray(encoded, dtype=np.float32))
            done = offset + len(batch)
            rate = done / max(time.perf_counter() - embed_started, 1e-9)
            logger.info(
                "embedded %s/%s (%.0f works/s)", f"{done:,}", f"{len(selected):,}", rate
            )
        matrix = np.concatenate(vectors, axis=0)
        embed_seconds = time.perf_counter() - embed_started

        logger.info("writing float32 truth layer")
        np.save(temporary / "float32-truth.npy", matrix)
        logger.info("writing calibrated INT8 refinement layer")
        int8_scales = fit_int8_scales(matrix)
        np.save(temporary / "int8-scales.npy", int8_scales)
        int8_path = temporary / "int8-vectors.bin"
        quantize_normalized(matrix, int8_scales).tofile(int8_path)

        logger.info("building %s candidate index", backend)
        index_started = time.perf_counter()
        if backend == "faiss":
            FaissPQIndex.build(matrix, temporary, stage)
        elif backend == "numpy":
            NumpyFlatIndex.build(matrix, temporary)
        else:
            raise ValueError("backend must be 'faiss' or 'numpy'")
        index_seconds = time.perf_counter() - index_started

        logger.info("candidate index built in %.1fs; writing metadata and title FTS", index_seconds)
        metadata_started = time.perf_counter()
        store = MetadataStore(temporary / "metadata.sqlite3")
        try:
            store.create_schema()
            store.insert_papers(selected)
        finally:
            store.close()
        metadata_seconds = time.perf_counter() - metadata_started

        float32_truth_bytes = (temporary / "float32-truth.npy").stat().st_size
        int8_bytes = int8_path.stat().st_size + (temporary / "int8-scales.npy").stat().st_size
        metadata_bytes = sum(
            item.stat().st_size
            for item in temporary.glob("metadata.sqlite3*")
            if item.is_file()
        )
        index_file = (
            temporary / FaissPQIndex.FILENAME
            if backend == "faiss"
            else temporary / NumpyFlatIndex.FILENAME
        )
        candidate_index_bytes = index_file.stat().st_size
        production_bytes = int8_bytes + metadata_bytes + candidate_index_bytes

        manifest = {
            "format_version": 1,
            "stage": stage.name,
            "records": len(selected),
            "dimension": VECTOR_DIMENSION,
            "embedder": embedder.name,
            "backend": backend,
            "candidate_count": min(stage.candidate_count, len(selected)),
            "nprobe": stage.nprobe,
            "build_seconds": time.perf_counter() - started,
            "embed_seconds": embed_seconds,
            "embed_records_per_second": len(selected) / max(embed_seconds, 1e-9),
            "index_seconds": index_seconds,
            "metadata_seconds": metadata_seconds,
            "component_bytes": {
                "benchmark_float32_truth": float32_truth_bytes,
                "candidate_index": candidate_index_bytes,
                "int8_refinement": int8_bytes,
                "metadata_and_title_fts": metadata_bytes,
            },
            "measured_production_bytes": production_bytes,
            "measured_production_bytes_per_record": production_bytes / len(selected),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest["generation_bytes"] = _directory_bytes(temporary)
        manifest["benchmark_bytes_per_record"] = manifest["generation_bytes"] / len(selected)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        if output.exists():
            raise FileExistsError(
                f"Refusing to replace existing generation {output}; choose a new output path"
            )
        temporary.replace(output)
        logger.info(
            "generation published at %s (%.1fs total, %.1f MB)",
            output,
            manifest["build_seconds"],
            manifest["generation_bytes"] / 1e6,
        )
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
