from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Stage


class CandidateIndex(Protocol):
    def search(
        self, query: np.ndarray, limit: int, nprobe: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...


class NumpyFlatIndex:
    """Exact inner-product index for smoke tests and Float32 ground truth."""

    FILENAME = "flat-vectors.npy"

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=np.float32)

    @classmethod
    def build(cls, vectors: np.ndarray, directory: Path) -> "NumpyFlatIndex":
        np.save(directory / cls.FILENAME, np.asarray(vectors, dtype=np.float32))
        return cls(vectors)

    @classmethod
    def load(cls, directory: Path) -> "NumpyFlatIndex":
        return cls(np.load(directory / cls.FILENAME, mmap_mode="r"))

    def search(
        self, query: np.ndarray, limit: int, nprobe: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        # Exact: there are no lists to probe, so nprobe has nothing to widen.
        scores = self.vectors @ np.asarray(query, dtype=np.float32)
        count = min(limit, len(scores))
        if count == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        ids = np.argpartition(scores, -count)[-count:]
        order = np.argsort(scores[ids])[::-1]
        selected = ids[order]
        return scores[selected], selected.astype(np.int64)


class FaissPQIndex:
    FILENAME = "ivfpq.faiss"
    IVF_DATA_FILENAME = "ivfpq-lists.bin"

    def __init__(self, index):
        self.index = index

    @staticmethod
    def _faiss():
        try:
            import faiss
        except ImportError as error:  # pragma: no cover - runtime dependency
            raise RuntimeError("Install faiss-cpu to build the IVF-PQ benchmark") from error
        return faiss

    @classmethod
    def build(cls, vectors: np.ndarray, directory: Path, stage: Stage) -> "FaissPQIndex":
        faiss = cls._faiss()
        values = np.ascontiguousarray(vectors, dtype=np.float32)
        if len(values) < 256:
            raise ValueError("IVF-PQ needs at least 256 records; use the smoke backend below that")
        list_count = min(stage.ivf_lists, max(1, len(values) // 40))
        quantizer = faiss.IndexFlatIP(values.shape[1])
        index = faiss.IndexIVFPQ(
            quantizer,
            values.shape[1],
            list_count,
            stage.pq_subquantizers,
            stage.pq_bits,
            faiss.METRIC_INNER_PRODUCT,
        )
        train_limit = min(len(values), max(10_000, list_count * 64))
        rng = np.random.default_rng(20260831)
        train_ids = rng.choice(len(values), size=train_limit, replace=False)
        index.train(np.ascontiguousarray(values[train_ids]))
        index.add_with_ids(values, np.arange(len(values), dtype=np.int64))
        index.nprobe = min(stage.nprobe, list_count)
        faiss.write_index(index, str(directory / cls.FILENAME))
        return cls(index)

    @classmethod
    def load(cls, directory: Path, nprobe: int) -> "FaissPQIndex":
        faiss = cls._faiss()
        flags = 0
        if (directory / cls.IVF_DATA_FILENAME).exists():
            # The build host and serving host use different absolute paths.
            # Resolve an OnDiskInvertedLists payload beside the index header.
            flags = faiss.IO_FLAG_READ_ONLY | faiss.IO_FLAG_ONDISK_SAME_DIR
        index = faiss.read_index(str(directory / cls.FILENAME), flags)
        index.nprobe = min(nprobe, index.nlist)
        return cls(index)

    def search(
        self, query: np.ndarray, limit: int, nprobe: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if nprobe is None or int(nprobe) == int(self.index.nprobe):
            scores, ids = self.index.search(values, limit)
        else:
            # Per-call parameters, never index.nprobe: the index is shared by
            # every request thread, and mutating it would widen or narrow
            # concurrent searches it was not meant for.
            params = self._faiss().SearchParametersIVF(
                nprobe=max(1, min(int(nprobe), int(self.index.nlist)))
            )
            scores, ids = self.index.search(values, limit, params=params)
        valid = ids[0] >= 0
        return scores[0][valid], ids[0][valid]
