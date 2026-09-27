from __future__ import annotations

from hashlib import blake2b
import os
from pathlib import Path
import re
from typing import Protocol, Sequence

import numpy as np

from .config import MODEL_NAME, QUERY_PREFIX, VECTOR_DIMENSION

TOKEN_PATTERN = re.compile(r"[\w-]+", re.UNICODE)


class Embedder(Protocol):
    dimension: int
    name: str

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, query: str) -> np.ndarray: ...


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


class HashingEmbedder:
    """Dependency-light deterministic embedder used only by tests and smoke runs."""

    def __init__(self, dimension: int = VECTOR_DIMENSION):
        self.dimension = dimension
        self.name = f"hashing-smoke-{dimension}"

    def _one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in TOKEN_PATTERN.findall(text.casefold()):
            digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
            integer = int.from_bytes(digest, "little")
            vector[integer % self.dimension] += 1.0 if integer & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._one(text) for text in texts], axis=0)

    def encode_query(self, query: str) -> np.ndarray:
        return self._one(query)


def _init_embed_worker(
    model_name, device, batch_size, cache_folder, max_tokens, dtype, threads
):
    """Initializer for embedding worker processes: one model per worker,
    pinned to a slice of the CPU so workers don't fight over threads."""
    global _EMBED_WORKER  # noqa: PLW0603 - per-process singleton
    import torch

    torch.set_num_threads(threads)
    _EMBED_WORKER = SentenceTransformerEmbedder(
        model_name,
        device=device,
        batch_size=batch_size,
        cache_folder=cache_folder,
        max_tokens=max_tokens,
        dtype=dtype,
    )


def _embed_worker_encode(texts: list[str]) -> np.ndarray:
    return _EMBED_WORKER.encode_documents(texts)


_TOKENIZER = None
_TOKENIZER_MAX_TOKENS = 512
_TOKENIZER_DO_LOWER_CASE = False


def _init_tokenizer_worker(
    model_name: str,
    cache_folder: str | None,
    max_tokens: int,
    rayon_threads: int,
    do_lower_case: bool,
) -> None:
    """Initialize one CPU-only tokenizer process without loading the model."""
    global _TOKENIZER, _TOKENIZER_MAX_TOKENS, _TOKENIZER_DO_LOWER_CASE
    os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_folder,
        local_files_only=True,
        use_fast=True,
    )
    _TOKENIZER_MAX_TOKENS = max_tokens
    _TOKENIZER_DO_LOWER_CASE = do_lower_case


def _tokenize_worker_batch(
    item: tuple[int, list[str]],
) -> tuple[int, dict[str, object]]:
    """Tokenize one length-sorted batch and preserve its sequence number."""
    if _TOKENIZER is None:  # pragma: no cover - process initializer invariant
        raise RuntimeError("tokenizer worker was not initialized")
    batch_index, texts = item
    prepared = [str(text).strip() for text in texts]
    if _TOKENIZER_DO_LOWER_CASE:
        prepared = [text.lower() for text in prepared]
    features = _TOKENIZER(
        prepared,
        padding=True,
        truncation="longest_first",
        return_tensors="pt",
        max_length=_TOKENIZER_MAX_TOKENS,
    )
    return batch_index, dict(features)


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        *,
        device: str = "cpu",
        batch_size: int = 128,
        cache_folder: str | Path | None = None,
        max_tokens: int | None = None,
        workers: int = 1,
        tokenizer_processes: int = 0,
        tokenizer_threads: int = 1,
        dtype: str = "float32",
    ):
        if workers <= 0:
            raise ValueError("workers must be positive")
        if tokenizer_processes < 0:
            raise ValueError("tokenizer_processes must not be negative")
        if tokenizer_threads <= 0:
            raise ValueError("tokenizer_threads must be positive")
        if workers > 1 and tokenizer_processes:
            raise ValueError(
                "embedding workers and tokenizer processes cannot be enabled together"
            )
        configured_cache = cache_folder or os.getenv("OPENALEX_SEARCH_MODEL_CACHE") or None
        if configured_cache:
            cache_path = Path(configured_cache).expanduser()
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise RuntimeError(
                    f"Could not create model cache {cache_path}: {error}"
                ) from error
            if not os.access(cache_path, os.W_OK):
                raise RuntimeError(
                    f"Model cache {cache_path} is not writable by the service user"
                )
            configured_cache = str(cache_path)
            # huggingface_hub resolves its token/xet paths from $HF_HOME at import
            # time, so this must be set before sentence_transformers is imported;
            # otherwise the service user hits EACCES under an unreadable $HOME.
            os.environ.setdefault("HF_HOME", configured_cache)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:  # pragma: no cover - runtime dependency
            raise RuntimeError("Install the runtime dependencies to use BGE embeddings") from error
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=configured_cache,
        )
        normalized_dtype = dtype.casefold()
        dtype_names = {
            "float32": "float32",
            "bfloat16": "bfloat16",
            "float16": "float16",
        }
        if normalized_dtype not in dtype_names:
            choices = ", ".join(dtype_names)
            raise ValueError(f"dtype must be one of: {choices}")
        if normalized_dtype != "float32":
            import torch

            if str(device).startswith("cpu") and normalized_dtype == "float16":
                raise ValueError("float16 inference is not supported on CPU")
            self.model.to(dtype=getattr(torch, dtype_names[normalized_dtype]))
        self.name = model_name
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = batch_size
        self.dtype = normalized_dtype
        if max_tokens is not None:
            if not 32 <= max_tokens <= int(self.model.max_seq_length):
                raise ValueError(
                    f"max_tokens must be between 32 and {self.model.max_seq_length}"
                )
            self.model.max_seq_length = max_tokens
        self._config = (
            model_name,
            device,
            batch_size,
            configured_cache,
            max_tokens,
            normalized_dtype,
        )
        self.workers = workers
        self._pool = None
        self.tokenizer_processes = tokenizer_processes
        self.tokenizer_threads = tokenizer_threads
        self._tokenizer_pool = None
        self._tokenizer_model_name = model_name
        self._tokenizer_do_lower_case = False
        if tokenizer_processes:
            transformer = self.model[0]
            self._tokenizer_model_name = str(
                getattr(transformer.tokenizer, "name_or_path", model_name)
            )
            self._tokenizer_do_lower_case = bool(transformer.do_lower_case)
        if self.dimension != VECTOR_DIMENSION:
            raise RuntimeError(
                f"Model {model_name} produces {self.dimension} dimensions; "
                f"expected {VECTOR_DIMENSION}"
            )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=min(self.batch_size, max(1, len(texts))),
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _ensure_tokenizer_pool(self):
        if self._tokenizer_pool is not None:
            return self._tokenizer_pool
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        max_tokens = int(self.model.max_seq_length)
        self._tokenizer_pool = ProcessPoolExecutor(
            max_workers=self.tokenizer_processes,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_tokenizer_worker,
            initargs=(
                self._tokenizer_model_name,
                self._config[3],
                max_tokens,
                self.tokenizer_threads,
                self._tokenizer_do_lower_case,
            ),
        )
        # Start and initialize every process before the first measured batch.
        warmups = [
            (index, ["warmup"] * 256) for index in range(self.tokenizer_processes)
        ]
        list(self._tokenizer_pool.map(_tokenize_worker_batch, warmups, chunksize=1))
        return self._tokenizer_pool

    def _encode_process_tokenized(self, texts: Sequence[str]) -> np.ndarray:
        """Run exact SentenceTransformers forward passes on process-tokenized batches.

        Only tokenization moves to CPU worker processes. Transformer, native
        pooling, truncation, normalization, and order restoration mirror the
        installed SentenceTransformers encode path.
        """
        import torch
        from sentence_transformers.util import batch_to_device, truncate_embeddings

        values = list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        self.model.eval()
        length_sorted_idx = np.argsort(
            [-self.model._text_length(text) for text in values]
        )
        sorted_texts = [values[index] for index in length_sorted_idx]
        indexed_batches = [
            (batch_index, sorted_texts[offset : offset + self.batch_size])
            for batch_index, offset in enumerate(
                range(0, len(sorted_texts), self.batch_size)
            )
        ]

        all_embeddings: list[torch.Tensor] = []
        pool = self._ensure_tokenizer_pool()
        for expected_index, result in enumerate(
            pool.map(_tokenize_worker_batch, indexed_batches, chunksize=1)
        ):
            batch_index, features = result
            if batch_index != expected_index:  # pragma: no cover - executor contract
                raise RuntimeError(
                    "tokenizer batch order changed: "
                    f"expected {expected_index}, got {batch_index}"
                )
            device_features = batch_to_device(features, self.model.device)
            with torch.no_grad():
                out_features = self.model.forward(device_features)
                embeddings = truncate_embeddings(
                    out_features["sentence_embedding"], self.model.truncate_dim
                ).detach()
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(embeddings.cpu())

        sorted_vectors = np.asarray(
            [embedding.float().numpy() for embedding in all_embeddings],
            dtype=np.float32,
        )
        return sorted_vectors[np.argsort(length_sorted_idx)]

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        if self.tokenizer_processes and len(texts) >= self.tokenizer_processes * 256:
            return self._encode_process_tokenized(texts)
        if self.workers == 1 or len(texts) < self.workers * 256:
            return self._encode(texts)
        if self._pool is None:
            import multiprocessing
            import os

            threads = max(1, (os.cpu_count() or self.workers) // self.workers)
            self._pool = multiprocessing.get_context("spawn").Pool(
                self.workers,
                initializer=_init_embed_worker,
                initargs=(*self._config, threads),
            )
        texts = list(texts)
        step = (len(texts) + self.workers - 1) // self.workers
        shards = [texts[i : i + step] for i in range(0, len(texts), step)]
        return np.concatenate(self._pool.map(_embed_worker_encode, shards), axis=0)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None
        if self._tokenizer_pool is not None:
            self._tokenizer_pool.shutdown(wait=True, cancel_futures=True)
            self._tokenizer_pool = None

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([f"{QUERY_PREFIX}{query}"])[0]
