from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np


_TOKENIZER = None
_MAX_TOKENS = 512
_DO_LOWER_CASE = False


def _load_texts(path: Path, limit: int) -> list[str]:
    """Rebuild representative document text from committed card metadata.

    Metadata stores a 1,500-character abstract snippet while production embeds
    up to 2,000 characters. These texts therefore reproduce the runtime shape
    closely, but equivalence below compares the two code paths on the exact
    same strings rather than claiming to reconstruct prior production inputs.
    """
    texts: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as rows:
        for line in rows:
            row = json.loads(line)
            title = str(row.get("title") or "")
            snippet = str(row.get("snippet") or "")
            texts.append(f"{title}. {snippet}" if snippet else title)
            if len(texts) == limit:
                break
    if len(texts) != limit:
        raise RuntimeError(f"expected {limit} texts, found {len(texts)}")
    return texts


def _init_tokenizer_worker(
    model_path: str,
    max_tokens: int,
    rayon_threads: int,
    do_lower_case: bool,
) -> None:
    global _TOKENIZER, _MAX_TOKENS, _DO_LOWER_CASE
    os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    _MAX_TOKENS = max_tokens
    _DO_LOWER_CASE = do_lower_case


def _tokenize_batch(
    item: tuple[int, list[str]],
) -> tuple[int, dict[str, object], float]:
    if _TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    batch_index, texts = item
    prepared = [str(text).strip() for text in texts]
    if _DO_LOWER_CASE:
        prepared = [text.lower() for text in prepared]
    started = time.perf_counter()
    features = _TOKENIZER(
        prepared,
        padding=True,
        truncation="longest_first",
        return_tensors="pt",
        max_length=_MAX_TOKENS,
    )
    elapsed = time.perf_counter() - started
    return batch_index, dict(features), elapsed


def _reference_encode(model, texts: list[str], batch_size: int) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - started
    return np.asarray(vectors, dtype=np.float32), elapsed


def _process_tokenized_encode(
    model,
    texts: list[str],
    *,
    model_path: Path,
    batch_size: int,
    processes: int,
    rayon_threads: int,
    max_tokens: int,
) -> tuple[np.ndarray, dict[str, object]]:
    import torch
    from sentence_transformers.util import batch_to_device, truncate_embeddings

    model.eval()
    length_sorted_idx = np.argsort([-model._text_length(text) for text in texts])
    sorted_texts = [texts[index] for index in length_sorted_idx]
    indexed_batches = [
        (batch_index, sorted_texts[offset : offset + batch_size])
        for batch_index, offset in enumerate(range(0, len(sorted_texts), batch_size))
    ]

    context = mp.get_context("spawn")
    all_embeddings: list[torch.Tensor] = []
    tokenization_seconds: list[float] = []
    shapes: list[tuple[int, ...]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=processes,
        mp_context=context,
        initializer=_init_tokenizer_worker,
        initargs=(
            str(model_path),
            max_tokens,
            rayon_threads,
            bool(model[0].do_lower_case),
        ),
    ) as pool:
        # Pay process and tokenizer initialization before the measured pass.
        warmups = [(index, ["warmup"] * 256) for index in range(processes)]
        list(pool.map(_tokenize_batch, warmups))
        measured_started = time.perf_counter()
        for expected_index, result in enumerate(
            pool.map(_tokenize_batch, indexed_batches)
        ):
            batch_index, features, tokenization_elapsed = result
            if batch_index != expected_index:
                raise RuntimeError(
                    f"tokenizer batch order changed: expected {expected_index}, got {batch_index}"
                )
            shapes.append(tuple(features["input_ids"].shape))
            tokenization_seconds.append(tokenization_elapsed)
            features = batch_to_device(features, model.device)
            with torch.no_grad():
                out_features = model.forward(features)
                embeddings = truncate_embeddings(
                    out_features["sentence_embedding"], model.truncate_dim
                ).detach()
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(embeddings.cpu())
        measured_seconds = time.perf_counter() - measured_started
    total_seconds = time.perf_counter() - started

    sorted_vectors = np.asarray(
        [embedding.float().numpy() for embedding in all_embeddings],
        dtype=np.float32,
    )
    vectors = sorted_vectors[np.argsort(length_sorted_idx)]
    return vectors, {
        "batch_shapes": shapes,
        "batch_tokenize_seconds": tokenization_seconds,
        "measured_seconds": measured_seconds,
        "total_with_worker_startup_seconds": total_seconds,
    }


def _equivalence_report(
    reference: np.ndarray,
    candidate: np.ndarray,
    scales_path: Path | None,
) -> dict[str, object]:
    if reference.shape != candidate.shape:
        raise RuntimeError(
            f"candidate shape {candidate.shape} does not match reference {reference.shape}"
        )
    absolute = np.abs(reference - candidate)
    dot_products = np.einsum("ij,ij->i", reference, candidate)
    denominators = np.linalg.norm(reference, axis=1) * np.linalg.norm(candidate, axis=1)
    cosine = np.divide(
        dot_products,
        denominators,
        out=np.zeros_like(dot_products),
        where=denominators > 0,
    )
    report: dict[str, object] = {
        "array_equal": bool(np.array_equal(reference, candidate)),
        "allclose_atol_1e_5": bool(np.allclose(reference, candidate, atol=1e-5, rtol=0)),
        "max_abs_error": float(np.max(absolute)),
        "mean_abs_error": float(np.mean(absolute)),
        "min_cosine": float(np.min(cosine)),
        "mean_cosine": float(np.mean(cosine)),
    }
    if scales_path is not None:
        from openalex_semantic_search.quantization import quantize_normalized

        scales = np.load(scales_path)
        reference_codes = quantize_normalized(reference, scales)
        candidate_codes = quantize_normalized(candidate, scales)
        matches = reference_codes == candidate_codes
        report["int8_all_equal"] = bool(np.all(matches))
        report["int8_element_match_fraction"] = float(np.mean(matches))
        report["int8_rows_all_equal_fraction"] = float(np.mean(np.all(matches, axis=1)))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scales", type=Path)
    parser.add_argument("--records", type=int, default=65_536)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--rayon-threads", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    for name in ("records", "batch_size", "processes", "rayon_threads", "max_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")

    from sentence_transformers import SentenceTransformer

    texts = _load_texts(args.metadata, args.records)
    model = SentenceTransformer(str(args.model), device=args.device)
    model.max_seq_length = args.max_tokens
    if args.dtype != "float32":
        import torch

        model.to(dtype=getattr(torch, args.dtype))

    # Warm GPU kernels before both measured paths.
    model.encode(
        texts[:256],
        batch_size=256,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    reference, reference_seconds = _reference_encode(model, texts, args.batch_size)
    candidate, fast_metrics = _process_tokenized_encode(
        model,
        texts,
        model_path=args.model,
        batch_size=args.batch_size,
        processes=args.processes,
        rayon_threads=args.rayon_threads,
        max_tokens=args.max_tokens,
    )
    equivalence = _equivalence_report(reference, candidate, args.scales)
    fast_seconds = float(fast_metrics["measured_seconds"])
    report = {
        "configuration": {
            "batch_size": args.batch_size,
            "device": args.device,
            "dtype": args.dtype,
            "max_tokens": args.max_tokens,
            "processes": args.processes,
            "rayon_threads_per_process": args.rayon_threads,
            "records": args.records,
        },
        "equivalence": equivalence,
        "fast_path": {
            **fast_metrics,
            "works_per_second": args.records / fast_seconds,
        },
        "reference": {
            "seconds": reference_seconds,
            "works_per_second": args.records / reference_seconds,
        },
        "speedup": reference_seconds / fast_seconds,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not equivalence["allclose_atol_1e_5"] or equivalence["min_cosine"] < 0.999999:
        return 2
    if args.scales is not None and not equivalence["int8_all_equal"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
