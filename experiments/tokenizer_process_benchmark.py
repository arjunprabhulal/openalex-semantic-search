from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing as mp
import os
from pathlib import Path
import time


_TOKENIZER = None
_MAX_TOKENS = 512


def _load_texts(path: Path, limit: int) -> list[str]:
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


def _init_worker(model_path: str, max_tokens: int, rayon_threads: int) -> None:
    global _TOKENIZER, _MAX_TOKENS
    os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )
    _MAX_TOKENS = max_tokens


def _tokenize(batch: list[str]) -> tuple[int, tuple[int, ...], float]:
    if _TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    started = time.perf_counter()
    features = _TOKENIZER(
        batch,
        padding=True,
        truncation=True,
        max_length=_MAX_TOKENS,
        return_tensors="pt",
    )
    elapsed = time.perf_counter() - started
    return len(batch), tuple(features["input_ids"].shape), elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--records", type=int, default=32_768)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--rayon-threads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.records <= 0 or args.batch_size <= 0 or args.processes <= 0:
        parser.error("records, batch-size, and processes must be positive")
    texts = _load_texts(args.metadata, args.records)
    texts.sort(key=len, reverse=True)
    batches = [
        texts[offset : offset + args.batch_size]
        for offset in range(0, len(texts), args.batch_size)
    ]

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.processes,
        mp_context=context,
        initializer=_init_worker,
        initargs=(str(args.model), args.max_tokens, args.rayon_threads),
    ) as pool:
        # Initialize every process before the measured pass.
        warmups = [["warmup"] * 256 for _ in range(args.processes)]
        list(pool.map(_tokenize, warmups))
        started = time.perf_counter()
        results = list(pool.map(_tokenize, batches))
        wall_seconds = time.perf_counter() - started

    processed = sum(result[0] for result in results)
    report = {
        "batch_shapes": [result[1] for result in results],
        "batch_tokenize_seconds": [round(result[2], 6) for result in results],
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "processes": args.processes,
        "rayon_threads_per_process": args.rayon_threads,
        "records": processed,
        "wall_seconds": round(wall_seconds, 6),
        "works_per_second": round(processed / wall_seconds, 2),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
