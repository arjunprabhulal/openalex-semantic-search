"""Sweep the read-time FAISS settings on the served generation. Read-only.

nprobe and the candidate count were tuned on the 1M stage (4.7% of lists
probed) and carried to the full corpus unmeasured (0.15% probed). Both are
query-time parameters, so this measures them directly against the generation
on disk without changing it:

* recall@20 of each (nprobe, k) setting against a reference search with a much
  larger nprobe and k, after the same INT8 rescoring the service applies
  (an exact scan of 315M vectors is ~121GB of reads per query, so the
  reference is a wide approximate search, and the recall is relative to it);
* whether each labelled paper reaches the candidate pool at all;
* candidate-search latency, median and p95.

Run it on the serving host during a quiet period; it loads the index and the
query model in a second process and competes for CPU:

    nice -n 19 python benchmark/tune_search.py \
        --generation /var/lib/openalex-semantic-search/generations/full-v1 \
        --work-ids /path/to/work-id-index \
        --nprobe 192,384,768 --candidates 1000,2000,4000 --threads 4

Then set OPENALEX_SEARCH_NPROBE / OPENALEX_SEARCH_CANDIDATES to the chosen values.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None, *, embedder=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--work-ids", type=Path, help="work id index, to locate labelled papers")
    parser.add_argument("--queries", type=Path, default=HERE / "labelled-queries.json")
    parser.add_argument("--nprobe", default="192,384,768")
    parser.add_argument("--candidates", default="1000,2000,4000")
    parser.add_argument("--reference-nprobe", type=int, default=4096)
    parser.add_argument("--reference-k", type=int, default=8000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4, help="FAISS OpenMP threads")
    parser.add_argument("--model", default=os.environ.get("OPENALEX_SEARCH_MODEL", "BAAI/bge-small-en-v1.5"))
    parser.add_argument("--out", type=Path, help="write the full report as JSON")
    args = parser.parse_args(argv)

    import faiss

    from openalex_semantic_search.embeddings import SentenceTransformerEmbedder
    from openalex_semantic_search.engine import SearchEngine
    from openalex_semantic_search.quantization import int8_scores
    from openalex_semantic_search.work_index import WorkIdIndex

    faiss.omp_set_num_threads(args.threads)
    nprobes = [int(value) for value in args.nprobe.split(",")]
    ks = [int(value) for value in args.candidates.split(",")]
    engine = SearchEngine(
        args.generation,
        embedder or SentenceTransformerEmbedder(args.model, device="cpu"),
        work_ids=WorkIdIndex.load(args.work_ids) if args.work_ids else None,
    )
    if engine.manifest["backend"] != "faiss":
        engine.close()
        print("This sweep is for FAISS generations.", file=sys.stderr)
        return 2
    queries = [
        entry
        for entry in json.loads(args.queries.read_text(encoding="utf-8"))["queries"]
        if "expect" in entry and entry["request"].get("sort", "relevance") == "relevance"
    ]

    def top20(vector: np.ndarray, ids: np.ndarray) -> set[int]:
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return set()
        scores = int8_scores(vector, engine.codes[np.sort(ids)], engine.int8_scales)
        order = np.argsort(scores)[::-1][:20]
        return {int(value) for value in np.sort(ids)[order]}

    def labelled_rows(entry: dict) -> set[int]:
        if engine.work_ids is None:
            return set()
        from openalex_semantic_search.compact_store import stable_text_hash

        rows = set()
        for identifier in entry["expect"].get("work_ids", ()):
            rows.update(engine.work_ids.candidate_rows(identifier, stable_text_hash))
        return rows

    report: dict[str, dict] = {}
    try:
        prepared = []
        for entry in queries:
            vector = engine._embed_query(entry["request"]["query"])
            _, reference = engine.index.search(vector, args.reference_k, nprobe=args.reference_nprobe)
            prepared.append((entry, vector, top20(vector, reference), labelled_rows(entry)))
        for nprobe in nprobes:
            for k in ks:
                recalls, reached, labelled, latencies = [], 0, 0, []
                for entry, vector, truth, targets in prepared:
                    for _ in range(args.repeats):
                        started = time.perf_counter()
                        _, ids = engine.index.search(vector, k, nprobe=nprobe)
                        latencies.append((time.perf_counter() - started) * 1000)
                    recalls.append(len(truth & top20(vector, ids)) / max(1, len(truth)))
                    if targets:
                        labelled += 1
                        reached += bool(targets & {int(value) for value in ids})
                latencies.sort()
                name = f"nprobe={nprobe} k={k}"
                report[name] = {
                    "recall_at_20_mean": round(statistics.fmean(recalls), 4),
                    "recall_at_20_min": round(min(recalls), 4),
                    "labelled_in_pool": f"{reached}/{labelled}",
                    "latency_ms_p50": round(latencies[len(latencies) // 2], 1),
                    "latency_ms_p95": round(latencies[int(len(latencies) * 0.95) - 1], 1),
                }
                print(name, report[name], flush=True)
    finally:
        engine.close()
    print(
        f"\nReference: nprobe={args.reference_nprobe}, k={args.reference_k}. "
        f"Served today: nprobe={engine.nprobe}, k={engine.candidate_count}."
    )
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
