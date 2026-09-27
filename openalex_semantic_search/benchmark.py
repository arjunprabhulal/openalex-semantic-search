from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import statistics
import time
from typing import Sequence

logger = logging.getLogger(__name__)

import numpy as np

from .embeddings import Embedder
from .engine import SearchEngine
from .config import FULL_CORPUS_RECORDS, QUERY_BUDGET_MS, STORAGE_BUDGET_BYTES
from .quantization import int8_scores
from .store import Filters, normalize_title


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile_value))


def recall(expected: Sequence[int], actual: Sequence[int]) -> float:
    expected_set = set(expected)
    return len(expected_set.intersection(actual)) / max(1, len(expected_set))


def _latency_run(engine: SearchEngine, query: str) -> tuple[float, str | None]:
    started = time.perf_counter()
    try:
        engine.search(query, limit=20)
        return (time.perf_counter() - started) * 1_000, None
    except Exception as error:  # The report must retain failures instead of aborting.
        return (time.perf_counter() - started) * 1_000, type(error).__name__


def run_benchmark(
    generation: Path,
    embedder: Embedder,
    queries: Sequence[str],
    *,
    title_queries: Sequence[str] | None = None,
    repeats: int = 3,
    concurrency_levels: Sequence[int] = (1, 2, 4),
) -> dict:
    if not queries:
        raise ValueError("At least one benchmark query is required")
    engine = SearchEngine(generation, embedder)
    truth_path = generation / "float32-truth.npy"
    truth = np.load(truth_path, mmap_mode="r") if truth_path.exists() else None
    truth_mode = "float32" if truth is not None else "int8_exact_scan"
    if truth is None:
        # Full float32 truth is benchmark-only and infeasible at large scale;
        # exact scan of the dequantized INT8 layer measures candidate-index
        # loss precisely (INT8 fidelity is checked separately below).
        logger.info("no float32 truth matrix; using exact INT8 scan as ground truth")

    def exact_scores(vector: np.ndarray) -> np.ndarray:
        if truth is not None:
            return truth @ vector
        scores = np.empty(engine.record_count, dtype=np.float32)
        for offset in range(0, engine.record_count, 250_000):
            block = slice(offset, min(offset + 250_000, engine.record_count))
            scores[block] = int8_scores(vector, engine.codes[block], engine.int8_scales)
        return scores

    try:
        logger.info("measuring recall on %d semantic queries", len(queries))
        pq_recall_at_candidates: list[float] = []
        int8_recall_at_20: list[float] = []
        for query in queries:
            vector = embedder.encode_query(query)
            truth_scores = exact_scores(vector)
            # argpartition: O(n) top-k instead of a full corpus sort per query.
            top = np.argpartition(truth_scores, -20)[-20:]
            exact_ids = top[np.argsort(truth_scores[top])[::-1]].tolist()
            _, pq_ids_array = engine.index.search(vector, engine.candidate_count)
            pq_ids = [int(value) for value in pq_ids_array]
            pq_recall_at_candidates.append(recall(exact_ids, pq_ids))
            refined_scores = int8_scores(vector, engine.codes[pq_ids], engine.int8_scales)
            refined_order = np.argsort(refined_scores)[::-1][:20]
            refined_ids = [pq_ids[index] for index in refined_order]
            int8_recall_at_20.append(recall(exact_ids, refined_ids))
        resolved_title_queries = list(title_queries if title_queries is not None else queries)
        if not resolved_title_queries:
            raise ValueError("At least one exact-title query is required")
        logger.info("running %d exact-title probes", len(resolved_title_queries))
        title_probe_results: list[dict] = []
        title_successes = 0
        for query in resolved_title_queries:
            response = engine.search(query, limit=20)
            normalized_query = normalize_title(query)
            matched = next(
                (
                    result
                    for result in response.results
                    if normalized_query in normalize_title(result.title)
                ),
                None,
            )
            success = matched is not None
            if success:
                title_successes += 1
            title_probe_results.append(
                {
                    "query": query,
                    "success": success,
                    "matched_title": matched.title if matched else None,
                    "rank": next(
                        (
                            index + 1
                            for index, result in enumerate(response.results)
                            if matched is not None and result.row_id == matched.row_id
                        ),
                        None,
                    ),
                }
            )

        concurrency: dict[str, dict] = {}
        workload = [query for query in queries for _ in range(repeats)]
        for workers in concurrency_levels:
            logger.info(
                "latency run: %d requests at concurrency %d", len(workload), workers
            )
            level_started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                outcomes = list(executor.map(lambda value: _latency_run(engine, value), workload))
            wall_seconds = time.perf_counter() - level_started
            latencies = [latency for latency, _ in outcomes]
            errors: dict[str, int] = {}
            for _, error_name in outcomes:
                if error_name:
                    errors[error_name] = errors.get(error_name, 0) + 1
            concurrency[str(workers)] = {
                "requests": len(latencies),
                "p50_ms": percentile(latencies, 50),
                "p95_ms": percentile(latencies, 95),
                "p99_ms": percentile(latencies, 99),
                "mean_ms": statistics.fmean(latencies),
                "throughput_qps": len(latencies) / max(wall_seconds, 1e-9),
                "errors": errors,
            }

        # Force the selective path and verify every returned row obeys it.
        logger.info("checking selective-filter correctness")
        rare_year_row = engine.store.rarest_year()
        filter_correct = True
        selective_count = 0
        if rare_year_row:
            year = int(rare_year_row[0])
            selective = engine.search(
                queries[0],
                limit=20,
                filters=Filters(year_min=year, year_max=year),
            )
            selective_count = selective.filtered_records
            filter_correct = all(result.publication_year == year for result in selective.results)

        # INT8 quantization fidelity vs the shipped float32 sample (assembled
        # generations): cosine between the true vector and its dequantized row.
        fidelity = None
        sample_path = generation / "float32-truth-sample.npy"
        if sample_path.exists():
            sample_vectors = np.load(sample_path)
            sample_ids = np.load(generation / "float32-truth-ids.npy")
            truth_rows_path = generation / "float32-truth-row-ids.npy"
            truth_rows = (
                np.load(truth_rows_path, mmap_mode="r")
                if truth_rows_path.exists()
                else None
            )
            if truth_rows is not None and len(truth_rows) != len(sample_ids):
                raise RuntimeError("Float32 truth row ids do not align with truth samples")
            picks = np.linspace(0, len(sample_ids) - 1, min(1_000, len(sample_ids))).astype(int)
            cosines = []
            for pick in picks:
                if truth_rows is not None:
                    row_id = int(truth_rows[pick])
                else:
                    row_id = engine.store.row_id_for_openalex_id(
                        f"https://openalex.org/W{int(sample_ids[pick])}"
                    )
                if row_id is None:
                    continue
                true_vec = sample_vectors[pick]
                dequant = engine.codes[row_id].astype(np.float32) * engine.int8_scales
                denom = float(np.linalg.norm(true_vec) * np.linalg.norm(dequant)) or 1.0
                cosines.append(float(true_vec @ dequant) / denom)
            if cosines:
                fidelity = {
                    "checked": len(cosines),
                    "cos_mean": statistics.fmean(cosines),
                    "cos_min": min(cosines),
                }
                logger.info(
                    "INT8 fidelity: cos mean %.4f min %.4f over %d samples",
                    fidelity["cos_mean"], fidelity["cos_min"], fidelity["checked"],
                )

        manifest = engine.manifest
        production_bytes_per_record = manifest["measured_production_bytes_per_record"]
        projected_full_corpus_bytes = production_bytes_per_record * FULL_CORPUS_RECORDS
        latency_budget_pass = all(
            not level["errors"] and level["p99_ms"] <= QUERY_BUDGET_MS
            for level in concurrency.values()
        )
        return {
            "records": engine.record_count,
            "embedder": embedder.name,
            "backend": manifest["backend"],
            "truth_mode": truth_mode,
            "int8_fidelity": fidelity,
            "pq_recall_at_candidates_mean": statistics.fmean(pq_recall_at_candidates),
            "pq_recall_at_candidates_min": min(pq_recall_at_candidates),
            "int8_refined_recall_at_20_mean": statistics.fmean(int8_recall_at_20),
            "int8_refined_recall_at_20_min": min(int8_recall_at_20),
            "exact_title_success_rate": title_successes / len(resolved_title_queries),
            "exact_title_probes": title_probe_results,
            "selective_filter_correct": filter_correct,
            "selective_filter_records": selective_count,
            "concurrency": concurrency,
            "component_bytes": manifest["component_bytes"],
            "component_build_seconds": {
                "total": manifest["build_seconds"],
                "embedding": manifest["embed_seconds"],
                "candidate_index": manifest["index_seconds"],
                "metadata_and_title_fts": manifest["metadata_seconds"],
            },
            "benchmark_bytes_per_record": manifest["benchmark_bytes_per_record"],
            "measured_production_bytes_per_record": production_bytes_per_record,
            "full_corpus_records_for_projection": FULL_CORPUS_RECORDS,
            "projected_full_corpus_bytes": projected_full_corpus_bytes,
            "storage_budget_bytes": STORAGE_BUDGET_BYTES,
            "query_budget_ms": QUERY_BUDGET_MS,
            "embed_records_per_second": manifest["embed_records_per_second"],
            "gate": {
                "pq_candidate_recall_pass": statistics.fmean(pq_recall_at_candidates) >= 0.95,
                "int8_refined_recall_at_20_pass": statistics.fmean(int8_recall_at_20) >= 0.95,
                "exact_title_pass": title_successes == len(resolved_title_queries),
                "filter_pass": filter_correct,
                "storage_projection_pass": projected_full_corpus_bytes
                <= STORAGE_BUDGET_BYTES,
                "latency_and_stability_pass": latency_budget_pass,
            },
        }
    finally:
        engine.close()


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
