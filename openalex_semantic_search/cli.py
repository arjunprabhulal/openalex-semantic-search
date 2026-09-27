from __future__ import annotations

import argparse
from itertools import chain
import json
import logging
import os
from pathlib import Path
import sys

from .assemble import assemble_generation
from .backfill import run_backfill
from .benchmark import run_benchmark, write_report
from .title_prefix import build_title_prefix_index
from .work_index import build_work_id_index
from .builder import build_generation
from .config import MODEL_NAME, STAGE_ORDER, get_stage, require_stage_confirmation
from .embeddings import HashingEmbedder, SentenceTransformerEmbedder
from .records import (
    iter_benchmark_seed_works,
    iter_local_works,
    iter_s3_sampled_works,
    usable_papers,
)


def _embedder(args):
    if getattr(args, "smoke_embedder", False):
        return HashingEmbedder()
    return SentenceTransformerEmbedder(
        getattr(args, "model", MODEL_NAME),
        device=getattr(args, "device", "cpu"),
        batch_size=getattr(args, "batch_size", 128),
        max_tokens=getattr(args, "max_tokens", None),
        workers=getattr(args, "embed_workers", 1),
        tokenizer_processes=getattr(args, "tokenizer_processes", 0),
        tokenizer_threads=getattr(args, "tokenizer_threads", 1),
        dtype=getattr(args, "dtype", "float32"),
    )


def build_command(args) -> int:
    stage = get_stage(args.stage)
    require_stage_confirmation(stage, args.confirm)
    if args.input:
        works = iter_local_works(args.input)
    else:
        # The small overlay makes exact-title probes deterministic. The same IDs
        # are deduplicated if their full records also occur in the S3 sample.
        works = chain(
            iter_benchmark_seed_works(),
            iter_s3_sampled_works(stage.records),
        )
    papers = usable_papers(works, stage.records)
    manifest = build_generation(
        papers,
        stage=stage,
        output=args.output,
        embedder=_embedder(args),
        backend=args.backend,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def benchmark_command(args) -> int:
    queries_data = json.loads(args.queries.read_text(encoding="utf-8"))
    if isinstance(queries_data, list):
        semantic_queries = queries_data
        title_queries = queries_data
    elif isinstance(queries_data, dict):
        semantic_queries = queries_data.get("semantic")
        title_queries = queries_data.get("exact_titles")
    else:
        raise ValueError("queries file must be a JSON array or benchmark query object")
    if not isinstance(semantic_queries, list) or not all(
        isinstance(item, str) for item in semantic_queries
    ):
        raise ValueError("queries file must contain a semantic string array")
    if not isinstance(title_queries, list) or not all(
        isinstance(item, str) for item in title_queries
    ):
        raise ValueError("queries file must contain an exact_titles string array")
    report = run_benchmark(
        args.generation,
        _embedder(args),
        semantic_queries,
        title_queries=title_queries,
        repeats=args.repeats,
    )
    write_report(report, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all(report["gate"].values()) else 1


def backfill_command(args) -> int:
    # Snapshot identity is a correctness requirement, not best-effort
    # telemetry: a long GPU run must never span two quarterly releases.
    from .records import snapshot_metadata

    snapshot_meta = snapshot_metadata()
    manifest = run_backfill(
        args.output,
        _embedder(args),
        shard_size=args.shard_size,
        embed_subchunk=args.embed_subchunk,
        limit=args.limit,
        seed_works=(
            iter_benchmark_seed_works()
            if args.limit is not None and not args.no_benchmark_seeds
            else None
        ),
        snapshot_meta=snapshot_meta,
        snapshot_probe=snapshot_metadata,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "checksums"}, indent=2))
    return 0


def assemble_command(args) -> int:
    stage = get_stage(args.stage)
    require_stage_confirmation(stage, args.confirm)
    backfill_manifest = json.loads(
        (args.artifacts / "backfill-manifest.json").read_text(encoding="utf-8")
    )
    manifest = assemble_generation(
        args.artifacts,
        stage=stage,
        output=args.output,
        backend=args.backend,
        metadata_backend=args.metadata_backend,
        metadata_workers=args.metadata_workers,
        metadata_block_rows=args.metadata_block_rows,
        embedder_name=backfill_manifest["embedder"],
        allow_partial=args.allow_partial_stage,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def build_work_id_command(args) -> int:
    """Build the id lookup index from an existing generation, without touching it."""
    from .compact_store import CompactMetadataStore

    manifest = build_work_id_index(
        args.generation,
        args.output,
        store_factory=CompactMetadataStore,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def build_title_prefix_command(args) -> int:
    """Build the prefix index from an existing generation, without touching it."""
    from .compact_store import CompactMetadataStore, stable_text_hash

    manifest = build_title_prefix_index(
        args.generation,
        args.output,
        hasher=stable_text_hash,
        store_factory=CompactMetadataStore,
        batch_rows=args.batch_rows,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def serve_command(args) -> int:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("Install runtime dependencies to serve the app") from error
    os.environ["OPENALEX_SEARCH_INDEX"] = str(args.generation)
    if getattr(args, "overrides", None):
        os.environ["OPENALEX_SEARCH_OVERRIDES"] = str(args.overrides)
    if getattr(args, "title_prefixes", None):
        os.environ["OPENALEX_SEARCH_TITLE_PREFIXES"] = str(args.title_prefixes)
    if getattr(args, "work_ids", None):
        os.environ["OPENALEX_SEARCH_WORK_IDS"] = str(args.work_ids)
    if getattr(args, "supplement", None):
        os.environ["OPENALEX_SEARCH_SUPPLEMENT"] = str(args.supplement)
    uvicorn.run(
        "openalex_semantic_search.api:app_from_environment",
        factory=True,
        host=args.host,
        port=args.port,
        workers=1,
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="openalex-semantic-search")
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build one immutable benchmark generation")
    build.add_argument("--stage", choices=STAGE_ORDER, default="10k")
    build.add_argument("--confirm")
    build.add_argument("--input", type=Path, help="local .jsonl or .jsonl.gz; omit for public S3")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--backend", choices=("faiss", "numpy"), default="faiss")
    build.add_argument("--model", default=MODEL_NAME)
    build.add_argument("--device", default="cpu")
    build.add_argument("--batch-size", type=int, default=128)
    build.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    build.add_argument(
        "--embed-workers",
        type=int,
        default=1,
        help="embedding worker processes; try cores/4 to use the whole CPU",
    )
    build.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="cap token length per text (speed/quality trade; default: model max)",
    )
    build.add_argument("--smoke-embedder", action="store_true")
    build.set_defaults(run=build_command)

    benchmark = commands.add_parser(
        "benchmark",
        help="measure recall, title hits, filters and latency",
    )
    benchmark.add_argument("--generation", type=Path, required=True)
    benchmark.add_argument("--queries", type=Path, required=True)
    benchmark.add_argument("--report", type=Path, required=True)
    benchmark.add_argument("--model", default=MODEL_NAME)
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--batch-size", type=int, default=128)
    benchmark.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    benchmark.add_argument("--repeats", type=int, default=3)
    benchmark.add_argument("--smoke-embedder", action="store_true")
    benchmark.set_defaults(run=benchmark_command)

    backfill = commands.add_parser(
        "backfill-embed",
        help="stream the snapshot and emit sharded embedding artifacts (GPU box)",
    )
    backfill.add_argument("--output", type=Path, required=True)
    backfill.add_argument("--shard-size", type=int, default=1_000_000)
    backfill.add_argument(
        "--embed-subchunk",
        type=int,
        default=65_536,
        help=(
            "maximum records per GPU embedding call; two Float32 chunks may be "
            "resident while CPU artifact writes overlap inference"
        ),
    )
    backfill.add_argument("--limit", type=int, default=None)
    backfill.add_argument("--model", default=MODEL_NAME)
    backfill.add_argument("--device", default="cuda")
    backfill.add_argument("--batch-size", type=int, default=1024)
    backfill.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    backfill.add_argument("--max-tokens", type=int, default=None)
    backfill.add_argument("--embed-workers", type=int, default=1)
    backfill.add_argument(
        "--tokenizer-processes",
        type=int,
        default=0,
        help="CPU tokenizer processes feeding one GPU model; 0 uses the reference path",
    )
    backfill.add_argument(
        "--tokenizer-threads",
        type=int,
        default=1,
        help="Rayon threads per tokenizer process",
    )
    backfill.add_argument(
        "--no-benchmark-seeds",
        action="store_true",
        help="do not prepend the five rich title probes to a finite validation run",
    )
    backfill.add_argument("--smoke-embedder", action="store_true")
    backfill.set_defaults(run=backfill_command)

    assemble = commands.add_parser(
        "assemble",
        help="assemble a servable generation from backfill artifacts, streaming",
    )
    assemble.add_argument("--artifacts", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--stage", choices=STAGE_ORDER, required=True)
    assemble.add_argument("--confirm")
    assemble.add_argument("--backend", choices=("faiss", "numpy"), default="faiss")
    assemble.add_argument(
        "--metadata-backend",
        choices=("sqlite", "compact"),
        default="sqlite",
        help="compact is the bounded-storage rich metadata backend for the full corpus",
    )
    assemble.add_argument("--metadata-workers", type=int, default=8)
    assemble.add_argument("--metadata-block-rows", type=int, default=256)
    assemble.add_argument("--allow-partial-stage", action="store_true")
    assemble.add_argument(
        "--resume",
        action="store_true",
        help="preserve and resume checksum-verified assembly checkpoints",
    )
    assemble.set_defaults(run=assemble_command)

    title_prefix = commands.add_parser(
        "build-title-prefix-index",
        help="build a leading-word-prefix title index over an existing generation",
    )
    title_prefix.add_argument("--generation", type=Path, required=True)
    title_prefix.add_argument("--output", type=Path, required=True)
    title_prefix.add_argument("--batch-rows", type=int, default=200_000)
    title_prefix.add_argument("--workers", type=int, default=1)
    title_prefix.set_defaults(run=build_title_prefix_command)

    work_ids = commands.add_parser(
        "build-work-id-index",
        help="build an OpenAlex id lookup index over an existing generation",
    )
    work_ids.add_argument("--generation", type=Path, required=True)
    work_ids.add_argument("--output", type=Path, required=True)
    work_ids.add_argument("--workers", type=int, default=1)
    work_ids.set_defaults(run=build_work_id_command)

    serve = commands.add_parser("serve", help="serve the standalone API and search interface")
    serve.add_argument(
        "--overrides",
        type=Path,
        default=None,
        help="JSON file of hand-verified metadata corrections applied at read time",
    )
    serve.add_argument(
        "--title-prefixes",
        type=Path,
        default=None,
        help="directory holding a prebuilt title prefix index",
    )
    serve.add_argument(
        "--work-ids",
        type=Path,
        default=None,
        help="directory holding a prebuilt OpenAlex id lookup index",
    )
    serve.add_argument(
        "--supplement",
        type=Path,
        default=None,
        help="JSON file of hand-verified records for works missing from the generation",
    )
    serve.add_argument("--generation", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8100)
    serve.set_defaults(run=serve_command)
    return root


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = parser().parse_args(argv)
    try:
        return int(args.run(args))
    except (RuntimeError, ValueError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
