from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrays_equal(reference: Path, candidate: Path, rows_per_chunk: int = 65_536) -> bool:
    left = np.load(reference, mmap_mode="r")
    right = np.load(candidate, mmap_mode="r")
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.ndim == 0:
        return bool(np.array_equal(left, right))
    for offset in range(0, len(left), rows_per_chunk):
        if not np.array_equal(
            left[offset : offset + rows_per_chunk],
            right[offset : offset + rows_per_chunk],
        ):
            return False
    return True


def _gzip_payloads_equal(reference: Path, candidate: Path) -> bool:
    with gzip.open(reference, "rb") as left, gzip.open(candidate, "rb") as right:
        while True:
            left_chunk = left.read(4 * 1024 * 1024)
            right_chunk = right.read(4 * 1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--records", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()

    candidate_manifest = json.loads(
        (args.candidate / "backfill-manifest.json").read_text(encoding="utf-8")
    )
    if candidate_manifest["records"] != args.records:
        raise RuntimeError("candidate record count does not match the gate")
    if candidate_manifest["shards"] != args.shards:
        raise RuntimeError("candidate shard count does not match the gate")

    for name, expected in candidate_manifest["checksums"].items():
        actual = _sha256(args.candidate / name)
        if actual != expected:
            raise RuntimeError(f"candidate checksum mismatch: {name}")

    snapshot_reference = json.loads(
        (args.reference / "snapshot-pin.json").read_text(encoding="utf-8")
    )
    snapshot_candidate = json.loads(
        (args.candidate / "snapshot-pin.json").read_text(encoding="utf-8")
    )
    if snapshot_reference.get("manifest_etag") != snapshot_candidate.get("manifest_etag"):
        raise RuntimeError("reference and candidate use different OpenAlex snapshots")

    if not _arrays_equal(
        args.reference / "int8-scales.npy",
        args.candidate / "int8-scales.npy",
    ):
        raise RuntimeError("INT8 calibration scales differ")

    all_ids: list[np.ndarray] = []
    compared: dict[str, int] = {"array_files": 1, "metadata_files": 0}
    for shard in range(args.shards):
        stem = f"shard-{shard:05d}"
        for suffix in ("ids.npy", "int8.npy", "truth.npy", "tids.npy"):
            name = f"{stem}.{suffix}"
            if not _arrays_equal(args.reference / name, args.candidate / name):
                raise RuntimeError(f"reference/candidate array mismatch: {name}")
            compared["array_files"] += 1
        metadata_name = f"{stem}.meta.jsonl.gz"
        if not _gzip_payloads_equal(
            args.reference / metadata_name,
            args.candidate / metadata_name,
        ):
            raise RuntimeError(f"reference/candidate metadata mismatch: {metadata_name}")
        compared["metadata_files"] += 1
        all_ids.append(np.load(args.candidate / f"{stem}.ids.npy"))

    ids = np.concatenate(all_ids)
    if len(ids) != args.records or len(np.unique(ids)) != args.records:
        raise RuntimeError("candidate IDs are incomplete or duplicated")

    print(
        json.dumps(
            {
                "candidate": str(args.candidate),
                "compared": compared,
                "exact_reference_match": True,
                "records": args.records,
                "shards": args.shards,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
