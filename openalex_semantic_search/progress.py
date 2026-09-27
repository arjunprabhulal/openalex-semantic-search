from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


_SHARD_MARKER = re.compile(r"^shard-(\d{5})\.done$")


def _read_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_index_progress(
    artifacts: Path | None,
    *,
    indexed_records: int,
) -> dict[str, Any]:
    """Summarize only crash-atomic backfill checkpoints visible on this host.

    Replication copies a shard's ``.done`` marker last, after its artifacts and
    checksums have been verified.  This reader therefore ignores temporary
    files, stops at the first marker gap or inconsistency, and never presents
    an in-flight shard as durable progress.
    """

    result: dict[str, Any] = {
        "available": False,
        "state": "unavailable",
        "indexed_records": max(0, int(indexed_records)),
        "verified_embedding_records": 0,
        "committed_shards": 0,
        "source_files_reached": None,
        "source_files_total": None,
        "source_progress_percent": None,
        "snapshot_date": None,
        "last_checkpoint_at": None,
    }
    if artifacts is None or not artifacts.is_dir():
        return result

    markers: dict[int, Path] = {}
    try:
        candidates = artifacts.glob("shard-*.done")
        for path in candidates:
            match = _SHARD_MARKER.fullmatch(path.name)
            if match:
                markers[int(match.group(1))] = path
    except OSError:
        return result

    verified = 0
    completed = 0
    source_file_index: int | None = None
    last_checkpoint: float | None = None
    while completed in markers:
        path = markers[completed]
        marker = _read_object(path)
        if marker is None:
            break
        try:
            records = int(marker["records"])
            records_done = int(marker["records_done"])
            file_index = int(marker["file_index"])
        except (KeyError, TypeError, ValueError):
            break
        if records <= 0 or records_done != verified + records or file_index < 0:
            break
        try:
            modified = path.stat().st_mtime
        except OSError:
            break
        verified = records_done
        source_file_index = file_index
        last_checkpoint = max(last_checkpoint or modified, modified)
        completed += 1

    snapshot = _read_object(artifacts / "snapshot-pin.json") or {}
    try:
        source_files_total = int(snapshot.get("file_count"))
    except (TypeError, ValueError):
        source_files_total = 0
    if source_files_total <= 0:
        source_files_total = None

    source_files_reached = source_file_index + 1 if source_file_index is not None else None
    source_percent = None
    if source_files_reached is not None and source_files_total:
        source_percent = round(
            min(100.0, (source_files_reached / source_files_total) * 100.0),
            2,
        )

    manifest = _read_object(artifacts / "backfill-manifest.json") or {}
    try:
        manifest_records = int(manifest.get("records", -1))
        manifest_shards = int(manifest.get("shards", -1))
    except (TypeError, ValueError):
        manifest_records = manifest_shards = -1
    complete = bool(
        manifest.get("source_exhausted")
        and manifest_records == verified
        and manifest_shards == completed
    )
    checkpoint_at = None
    if last_checkpoint is not None:
        checkpoint_at = datetime.fromtimestamp(last_checkpoint, tz=timezone.utc).isoformat()

    result.update(
        available=True,
        state="complete" if complete else "building",
        verified_embedding_records=verified,
        committed_shards=completed,
        source_files_reached=source_files_reached,
        source_files_total=source_files_total,
        source_progress_percent=source_percent,
        snapshot_date=snapshot.get("manifest_date"),
        last_checkpoint_at=checkpoint_at,
    )
    return result
