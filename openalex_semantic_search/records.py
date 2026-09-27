from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full, Queue
import random
from threading import Event
import time
import traceback
from typing import Any, BinaryIO, Callable, Iterable, Iterator

logger = logging.getLogger(__name__)


# S3 ingestion overlaps network reads, gzip decompression, and JSON parsing for
# a small window of files. Each worker may buffer only this many record batches,
# keeping memory use independent of the number of files in the manifest.
_S3_DOWNLOAD_WORKERS = 4
_S3_RECORD_BATCH_SIZE = 256
_S3_PREFETCH_BATCHES_PER_FILE = 2
_S3_FILE_DONE = object()
_S3_PROCESS_WORKERS = max(1, int(os.environ.get("OPENALEX_SEARCH_INGEST_WORKERS", "8")))
_S3_STREAM_ATTEMPTS = max(1, int(os.environ.get("OPENALEX_SEARCH_S3_ATTEMPTS", "5")))


@dataclass(frozen=True, slots=True)
class _S3FileFailure:
    error: Exception


@dataclass(slots=True)
class _S3FileChannel:
    file_index: int
    key: str
    messages: Queue[object]
    future: Future[None]


@dataclass(frozen=True, slots=True)
class _S3ParsedFileDone:
    logical_records: int
    accepted_papers: int
    attempts: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class _S3ParsedFileFailure:
    message: str
    traceback_text: str


@dataclass(slots=True)
class _S3ProcessChannel:
    file_index: int
    key: str
    first_line: int
    messages: Any
    terminal: Any
    process: Any


@dataclass(frozen=True, slots=True)
class Paper:
    openalex_id: str
    title: str
    embedding_text: str
    snippet: str
    authors: tuple[str, ...]
    publication_year: int
    doi: str
    cited_by_count: int
    topic: str
    field: str
    is_oa: bool
    oa_url: str
    work_type: str = ""
    venue: str = ""
    publication_date: str = ""
    landing_url: str = ""

    @property
    def view_url(self) -> str:
        """Preferred landing page for a reader."""
        return self.doi or self.landing_url or self.oa_url or self.openalex_id


def reconstruct_abstract(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        positioned.extend(
            (position, word)
            for position in positions
            if isinstance(position, int) and position >= 0
        )
    positioned.sort(key=lambda item: item[0])
    return " ".join(word for _, word in positioned)


def _clean_text(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def parse_work(
    raw: object,
    *,
    snippet_chars: int = 1_500,
    include_expansion: bool = False,
) -> Paper | None:
    if not isinstance(raw, dict):
        return None
    if not include_expansion and raw.get("is_xpac", False):
        return None
    openalex_id = _clean_text(raw.get("id"))
    title = _clean_text(raw.get("title") or raw.get("display_name"))
    if not openalex_id.startswith("https://openalex.org/W") or not title:
        return None

    abstract = reconstruct_abstract(raw.get("abstract_inverted_index"))
    embedding_text = f"{title}. {abstract}" if abstract else title

    authors: list[str] = []
    for authorship in raw.get("authorships") or []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        name = _clean_text(author.get("display_name")) if isinstance(author, dict) else ""
        if name:
            authors.append(name)
        if len(authors) == 8:
            break

    topic = ""
    field = ""
    primary_topic = raw.get("primary_topic")
    if isinstance(primary_topic, dict):
        topic = _clean_text(primary_topic.get("display_name"))
        topic_field = primary_topic.get("field")
        if isinstance(topic_field, dict):
            field = _clean_text(topic_field.get("display_name"))

    open_access = raw.get("open_access")
    if not isinstance(open_access, dict):
        open_access = {}

    venue = ""
    landing_url = ""
    primary_location = raw.get("primary_location")
    if isinstance(primary_location, dict):
        source = primary_location.get("source")
        if isinstance(source, dict):
            venue = _clean_text(source.get("display_name"))
        landing_url = _clean_text(primary_location.get("landing_page_url"))

    return Paper(
        openalex_id=openalex_id,
        title=title,
        embedding_text=embedding_text[:2_000],
        snippet=abstract[:snippet_chars],
        authors=tuple(authors),
        publication_year=int(raw.get("publication_year") or 0),
        doi=_clean_text(raw.get("doi")),
        cited_by_count=max(0, int(raw.get("cited_by_count") or 0)),
        topic=topic,
        field=field,
        is_oa=bool(open_access.get("is_oa", False)),
        oa_url=_clean_text(open_access.get("oa_url")),
        work_type=_clean_text(raw.get("type")),
        venue=venue,
        publication_date=_clean_text(raw.get("publication_date")),
        landing_url=landing_url,
    )


def _iter_json_lines(stream: BinaryIO) -> Iterator[dict]:
    for line in stream:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, dict):
            yield value


def iter_local_works(path: Path) -> Iterator[dict]:
    with path.open("rb") as raw:
        if path.suffix == ".gz":
            with gzip.GzipFile(fileobj=raw, mode="rb") as archive:
                yield from _iter_json_lines(archive)
        else:
            yield from _iter_json_lines(raw)


def iter_benchmark_seed_works() -> Iterator[dict]:
    """Return a tiny, auditable title-probe overlay for staged benchmarks."""
    path = Path(__file__).with_name("benchmark-seed-works.json")
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise RuntimeError("benchmark-seed-works.json must contain a JSON array")
    for value in values:
        if isinstance(value, dict):
            yield value


def iter_s3_works(
    *,
    bucket: str = "openalex",
    prefix: str = "data/jsonl/works/",
) -> Iterator[dict]:
    """Stream public OpenAlex snapshot objects without writing archives to disk."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("Install the runtime dependencies to read OpenAlex S3") from error

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            if not key.endswith(".gz"):
                continue
            response = client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            try:
                with gzip.GzipFile(fileobj=body, mode="rb") as archive:
                    yield from _iter_json_lines(archive)
            finally:
                body.close()


def snapshot_metadata(
    *, bucket: str = "openalex", prefix: str = "data/jsonl/works/"
) -> dict:
    """Identify the exact snapshot a backfill ran against, for provenance."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("Install the runtime dependencies to read OpenAlex S3") from error
    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    manifest_key = f"{prefix.rstrip('/')}/manifest.json"
    response = client.get_object(Bucket=bucket, Key=manifest_key)
    try:
        payload = response["Body"].read()
    finally:
        response["Body"].close()
    manifest = json.loads(payload)
    files = [f for f in manifest.get("files", []) if isinstance(f, dict)]
    return {
        "manifest_key": f"s3://{bucket}/{manifest_key}",
        "manifest_etag": response.get("ETag", "").strip('"'),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "manifest_date": manifest.get("date"),
        "manifest_last_modified": str(response.get("LastModified", "")),
        "file_count": len(files),
        "raw_record_total": sum(
            (f.get("meta") or {}).get("record_count", 0) for f in files
        ),
    }


def latest_first_manifest_files(files: object) -> list[dict]:
    """Return snapshot partitions newest-first.

    OpenAlex re-emits changed entities into newer ``updated_date`` partitions
    while older copies remain present. Streaming newest-first makes the
    backfill's bounded first-ID-wins dedupe select the latest record.
    """
    if not isinstance(files, list):
        return []
    valid = [item for item in files if isinstance(item, dict)]
    return sorted(valid, key=lambda item: str(item.get("url", "")), reverse=True)


def _put_s3_message(messages: Queue[object], message: object, stop: Event) -> bool:
    """Put into a bounded worker queue while remaining cancellable.

    The timeout matters when a caller stops consuming after reaching a finite
    record limit: generator finalization can set ``stop`` and release workers
    whose queues are full.
    """
    while not stop.is_set():
        try:
            messages.put(message, timeout=0.1)
            return True
        except Full:
            continue
    return False


def _stream_s3_file_to_queue(
    client,
    *,
    bucket: str,
    key: str,
    first_line: int,
    messages: Queue[object],
    stop: Event,
) -> None:
    """Download, decompress, and parse one object into bounded batches."""
    try:
        if stop.is_set():
            return
        obj = client.get_object(Bucket=bucket, Key=key)
        body = obj["Body"]
        try:
            batch: list[tuple[int, dict]] = []
            with gzip.GzipFile(fileobj=body, mode="rb") as archive:
                # Keep the historical checkpoint contract: line_index counts
                # parsed dictionary records, not physical JSONL lines.
                for line_index, work in enumerate(_iter_json_lines(archive)):
                    if stop.is_set():
                        return
                    if line_index < first_line:
                        continue
                    batch.append((line_index, work))
                    if len(batch) >= _S3_RECORD_BATCH_SIZE:
                        if not _put_s3_message(messages, batch, stop):
                            return
                        batch = []
                if batch and not _put_s3_message(messages, batch, stop):
                    return
        finally:
            body.close()
    except Exception as error:
        _put_s3_message(messages, _S3FileFailure(error), stop)
        raise
    else:
        _put_s3_message(messages, _S3_FILE_DONE, stop)


def _iter_ordered_s3_files(
    client,
    *,
    bucket: str,
    files: list[dict],
    skip_files: int,
    skip_lines: int,
    max_workers: int = _S3_DOWNLOAD_WORKERS,
) -> Iterator[tuple[int, int, dict]]:
    """Parse S3 files concurrently while yielding strict manifest order.

    Only ``max_workers`` files are scheduled at once and each has a bounded
    queue. A newer file is always drained completely before any older file is
    yielded, even if the older download finishes first.
    """
    marker = f"s3://{bucket}/"
    jobs = (
        (
            file_index,
            str(files[file_index].get("url", ""))[len(marker) :],
            skip_lines if file_index == skip_files else 0,
        )
        for file_index in range(skip_files, len(files))
        if str(files[file_index].get("url", "")).startswith(marker)
        and str(files[file_index].get("url", "")).endswith(".gz")
    )
    worker_count = max(1, int(max_workers))
    stop = Event()
    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="openalex-s3",
    )
    pending: deque[_S3FileChannel] = deque()

    def schedule_next() -> bool:
        try:
            file_index, key, first_line = next(jobs)
        except StopIteration:
            return False
        messages: Queue[object] = Queue(maxsize=_S3_PREFETCH_BATCHES_PER_FILE)
        future = executor.submit(
            _stream_s3_file_to_queue,
            client,
            bucket=bucket,
            key=key,
            first_line=first_line,
            messages=messages,
            stop=stop,
        )
        pending.append(_S3FileChannel(file_index, key, messages, future))
        return True

    try:
        for _ in range(worker_count):
            if not schedule_next():
                break

        while pending:
            channel = pending.popleft()
            logger.info(
                "streaming snapshot file %d/%d: %s",
                channel.file_index + 1,
                len(files),
                channel.key,
            )
            while True:
                message = channel.messages.get()
                if message is _S3_FILE_DONE:
                    break
                if isinstance(message, _S3FileFailure):
                    # Re-raise through Future.result() to retain the worker's
                    # original traceback.
                    channel.future.result()
                    raise message.error  # pragma: no cover - defensive
                batch = message
                if not isinstance(batch, list):  # pragma: no cover - defensive
                    raise RuntimeError("S3 ingestion worker returned an invalid batch")
                for line_index, work in batch:
                    yield channel.file_index, line_index, work
            channel.future.result()
            schedule_next()
    finally:
        stop.set()
        executor.shutdown(wait=True, cancel_futures=True)


def _unsigned_s3_client():
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("Install the runtime dependencies to read OpenAlex S3") from error
    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            connect_timeout=30,
            read_timeout=120,
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )


def _put_process_message(messages: Any, message: object, stop: Any) -> bool:
    while not stop.is_set():
        try:
            messages.put(message, timeout=0.1)
            return True
        except Full:
            continue
    return False


def _stream_s3_file_to_parsed_queue(
    *,
    bucket: str,
    key: str,
    first_line: int,
    include_expansion: bool,
    messages: Any,
    stop: Any,
    terminal: Any | None = None,
    max_attempts: int = _S3_STREAM_ATTEMPTS,
) -> None:
    """Parse one S3 object in a separate process with cursor-safe retries.

    A batch is acknowledged only after it enters the process queue. If a
    response stream breaks, the worker reopens the gzip member and skips to
    the first unacknowledged logical record. This prevents a transient network
    failure from restarting an entire 1M embedding shard or yielding records
    twice.
    """
    started = time.monotonic()
    next_line = max(0, int(first_line))
    accepted_papers = 0
    client = _unsigned_s3_client()

    def publish_terminal(message: object) -> None:
        if terminal is None:
            _put_process_message(messages, message, stop)
        else:
            terminal.send(message)

    for attempt in range(1, max(1, int(max_attempts)) + 1):
        if stop.is_set():
            return
        body = None
        try:
            response = client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            batch: list[tuple[int, Paper | None]] = []
            with gzip.GzipFile(fileobj=body, mode="rb") as archive:
                for line_index, raw in enumerate(_iter_json_lines(archive)):
                    if stop.is_set():
                        return
                    if line_index < next_line:
                        continue
                    paper = parse_work(raw, include_expansion=include_expansion)
                    batch.append((line_index, paper))
                    if len(batch) >= _S3_RECORD_BATCH_SIZE:
                        if not _put_process_message(messages, batch, stop):
                            return
                        next_line = batch[-1][0] + 1
                        accepted_papers += sum(paper is not None for _, paper in batch)
                        batch = []
                if batch:
                    if not _put_process_message(messages, batch, stop):
                        return
                    next_line = batch[-1][0] + 1
                    accepted_papers += sum(paper is not None for _, paper in batch)
        except Exception as error:
            if stop.is_set():
                return
            if attempt >= max(1, int(max_attempts)):
                publish_terminal(
                    _S3ParsedFileFailure(str(error), traceback.format_exc())
                )
                return
            delay = min(10.0, float(2 ** (attempt - 1)))
            logger.warning(
                "snapshot stream %s failed at logical line %d on attempt %d/%d; "
                "retrying in %.1fs: %s",
                key,
                next_line,
                attempt,
                max_attempts,
                delay,
                error,
            )
            if stop.wait(delay):
                return
            continue
        finally:
            if body is not None:
                try:
                    body.close()
                except Exception:
                    pass
        # A multiprocessing.Queue serializes through a background feeder
        # thread. A worker can therefore exit cleanly while its final queue
        # marker is still unavailable (or a feeder failure can drop it). Send
        # the small terminal envelope synchronously over a pipe instead. The
        # coordinator uses its record count to prove every queued batch was
        # received before accepting completion.
        publish_terminal(
            _S3ParsedFileDone(
                logical_records=next_line,
                accepted_papers=accepted_papers,
                attempts=attempt,
                elapsed_seconds=time.monotonic() - started,
            )
        )
        return


def _iter_ordered_s3_papers(
    *,
    bucket: str,
    files: list[dict],
    skip_files: int,
    skip_lines: int,
    include_expansion: bool,
    max_workers: int = _S3_PROCESS_WORKERS,
    process_context: Any | None = None,
    worker_target: Callable[..., None] = _stream_s3_file_to_parsed_queue,
) -> Iterator[tuple[int, int, Paper | None]]:
    """Parse manifest files in real CPU processes and yield strict order.

    Each process may finish and buffer one complete snapshot part while the
    coordinator drains an earlier part. The number of live parts is bounded by
    ``max_workers``; RAM use therefore depends on that fixed window, never the
    entire manifest.
    """
    marker = f"s3://{bucket}/"
    jobs = (
        (
            file_index,
            str(files[file_index].get("url", ""))[len(marker) :],
            skip_lines if file_index == skip_files else 0,
        )
        for file_index in range(skip_files, len(files))
        if str(files[file_index].get("url", "")).startswith(marker)
        and str(files[file_index].get("url", "")).endswith(".gz")
    )
    worker_count = max(1, int(max_workers))
    context = process_context or mp.get_context("spawn")
    stop = context.Event()
    pending: deque[_S3ProcessChannel] = deque()
    active: list[_S3ProcessChannel] = []

    def schedule_next() -> bool:
        try:
            file_index, key, first_line = next(jobs)
        except StopIteration:
            return False
        # Deliberately unbounded per file: a later process must be able to
        # finish while an earlier file is drained, otherwise strict ordering
        # collapses the pool back to one effective core. The worker window is
        # the memory bound.
        messages = context.Queue()
        terminal_receive, terminal_send = context.Pipe(duplex=False)
        process = context.Process(
            target=worker_target,
            kwargs={
                "bucket": bucket,
                "key": key,
                "first_line": first_line,
                "include_expansion": include_expansion,
                "messages": messages,
                "terminal": terminal_send,
                "stop": stop,
                "max_attempts": _S3_STREAM_ATTEMPTS,
            },
            name=f"openalex-s3-{file_index}",
            daemon=True,
        )
        channel = _S3ProcessChannel(
            file_index,
            key,
            first_line,
            messages,
            terminal_receive,
            process,
        )
        process.start()
        terminal_send.close()
        pending.append(channel)
        active.append(channel)
        return True

    try:
        for _ in range(worker_count):
            if not schedule_next():
                break

        while pending:
            channel = pending.popleft()
            logger.info(
                "draining parsed snapshot file %d/%d: %s",
                channel.file_index + 1,
                len(files),
                channel.key,
            )
            terminal_message: _S3ParsedFileDone | _S3ParsedFileFailure | None = None
            terminal_eof = False
            received_records = 0
            while True:
                if (
                    terminal_message is None
                    and not terminal_eof
                    and channel.terminal.poll()
                ):
                    try:
                        terminal_message = channel.terminal.recv()
                    except EOFError:
                        terminal_eof = True
                if terminal_message is not None:
                    if isinstance(terminal_message, _S3ParsedFileFailure):
                        raise RuntimeError(
                            f"S3 ingestion failed for {channel.key}: "
                            f"{terminal_message.message}\n"
                            f"{terminal_message.traceback_text}"
                        )
                    if not isinstance(terminal_message, _S3ParsedFileDone):
                        raise RuntimeError(
                            "S3 parser process returned an invalid terminal message"
                        )
                    expected_records = terminal_message.logical_records - channel.first_line
                    if received_records == expected_records:
                        logger.info(
                            "parsed snapshot file %d/%d: %d logical records, %d usable "
                            "papers in %.1fs (%d attempt%s)",
                            channel.file_index + 1,
                            len(files),
                            terminal_message.logical_records,
                            terminal_message.accepted_papers,
                            terminal_message.elapsed_seconds,
                            terminal_message.attempts,
                            "" if terminal_message.attempts == 1 else "s",
                        )
                        break
                    if received_records > expected_records:
                        raise RuntimeError(
                            f"S3 parser returned {received_records} records for {channel.key}; "
                            f"terminal envelope promised {expected_records}"
                        )
                try:
                    message = channel.messages.get(timeout=0.1)
                except Empty:
                    if channel.process.is_alive():
                        continue
                    channel.process.join()
                    if (
                        terminal_message is None
                        and not terminal_eof
                        and channel.terminal.poll()
                    ):
                        try:
                            terminal_message = channel.terminal.recv()
                        except EOFError:
                            terminal_eof = True
                        if terminal_message is not None:
                            continue
                    raise RuntimeError(
                        "S3 parser process exited before completing "
                        f"{channel.key} (exit code {channel.process.exitcode})"
                    )
                if not isinstance(message, list):  # pragma: no cover - defensive
                    raise RuntimeError("S3 parser process returned an invalid batch")
                received_records += len(message)
                for line_index, paper in message:
                    yield channel.file_index, line_index, paper
            channel.process.join(timeout=5)
            if channel.process.is_alive():
                raise RuntimeError(f"S3 parser process did not exit for {channel.key}")
            if channel.process.exitcode != 0:
                raise RuntimeError(
                    f"S3 parser process exited {channel.process.exitcode} for {channel.key}"
                )
            active.remove(channel)
            channel.terminal.close()
            channel.messages.close()
            channel.messages.join_thread()
            channel.process.close()
            schedule_next()
    finally:
        stop.set()
        for channel in active:
            channel.process.join(timeout=2)
            if channel.process.is_alive():
                channel.process.terminate()
                channel.process.join(timeout=5)
            try:
                channel.terminal.close()
            except Exception:
                pass
            try:
                channel.messages.close()
                channel.messages.cancel_join_thread()
            except Exception:
                pass
            try:
                channel.process.close()
            except Exception:
                pass


def iter_s3_manifest_papers(
    *,
    bucket: str = "openalex",
    prefix: str = "data/jsonl/works/",
    skip_files: int = 0,
    skip_lines: int = 0,
    include_expansion: bool = False,
    max_workers: int = _S3_PROCESS_WORKERS,
) -> Iterator[tuple[int, int, Paper | None]]:
    """Parse the pinned snapshot in ordered, retrying worker processes."""
    client = _unsigned_s3_client()
    manifest_key = f"{prefix.rstrip('/')}/manifest.json"
    response = client.get_object(Bucket=bucket, Key=manifest_key)
    try:
        manifest = json.loads(response["Body"].read())
    finally:
        response["Body"].close()
    files = latest_first_manifest_files(manifest.get("files"))
    if not files:
        raise RuntimeError(f"OpenAlex manifest {manifest_key} did not contain files")
    yield from _iter_ordered_s3_papers(
        bucket=bucket,
        files=files,
        skip_files=skip_files,
        skip_lines=skip_lines,
        include_expansion=include_expansion,
        max_workers=max_workers,
    )


def iter_s3_manifest_works(
    *,
    bucket: str = "openalex",
    prefix: str = "data/jsonl/works/",
    skip_files: int = 0,
    skip_lines: int = 0,
) -> Iterator[tuple[int, int, dict]]:
    """Stream every snapshot work in manifest order, yielding
    (file_index, line_index, work) so a caller can checkpoint and resume.
    skip_files/skip_lines resume from a stored position. A bounded parallel
    window overlaps object download, decompression, and parsing without
    changing the observable record order."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("Install the runtime dependencies to read OpenAlex S3") from error

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    manifest_key = f"{prefix.rstrip('/')}/manifest.json"
    response = client.get_object(Bucket=bucket, Key=manifest_key)
    try:
        manifest = json.loads(response["Body"].read())
    finally:
        response["Body"].close()
    files = latest_first_manifest_files(manifest.get("files"))
    if not files:
        raise RuntimeError(f"OpenAlex manifest {manifest_key} did not contain files")
    yield from _iter_ordered_s3_files(
        client,
        bucket=bucket,
        files=files,
        skip_files=skip_files,
        skip_lines=skip_lines,
    )


def iter_s3_sampled_works(
    target_records: int,
    *,
    bucket: str = "openalex",
    prefix: str = "data/jsonl/works/",
    seed: int = 20260831,
) -> Iterator[dict]:
    """Sample snapshot files deterministically instead of taking one biased partition."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("Install the runtime dependencies to read OpenAlex S3") from error

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    manifest_key = f"{prefix.rstrip('/')}/manifest.json"
    response = client.get_object(Bucket=bucket, Key=manifest_key)
    try:
        manifest = json.loads(response["Body"].read())
    finally:
        response["Body"].close()
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"OpenAlex manifest {manifest_key} did not contain files")

    file_count = min(len(files), max(64, min(512, math.ceil(target_records / 100_000))))
    rng = random.Random(seed)
    # Weighted sampling prevents small tail part-files from being overrepresented.
    weighted: list[tuple[float, dict]] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        metadata = item.get("meta")
        record_count = metadata.get("record_count", 1) if isinstance(metadata, dict) else 1
        weight = max(1, int(record_count or 1))
        weighted.append((rng.random() ** (1.0 / weight), item))
    selected = [item for _, item in sorted(weighted, reverse=True)[:file_count]]
    # Read extra raw works because untitled and expansion records are rejected.
    raw_per_file = max(1, math.ceil(target_records * 3 / file_count))
    logger.info(
        "snapshot manifest lists %s files; sampling %s files, up to %s raw works each",
        f"{len(files):,}",
        len(selected),
        f"{raw_per_file:,}",
    )
    for position, item in enumerate(selected, start=1):
        url = item.get("url", "") if isinstance(item, dict) else ""
        marker = f"s3://{bucket}/"
        if not url.startswith(marker) or not url.endswith(".gz"):
            continue
        key = url[len(marker) :]
        logger.info("reading snapshot file %d/%d: %s", position, len(selected), key)
        object_response = client.get_object(Bucket=bucket, Key=key)
        body = object_response["Body"]
        try:
            with gzip.GzipFile(fileobj=body, mode="rb") as archive:
                for index, work in enumerate(_iter_json_lines(archive)):
                    if index >= raw_per_file:
                        break
                    yield work
        finally:
            body.close()


def usable_papers(
    works: Iterable[dict],
    limit: int,
    *,
    include_expansion: bool = False,
) -> Iterator[Paper]:
    seen: set[str] = set()
    raw_count = 0
    for raw in works:
        raw_count += 1
        paper = parse_work(raw, include_expansion=include_expansion)
        if paper is None or paper.openalex_id in seen:
            continue
        seen.add(paper.openalex_id)
        yield paper
        if len(seen) >= limit:
            logger.info(
                "usable ratio: %s usable from %s raw works (%.3f)",
                f"{len(seen):,}",
                f"{raw_count:,}",
                len(seen) / raw_count,
            )
            return
