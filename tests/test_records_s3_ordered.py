import gzip
import io
import json
import multiprocessing as mp
import sys
import threading
import time
import types

import pytest

import openalex_semantic_search.records as records
from openalex_semantic_search.records import iter_s3_manifest_works, usable_papers


def _jsonl_gz(*rows: object) -> bytes:
    payload = b"".join(
        json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n" for row in rows
    )
    return gzip.compress(payload)


class _FakeS3Client:
    def __init__(self, manifest: dict, objects: dict[str, bytes]):
        self._manifest = manifest
        self._objects = objects
        self.requested_keys: list[str] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        assert Bucket == "openalex"
        self.requested_keys.append(Key)
        if Key.endswith("manifest.json"):
            payload = json.dumps(self._manifest).encode("utf-8")
        else:
            payload = self._objects[Key]
        return {"Body": io.BytesIO(payload)}


class _TrackedBody(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.was_closed = False

    def close(self) -> None:
        self.was_closed = True
        super().close()


class _OverlappingS3Client:
    """Make file 1 ready first while file 0's get waits for it."""

    def __init__(self, keys: list[str], objects: dict[str, bytes]):
        self._keys = keys
        self._objects = objects
        self._barrier = threading.Barrier(2)
        self._later_ready = threading.Event()
        self.overlapped = False
        self.bodies: dict[str, _TrackedBody] = {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        assert Bucket == "openalex"
        if Key in self._keys[:2]:
            try:
                self._barrier.wait(timeout=2)
                self.overlapped = True
            except threading.BrokenBarrierError as error:
                raise AssertionError("two S3 downloads did not overlap") from error
            if Key == self._keys[0]:
                assert self._later_ready.wait(timeout=2)
            else:
                self._later_ready.set()
        body = _TrackedBody(self._objects[Key])
        self.bodies[Key] = body
        return {"Body": body}


class _FailAfterBytes(io.BytesIO):
    def __init__(self, payload: bytes, fail_after: int):
        super().__init__(payload)
        self._fail_after = fail_after

    def read(self, size: int = -1) -> bytes:
        if self.tell() >= self._fail_after:
            raise BrokenPipeError("injected response-stream failure")
        remaining = self._fail_after - self.tell()
        if size < 0 or size > remaining:
            size = remaining
        return super().read(size)


class _RetryingS3Client:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.calls = 0

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        assert Bucket == "openalex"
        self.calls += 1
        if self.calls == 1:
            body = _FailAfterBytes(self.payload, len(self.payload) * 3 // 4)
        else:
            body = io.BytesIO(self.payload)
        return {"Body": body}


def _fake_parsed_process_worker(
    *,
    bucket: str,
    key: str,
    first_line: int,
    include_expansion: bool,
    messages,
    terminal,
    stop,
    max_attempts: int,
) -> None:
    del bucket, first_line, include_expansion, max_attempts
    file_number = int(key.rsplit("-", 1)[1].split(".", 1)[0])
    time.sleep(0.15 if file_number == 0 else 0.01)
    messages.put([(0, None)])
    terminal.send(records._S3ParsedFileDone(1, 0, 1, 0.01))


def _terminal_only_parsed_process_worker(
    *,
    bucket: str,
    key: str,
    first_line: int,
    include_expansion: bool,
    messages,
    terminal,
    stop,
    max_attempts: int,
) -> None:
    del bucket, key, include_expansion, messages, stop, max_attempts
    terminal.send(records._S3ParsedFileDone(first_line, 0, 1, 0.001))


def _incomplete_success_parsed_process_worker(
    *,
    bucket: str,
    key: str,
    first_line: int,
    include_expansion: bool,
    messages,
    terminal,
    stop,
    max_attempts: int,
) -> None:
    del bucket, key, include_expansion, messages, stop, max_attempts
    # Simulate a queue feeder dropping one promised record. A zero exit code
    # must never make the coordinator accept an incomplete file.
    terminal.send(records._S3ParsedFileDone(first_line + 1, 1, 1, 0.001))


def _crashing_parsed_process_worker(**kwargs) -> None:
    del kwargs
    raise SystemExit(17)


def _install_fake_boto3(monkeypatch, client: _FakeS3Client) -> None:
    boto3 = types.ModuleType("boto3")
    boto3.client = lambda *args, **kwargs: client
    botocore = types.ModuleType("botocore")
    botocore.UNSIGNED = object()
    botocore_client = types.ModuleType("botocore.client")
    botocore_client.Config = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.client", botocore_client)


def _work(work_id: int, title: str) -> dict:
    return {"id": f"https://openalex.org/W{work_id}", "title": title}


def test_manifest_stream_and_first_id_wins_keep_newest_duplicate(monkeypatch):
    newest_key = "data/jsonl/works/updated_date=2026-06-01/part.gz"
    oldest_key = "data/jsonl/works/updated_date=2025-01-01/part.gz"
    client = _FakeS3Client(
        {
            # Deliberately oldest-first: the iterator owns the ordering contract.
            "files": [
                {"url": f"s3://openalex/{oldest_key}"},
                {"url": f"s3://openalex/{newest_key}"},
            ]
        },
        {
            newest_key: _jsonl_gz(_work(1, "new metadata"), _work(2, "new only")),
            oldest_key: _jsonl_gz(_work(1, "stale metadata"), _work(3, "old only")),
        },
    )
    _install_fake_boto3(monkeypatch, client)

    raw_works = (raw for _, _, raw in iter_s3_manifest_works())
    papers = list(usable_papers(raw_works, limit=3))

    assert [(paper.openalex_id, paper.title) for paper in papers] == [
        ("https://openalex.org/W1", "new metadata"),
        ("https://openalex.org/W2", "new only"),
        ("https://openalex.org/W3", "old only"),
    ]
    assert client.requested_keys[0] == "data/jsonl/works/manifest.json"
    assert set(client.requested_keys[1:]) == {newest_key, oldest_key}


def test_parallel_file_readiness_does_not_change_manifest_yield_order():
    keys = [
        "data/jsonl/works/updated_date=2026-06-01/part.gz",
        "data/jsonl/works/updated_date=2026-05-01/part.gz",
        "data/jsonl/works/updated_date=2026-04-01/part.gz",
    ]
    files = [{"url": f"s3://openalex/{key}"} for key in keys]
    client = _OverlappingS3Client(
        keys,
        {
            keys[0]: _jsonl_gz(_work(1, "newest partition")),
            keys[1]: _jsonl_gz(_work(2, "older partition ready first")),
            keys[2]: _jsonl_gz(_work(3, "oldest partition")),
        },
    )

    iterator = records._iter_ordered_s3_files(
        client,
        bucket="openalex",
        files=files,
        skip_files=0,
        skip_lines=0,
        max_workers=2,
    )
    first = next(iterator)

    # The third file is outside the two-file sliding window until file 0 is
    # drained, even though file 1 has already become ready.
    assert keys[2] not in client.bodies
    yielded = [first, *iterator]

    assert client.overlapped
    assert [(file_index, line_index, raw["title"]) for file_index, line_index, raw in yielded] == [
        (0, 0, "newest partition"),
        (1, 0, "older partition ready first"),
        (2, 0, "oldest partition"),
    ]
    assert set(client.bodies) == set(keys)
    assert all(body.was_closed for body in client.bodies.values())


def test_manifest_resume_cursor_skips_valid_rows_in_only_the_start_file(monkeypatch):
    first_key = "data/jsonl/works/updated_date=2026-06-01/part.gz"
    second_key = "data/jsonl/works/updated_date=2026-05-01/part.gz"
    # Blank/malformed/non-object lines are not part of the logical line_index.
    first_payload = gzip.compress(
        b"\n"
        + json.dumps(_work(1, "already committed")).encode("utf-8")
        + b"\nnot-json\n[]\n"
        + json.dumps(_work(2, "resume here")).encode("utf-8")
        + b"\n"
    )
    client = _FakeS3Client(
        {
            "files": [
                {"url": f"s3://openalex/{second_key}"},
                {"url": f"s3://openalex/{first_key}"},
            ]
        },
        {
            first_key: first_payload,
            second_key: _jsonl_gz(_work(3, "next file starts at zero")),
        },
    )
    _install_fake_boto3(monkeypatch, client)

    resumed = list(iter_s3_manifest_works(skip_files=0, skip_lines=1))

    assert [(file_index, line_index, raw["id"]) for file_index, line_index, raw in resumed] == [
        (0, 1, "https://openalex.org/W2"),
        (1, 0, "https://openalex.org/W3"),
    ]


def test_manifest_resume_starts_at_requested_file_without_fetching_earlier_files(monkeypatch):
    keys = [
        "data/jsonl/works/updated_date=2026-06-01/part.gz",
        "data/jsonl/works/updated_date=2026-05-01/part.gz",
        "data/jsonl/works/updated_date=2026-04-01/part.gz",
    ]
    client = _FakeS3Client(
        {"files": [{"url": f"s3://openalex/{key}"} for key in reversed(keys)]},
        {key: _jsonl_gz(_work(index + 1, key)) for index, key in enumerate(keys)},
    )
    _install_fake_boto3(monkeypatch, client)

    resumed = list(iter_s3_manifest_works(skip_files=1, skip_lines=0))

    assert [(file_index, line_index) for file_index, line_index, _ in resumed] == [
        (1, 0),
        (2, 0),
    ]
    assert client.requested_keys[0] == "data/jsonl/works/manifest.json"
    assert set(client.requested_keys[1:]) == {keys[1], keys[2]}


def test_process_parser_preserves_order_when_later_file_finishes_first():
    keys = [f"data/jsonl/works/part-{index}.gz" for index in range(3)]
    files = [{"url": f"s3://openalex/{key}"} for key in keys]
    context = mp.get_context("fork")

    yielded = list(
        records._iter_ordered_s3_papers(
            bucket="openalex",
            files=files,
            skip_files=0,
            skip_lines=0,
            include_expansion=False,
            max_workers=2,
            process_context=context,
            worker_target=_fake_parsed_process_worker,
        )
    )

    assert [(file_index, line_index) for file_index, line_index, _ in yielded] == [
        (0, 0),
        (1, 0),
        (2, 0),
    ]


def test_process_parser_accepts_empty_file_from_synchronous_terminal_pipe():
    files = [{"url": "s3://openalex/data/jsonl/works/empty.gz"}]

    yielded = list(
        records._iter_ordered_s3_papers(
            bucket="openalex",
            files=files,
            skip_files=0,
            skip_lines=0,
            include_expansion=False,
            max_workers=1,
            process_context=mp.get_context("spawn"),
            worker_target=_terminal_only_parsed_process_worker,
        )
    )

    assert yielded == []


def test_process_parser_rejects_zero_exit_when_promised_rows_are_missing():
    files = [{"url": "s3://openalex/data/jsonl/works/incomplete.gz"}]

    with pytest.raises(RuntimeError, match=r"before completing.*exit code 0"):
        list(
            records._iter_ordered_s3_papers(
                bucket="openalex",
                files=files,
                skip_files=0,
                skip_lines=0,
                include_expansion=False,
                max_workers=1,
                process_context=mp.get_context("spawn"),
                worker_target=_incomplete_success_parsed_process_worker,
            )
        )


def test_process_parser_reports_child_exit_instead_of_waiting_forever():
    files = [{"url": "s3://openalex/data/jsonl/works/part-0.gz"}]

    with pytest.raises(RuntimeError, match=r"exit code 17"):
        list(
            records._iter_ordered_s3_papers(
                bucket="openalex",
                files=files,
                skip_files=0,
                skip_lines=0,
                include_expansion=False,
                max_workers=1,
                process_context=mp.get_context("fork"),
                worker_target=_crashing_parsed_process_worker,
            )
        )


def test_process_parser_retries_broken_stream_without_duplicate_rows(monkeypatch):
    rows = [
        _work(index + 1, f"paper-{index:04d}-" + format(index * 2654435761, "064x"))
        for index in range(600)
    ]
    payload = _jsonl_gz(*rows)
    client = _RetryingS3Client(payload)
    monkeypatch.setattr(records, "_unsigned_s3_client", lambda: client)
    messages: records.Queue[object] = records.Queue()
    stop = threading.Event()

    records._stream_s3_file_to_parsed_queue(
        bucket="openalex",
        key="data/jsonl/works/retry.gz",
        first_line=0,
        include_expansion=False,
        messages=messages,
        stop=stop,
        max_attempts=2,
    )

    parsed = []
    while True:
        message = messages.get_nowait()
        if isinstance(message, records._S3ParsedFileDone):
            assert message.attempts == 2
            break
        assert isinstance(message, list)
        parsed.extend(message)
    assert client.calls == 2
    assert [line_index for line_index, _ in parsed] == list(range(600))
    assert all(paper is not None for _, paper in parsed)
