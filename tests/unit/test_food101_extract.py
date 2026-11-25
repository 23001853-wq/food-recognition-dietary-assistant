"""Synthetic HTTP and archive tests for the shared Food-101 Extract stage."""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import tarfile
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest
import yaml

from food_assistant.data_pipeline import extract
from food_assistant.data_pipeline.extract import (
    ExistingArchiveConflict,
    IntegrityError,
    PermanentHTTPError,
    RetryPolicy,
    SourceInfo,
    UnsupportedRangeError,
    _chunk_metadata_path,
    _source_url_digest,
    download_parallel,
    download_sequential,
)

URL = "https://example.invalid/food-101.tar.gz"
ETAG = '"fixture-v1"'
APPROVAL = {"academic_approval": "confirmed_by_user", "archive_url": URL}


def _download_sequential(url, target, **kwargs):
    """Call the network helper with explicit synthetic academic approval."""
    return download_sequential(url, target, approval=APPROVAL, **kwargs)


def _download_parallel(url, target, **kwargs):
    """Call the parallel helper with explicit synthetic academic approval."""
    return download_parallel(url, target, approval=APPROVAL, **kwargs)


class FakeResponse:
    """Small urllib-compatible response with deterministic bytes and headers."""

    def __init__(self, body: bytes, status: int, headers: dict[str, str] | None = None):
        self.body = body
        self.status = status
        self.headers = Message()
        for name, value in (headers or {}).items():
            self.headers[name] = value
        self.offset = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        """Read up to size bytes from the fake body."""
        if size < 0:
            size = len(self.body) - self.offset
        result = self.body[self.offset : self.offset + size]
        self.offset += len(result)
        return result

    def getcode(self) -> int:
        """Return the fake HTTP status code."""
        return self.status

    def close(self) -> None:
        """Mark the fake response closed."""
        self.closed = True


class InterruptedResponse(FakeResponse):
    """Response that raises a transport timeout after one valid body prefix."""

    def __init__(self, body: bytes, status: int, headers: dict[str, str]):
        super().__init__(body, status, headers)
        self.interrupted = False

    def read(self, size: int = -1) -> bytes:
        """Return one prefix, then emulate a connection reset once."""
        if self.offset and not self.interrupted:
            self.interrupted = True
            raise TimeoutError("synthetic timeout")
        return super().read(min(size, max(1, len(self.body) // 3)))


class TimedOutProbeResponse(FakeResponse):
    """Response that times out on its one-byte probe body once."""

    def read(self, size: int = -1) -> bytes:
        """Raise a timeout to exercise retry of a response-body read."""
        raise TimeoutError("synthetic probe timeout")


def _archive_bytes(padding_size: int = 0) -> bytes:
    """Create a readable Food-101-shaped tar.gz wholly in memory."""
    members = {
        "food-101/meta/classes.txt": b"pizza\n",
        "food-101/meta/train.txt": b"pizza/one\n",
        "food-101/meta/test.txt": b"pizza/two\n",
        "food-101/license_agreement.txt": b"synthetic terms",
        "food-101/images/pizza/one.jpg": b"synthetic image bytes",
        "food-101/padding.bin": b"x" * padding_size,
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


def _probe_response(payload: bytes, *, etag: str = ETAG) -> FakeResponse:
    """Return the exact one-byte range probe response for payload."""
    return FakeResponse(
        payload[:1],
        206,
        {
            "Content-Range": f"bytes 0-0/{len(payload)}",
            "Content-Length": "1",
            "ETag": etag,
        },
    )


def _range_response(payload: bytes, start: int, end: int, *, etag: str = ETAG) -> FakeResponse:
    """Return an exact requested range response for payload."""
    body = payload[start : end + 1]
    return FakeResponse(
        body,
        206,
        {
            "Content-Range": f"bytes {start}-{end}/{len(payload)}",
            "Content-Length": str(len(body)),
            "ETag": etag,
        },
    )


class ResponseQueue:
    """Provide scripted responses while retaining requests for assertions."""

    def __init__(self, values):
        self.values = list(values)
        self.requests = []

    def __call__(self, request, timeout=60):
        """Return or raise the next scripted HTTP result."""
        self.requests.append(request)
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def _http_error(status: int, retry_after: str | None = None) -> HTTPError:
    """Build an urllib HTTP error for status classification tests."""
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError(URL, status, "fixture", headers, None)


def _partial_evidence(target: Path, payload: bytes, etag: str) -> None:
    """Seed an existing partial with source-bound size and digest evidence."""
    part = target.with_name(f"{target.name}.part")
    body = payload[: len(payload) // 2]
    part.write_bytes(body)
    extract.atomic_write_json(
        part.with_name(f"{part.name}.json"),
        {
            "schema_version": 1,
            "source_url_sha256": _source_url_digest(URL),
            "source_url": URL,
            "expected_total_bytes": len(payload),
            "etag": etag,
            "last_modified": None,
            "committed_bytes": len(body),
            "partial_sha256": hashlib.sha256(body).hexdigest(),
        },
    )


def test_network_transfer_requires_explicit_matching_approval(tmp_path):
    """Block requests before probing when approval is absent or names another URL."""
    opener = ResponseQueue([])
    with pytest.raises(PermissionError, match="explicit approval"):
        download_sequential(URL, tmp_path / "blocked.tar.gz", opener=opener)
    with pytest.raises(PermissionError, match="explicit approval"):
        download_sequential(
            URL,
            tmp_path / "blocked.tar.gz",
            approval={
                "academic_approval": "confirmed_by_user",
                "archive_url": "https://other.invalid",
            },
            opener=opener,
        )
    assert not opener.requests


def test_probe_body_timeout_is_retried_within_attempt_limit():
    """Retry transient failure while reading the one-byte source probe."""
    payload = _archive_bytes()
    opener = ResponseQueue(
        [
            TimedOutProbeResponse(b"x", 206, {"Content-Range": f"bytes 0-0/{len(payload)}"}),
            _probe_response(payload),
        ]
    )
    delays = []

    info = extract.probe_source(
        URL,
        RetryPolicy(max_attempts=2, base_delay_seconds=0.01, jitter_fraction=0),
        approval=APPROVAL,
        opener=opener,
        sleeper=delays.append,
        random_value=lambda: 0,
    )

    assert info.total_bytes == len(payload)
    assert len(opener.requests) == 2
    assert delays == [0.01]


def test_sequential_retries_500_and_timeout_then_resumes_exact_range(tmp_path):
    """Retry transient HTTP/transport failures and persist an integrity-checked prefix."""
    payload = _archive_bytes(2 * 1024 * 1024)
    target = tmp_path / "food101.tar.gz"
    interrupted = InterruptedResponse(
        payload,
        206,
        {
            "Content-Range": f"bytes 0-{len(payload) - 1}/{len(payload)}",
            "Content-Length": str(len(payload)),
            "ETag": ETAG,
        },
    )
    opener = ResponseQueue(
        [
            _probe_response(payload),
            _http_error(500),
            interrupted,
            _range_response(payload, len(payload) // 3, len(payload) - 1),
        ]
    )
    delays = []
    result = _download_sequential(
        URL,
        target,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.01, jitter_fraction=0),
        opener=opener,
        sleeper=delays.append,
    )

    assert result["state"] == "PASS"
    assert target.read_bytes() == payload
    assert [request.headers.get("Range") for request in opener.requests] == [
        "bytes=0-0",
        f"bytes=0-{len(payload) - 1}",
        f"bytes=0-{len(payload) - 1}",
        f"bytes={len(payload) // 3}-{len(payload) - 1}",
    ]
    assert delays == [0.01, 0.01]
    assert result["official_checksum_verified"] is False
    assert result["local_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["tar_evidence"]["readable"] is True
    assert not target.with_name(f"{target.name}.part").exists()
    assert not target.with_name(f"{target.name}.part.json").exists()


@pytest.mark.parametrize("status", [404, 401, 403])
def test_permanent_http_errors_are_not_retried(tmp_path, status):
    """Do not repeat missing/unauthorized archive requests."""
    opener = ResponseQueue([_http_error(status), _probe_response(b"unused")])
    with pytest.raises(PermanentHTTPError, match=f"HTTP {status}"):
        _download_sequential(URL, tmp_path / "food101.tar.gz", opener=opener)
    assert len(opener.requests) == 1


def test_range_ignored_returns_full_body_without_appending_partial(tmp_path):
    """A resumed request answered with 200 must switch to scratch full download."""
    payload = _archive_bytes()
    target = tmp_path / "food101.tar.gz"
    _partial_evidence(target, payload, ETAG)
    original_partial = target.with_name(f"{target.name}.part").read_bytes()
    opener = ResponseQueue(
        [
            _probe_response(payload),
            FakeResponse(payload, 200, {"Content-Length": str(len(payload)), "ETag": ETAG}),
            FakeResponse(payload, 200, {"Content-Length": str(len(payload)), "ETag": ETAG}),
        ]
    )

    result = _download_sequential(URL, target, opener=opener)

    assert result["state"] == "PASS"
    assert target.read_bytes() == payload
    assert target.with_name(f"{target.name}.part").read_bytes() == original_partial


def test_http_416_falls_back_to_fresh_scratch_without_touching_partial(tmp_path):
    """Recover from an invalidated resume offset using a separate full-response scratch file."""
    payload = _archive_bytes()
    target = tmp_path / "food101.tar.gz"
    _partial_evidence(target, payload, ETAG)
    part = target.with_name(f"{target.name}.part")
    previous = part.read_bytes()
    opener = ResponseQueue(
        [
            _probe_response(payload),
            _http_error(416),
            _probe_response(payload),
            FakeResponse(payload, 200, {"Content-Length": str(len(payload)), "ETag": ETAG}),
        ]
    )

    result = _download_sequential(URL, target, opener=opener)

    assert result["state"] == "PASS"
    assert target.read_bytes() == payload
    assert part.read_bytes() == previous


def test_etag_change_on_resume_response_restarts_from_new_representation(tmp_path):
    """Use a full fresh transfer after If-Range reveals a new ETag; retain the old partial."""
    old_payload = _archive_bytes()
    new_payload = _archive_bytes(256)
    target = tmp_path / "food101.tar.gz"
    _partial_evidence(target, old_payload, ETAG)
    old_part = target.with_name(f"{target.name}.part").read_bytes()
    opener = ResponseQueue(
        [
            _probe_response(old_payload, etag=ETAG),
            FakeResponse(
                new_payload,
                200,
                {"Content-Length": str(len(new_payload)), "ETag": '"fixture-v2"'},
            ),
            FakeResponse(
                new_payload,
                200,
                {"Content-Length": str(len(new_payload)), "ETag": '"fixture-v2"'},
            ),
        ]
    )

    result = _download_sequential(URL, target, opener=opener)

    assert result["state"] == "PASS"
    assert target.read_bytes() == new_payload
    assert result["etag"] == '"fixture-v2"'
    assert target.with_name(f"{target.name}.part").read_bytes() == old_part


def test_invalid_content_range_is_rejected_without_publishing(tmp_path):
    """Reject a malformed response before accepting any bytes or creating an archive."""
    payload = _archive_bytes()
    malformed = FakeResponse(
        payload,
        206,
        {
            "Content-Range": f"bytes 1-{len(payload) - 1}/{len(payload)}",
            "Content-Length": str(len(payload)),
            "ETag": ETAG,
        },
    )
    opener = ResponseQueue([_probe_response(payload), malformed])
    with pytest.raises(UnsupportedRangeError, match="Content-Range"):
        _download_sequential(URL, tmp_path / "food101.tar.gz", opener=opener)
    assert not (tmp_path / "food101.tar.gz").exists()
    assert not list(tmp_path.glob("*.pending"))


@pytest.mark.parametrize("end_adjustment,total_adjustment", [(-1, 0), (0, 1)])
def test_wrong_range_end_or_total_is_rejected(tmp_path, end_adjustment, total_adjustment):
    """Reject a 206 whose end or total differs from the exact requested range."""
    payload = _archive_bytes()
    wrong_total = len(payload) + total_adjustment
    body = payload[: len(payload) + end_adjustment]
    response = FakeResponse(
        body,
        206,
        {
            "Content-Range": f"bytes 0-{len(body) - 1}/{wrong_total}",
            "Content-Length": str(len(body)),
            "ETag": ETAG,
        },
    )
    opener = ResponseQueue([_probe_response(payload), response])

    with pytest.raises(UnsupportedRangeError, match="Content-Range"):
        _download_sequential(URL, tmp_path / "wrong-range.tar.gz", opener=opener)

    assert not (tmp_path / "wrong-range.tar.gz").exists()


def test_changed_etag_invalidates_old_partial_and_restarts_from_zero(tmp_path):
    """Keep stale partial evidence untouched and fetch a new source representation."""
    old_payload = _archive_bytes()
    new_payload = _archive_bytes(128)
    target = tmp_path / "food101.tar.gz"
    _partial_evidence(target, old_payload, '"old"')
    old_part = target.with_name(f"{target.name}.part").read_bytes()
    opener = ResponseQueue(
        [
            _probe_response(new_payload, etag='"new"'),
            _range_response(new_payload, 0, len(new_payload) - 1, etag='"new"'),
        ]
    )

    result = _download_sequential(URL, target, opener=opener)

    assert result["state"] == "PASS"
    assert target.read_bytes() == new_payload
    assert target.with_name(f"{target.name}.part").read_bytes() == old_part
    assert opener.requests[-1].headers.get("Range") == f"bytes=0-{len(new_payload) - 1}"


def test_same_size_corrupt_parallel_chunk_is_refetched(tmp_path):
    """Require a recomputed chunk digest before reusing a same-size file."""
    payload = b"0123456789"
    info = SourceInfo(len(payload), ETAG, None, True)
    chunk = tmp_path / "000000.part"
    chunk.write_bytes(b"XXXXXXXXXX")
    extract.atomic_write_json(
        _chunk_metadata_path(chunk),
        {
            "schema_version": 1,
            "source_url_sha256": _source_url_digest(URL),
            "expected_total_bytes": len(payload),
            "range_start": 0,
            "range_end": len(payload) - 1,
            "expected_bytes": len(payload),
            "etag": ETAG,
            "last_modified": None,
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    opener = ResponseQueue([_range_response(payload, 0, len(payload) - 1)])

    fetched = extract._fetch_chunk(
        URL,
        chunk,
        0,
        len(payload) - 1,
        info,
        retry_policy=RetryPolicy(max_attempts=1),
        opener=opener,
        sleeper=lambda _: None,
        random_value=lambda: 0,
        timeout_seconds=5,
    )

    assert fetched.read_bytes() == payload
    assert len(opener.requests) == 1


def test_verified_parallel_chunk_is_reused_without_http(tmp_path):
    """Reuse a completed chunk only when sidecar boundaries and content hash match."""
    payload = b"a verified chunk"
    info = SourceInfo(len(payload), ETAG, None, True)
    chunk = tmp_path / "000000.part"
    chunk.write_bytes(payload)
    extract.atomic_write_json(
        _chunk_metadata_path(chunk),
        {
            "schema_version": 1,
            "source_url_sha256": _source_url_digest(URL),
            "expected_total_bytes": len(payload),
            "range_start": 0,
            "range_end": len(payload) - 1,
            "expected_bytes": len(payload),
            "etag": ETAG,
            "last_modified": None,
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    opener = ResponseQueue([])

    fetched = extract._fetch_chunk(
        URL,
        chunk,
        0,
        len(payload) - 1,
        info,
        retry_policy=RetryPolicy(max_attempts=1),
        opener=opener,
        sleeper=lambda _: None,
        random_value=lambda: 0,
        timeout_seconds=5,
    )

    assert fetched == chunk
    assert fetched.read_bytes() == payload
    assert not opener.requests


def test_failed_archive_digest_or_tar_does_not_publish_target(tmp_path):
    """Keep bad complete bytes in scratch and never expose them as a published archive."""
    payload = _archive_bytes()
    target = tmp_path / "food101.tar.gz"
    opener = ResponseQueue(
        [_probe_response(payload), _range_response(payload, 0, len(payload) - 1)]
    )
    with pytest.raises(IntegrityError, match="Official SHA-256"):
        _download_sequential(URL, target, official_sha256="0" * 64, opener=opener)
    assert not target.exists()
    assert not extract._archive_metadata_path(target).exists()
    assert not list(tmp_path.glob(".*.pending"))


def test_unreadable_tar_does_not_publish_even_with_matching_local_digest(tmp_path):
    """A self-consistent local digest cannot replace tar readability validation."""
    payload = b"not a tar archive"
    target = tmp_path / "bad-tar.tar.gz"
    opener = ResponseQueue(
        [_probe_response(payload), _range_response(payload, 0, len(payload) - 1)]
    )

    with pytest.raises(IntegrityError, match="readable tar"):
        _download_sequential(
            URL,
            target,
            official_sha256=hashlib.sha256(payload).hexdigest(),
            opener=opener,
        )

    assert not target.exists()
    assert not extract._archive_metadata_path(target).exists()


def test_existing_archive_without_provenance_is_a_conflict(tmp_path):
    """Do not infer source verification from a pre-existing archive's length or tar format."""
    target = tmp_path / "food101.tar.gz"
    target.write_bytes(_archive_bytes())
    opener = ResponseQueue([])
    with pytest.raises(ExistingArchiveConflict, match="lacks complete source-bound"):
        _download_sequential(URL, target, opener=opener)
    assert not opener.requests


def test_verified_existing_archive_is_skipped_without_http(tmp_path):
    """Validate an existing archive digest and provenance before idempotent skip."""
    payload = _archive_bytes()
    target = tmp_path / "food101.tar.gz"
    target.write_bytes(payload)
    metadata = {
        "schema_version": 1,
        "source_url_sha256": _source_url_digest(URL),
        "complete": True,
        "expected_total_bytes": len(payload),
        "local_sha256": hashlib.sha256(payload).hexdigest(),
        "official_sha256": None,
        "official_checksum_verified": False,
    }
    metadata["evidence_fingerprint"] = extract.checkpoint_digest(metadata)
    extract.atomic_write_json(extract._archive_metadata_path(target), metadata)
    opener = ResponseQueue([])

    result = _download_sequential(URL, target, opener=opener)

    assert result["state"] == "SKIPPED"
    assert not opener.requests


def test_retry_after_is_honored_within_configured_cap():
    """Respect Retry-After while keeping waits bounded by max_delay_seconds."""
    error = _http_error(429, retry_after="90")
    opener = ResponseQueue([error, FakeResponse(b"x", 200, {"Content-Length": "1"})])
    delays = []
    response = extract._open_with_retry(
        URL,
        {},
        RetryPolicy(max_attempts=2, base_delay_seconds=0.1, max_delay_seconds=2, jitter_fraction=0),
        opener=opener,
        sleeper=delays.append,
        random_value=lambda: 0,
    )
    assert response.status == 200
    assert delays == [2]


def test_parallel_combines_exact_chunks_and_reuses_verified_chunks(tmp_path):
    """Publish an archive only after all source-bound chunks verify and concatenate."""
    payload = _archive_bytes(64)
    target = tmp_path / "food101.tar.gz"
    chunk_size = max(1, len(payload) // 3)
    ranges = [
        (start, min(len(payload) - 1, start + chunk_size - 1))
        for start in range(0, len(payload), chunk_size)
    ]
    opener = ResponseQueue(
        [_probe_response(payload)] + [_range_response(payload, start, end) for start, end in ranges]
    )
    result = _download_parallel(
        URL,
        target,
        chunk_size=chunk_size,
        workers=2,
        retry_policy=RetryPolicy(max_attempts=1),
        opener=opener,
        sleeper=lambda _: None,
        random_value=lambda: 0,
    )
    assert result["state"] == "PASS"
    assert target.read_bytes() == payload
    assert len(list((tmp_path / "food101.tar.gz.chunks").glob("*.part"))) == len(ranges)


def test_parallel_rejects_server_that_ignores_range(tmp_path):
    """Do not accept full-body HTTP 200 responses as requested parallel chunks."""
    payload = _archive_bytes()
    opener = ResponseQueue(
        [
            _probe_response(payload),
            FakeResponse(payload, 200, {"Content-Length": str(len(payload)), "ETag": ETAG}),
        ]
    )
    with pytest.raises(UnsupportedRangeError, match="ignored the Range"):
        _download_parallel(
            URL,
            tmp_path / "food101.tar.gz",
            chunk_size=len(payload),
            retry_policy=RetryPolicy(max_attempts=1),
            opener=opener,
        )
    assert not (tmp_path / "food101.tar.gz").exists()


def test_combine_failure_never_publishes_archive(tmp_path, monkeypatch):
    """Clean the assembled scratch file and leave no complete archive after combine fails."""
    payload = _archive_bytes(64)
    target = tmp_path / "combine-failure.tar.gz"
    chunk_size = max(1, len(payload) // 2)
    ranges = [
        (start, min(len(payload) - 1, start + chunk_size - 1))
        for start in range(0, len(payload), chunk_size)
    ]
    opener = ResponseQueue(
        [_probe_response(payload)] + [_range_response(payload, start, end) for start, end in ranges]
    )

    def fail_combine(source, destination, length):
        """Simulate a local I/O failure during the staging archive combine."""
        raise OSError("synthetic combine I/O failure")

    monkeypatch.setattr(extract.shutil, "copyfileobj", fail_combine)
    with pytest.raises(OSError, match="combine I/O failure"):
        _download_parallel(
            URL,
            target,
            chunk_size=chunk_size,
            workers=1,
            retry_policy=RetryPolicy(max_attempts=1),
            opener=opener,
            sleeper=lambda _: None,
        )

    assert not target.exists()
    assert not extract._archive_metadata_path(target).exists()
    assert not list(tmp_path.glob(".*.pending"))


def _hold_archive_lock(target: str, ready, release) -> None:
    """Hold the shared archive lock in a separate process for contention testing."""
    with extract.resource_lock(Path(target), timeout_seconds=1):
        ready.set()
        release.wait(10)


def _competing_download(target: str, output) -> None:
    """Attempt a download with zero lock wait; no network opener should be reached."""
    try:
        download_parallel(
            URL,
            Path(target),
            approval=APPROVAL,
            opener=lambda *_args, **_kwargs: pytest.fail("network called"),
        )
    except BaseException as error:
        output.put(type(error).__name__)


def test_two_process_downloaders_share_one_archive_lock(tmp_path):
    """Block a competing process before HTTP access when the common archive lock is held."""
    context = multiprocessing.get_context("spawn")
    target = tmp_path / "food101.tar.gz"
    ready, release = context.Event(), context.Event()
    output = context.Queue()
    holder = context.Process(target=_hold_archive_lock, args=(str(target), ready, release))
    contender = context.Process(target=_competing_download, args=(str(target), output))
    holder.start()
    try:
        assert ready.wait(10)
        contender.start()
        contender.join(10)
        if contender.is_alive():
            contender.terminate()
            contender.join()
        assert output.get(timeout=2) == "RuntimeError"
        assert contender.exitcode == 0
    finally:
        release.set()
        holder.join(10)
        if holder.is_alive():
            holder.terminate()
            holder.join()
    assert holder.exitcode == 0


def test_cli_dry_run_has_no_network_lock_or_filesystem_writes():
    """Keep a default CLI dry run side-effect-free while policy remains disabled."""
    from scripts.download_food101 import run_download_cli

    result = run_download_cli("sequential", ["--dry-run"])

    assert result["state"] == "INCOMPLETE"
    assert result["download_policy"] == "disabled"
    assert result["network_requested"] is False
    assert result["writes_performed"] is False


def _write_temporary_elt_config(tmp_path: Path) -> Path:
    """Copy and relocate the checked-in ELT config into an isolated test workspace."""
    source = Path.cwd() / "configs" / "food101_elt.yaml"
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    for name, value in document["paths"].items():
        document["paths"][name] = str((tmp_path / value).resolve())
    document["approval_file"] = str((tmp_path / "approval.json").resolve())
    document["class_selection_file"] = str((tmp_path / "classes_poc.yaml").resolve())
    document["fixtures"]["root"] = str((tmp_path / "fixtures").resolve())
    config_path = tmp_path / "configs" / "food101_elt.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return config_path


def test_cli_disabled_policy_blocks_network_before_any_writer(tmp_path, monkeypatch):
    """Require an explicit approved config policy even with academic approval present."""
    from scripts import download_food101

    config_path = _write_temporary_elt_config(tmp_path)
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(APPROVAL), encoding="utf-8")
    original_loader = download_food101.load_pipeline_config
    monkeypatch.setattr(
        download_food101,
        "load_pipeline_config",
        lambda path, _workspace: original_loader(path, tmp_path),
    )
    monkeypatch.setattr(
        download_food101,
        "download_sequential",
        lambda *_args, **_kwargs: pytest.fail("network function called"),
    )

    with pytest.raises(PermissionError, match="download.policy is disabled"):
        download_food101.run_download_cli(
            "sequential",
            ["--config", str(config_path), "--approval", str(approval_path), "--download"],
        )

    assert not list(tmp_path.rglob("*.lock"))
    assert not (tmp_path / "data/raw/food-101.tar.gz").exists()


def test_verified_existing_raw_skips_download_and_preserves_tree(tmp_path, monkeypatch):
    """A verified existing raw source skips Extract without touching raw files."""
    from scripts import download_food101

    config_path = _write_temporary_elt_config(tmp_path)
    original_loader = download_food101.load_pipeline_config
    monkeypatch.setattr(
        download_food101,
        "load_pipeline_config",
        lambda path, _workspace: original_loader(path, tmp_path),
    )
    raw_root = tmp_path / "data/raw/food-101"
    raw_root.mkdir(parents=True)
    sentinel = raw_root / "immutable.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    before = sentinel.read_bytes()
    monkeypatch.setattr(
        download_food101,
        "verify_existing_raw",
        lambda path: {"state": "SKIPPED", "source_fingerprint": "fixture"},
    )
    monkeypatch.setattr(
        download_food101,
        "download_sequential",
        lambda *_args, **_kwargs: pytest.fail("network function called"),
    )

    result = download_food101.run_download_cli("sequential", ["--config", str(config_path)])

    assert result["state"] == "SKIPPED"
    assert sentinel.read_bytes() == before
    assert list(raw_root.iterdir()) == [sentinel]
