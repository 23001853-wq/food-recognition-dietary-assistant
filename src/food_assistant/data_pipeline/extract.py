"""Verified, resumable archive extraction over one shared HTTP safety contract."""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import math
import os
import random
import re
import shutil
import socket
import tarfile
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from food_assistant.data_pipeline.atomic import atomic_write_json, checkpoint_digest, resource_lock

LOGGER = logging.getLogger(__name__)
CONTENT_RANGE_PATTERN = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
ARCHIVE_METADATA_SCHEMA = 1
PARTIAL_METADATA_SCHEMA = 1
DEFAULT_CHUNK_SIZE = 32 * 1024 * 1024
DEFAULT_WORKERS = 4
IO_CHUNK_SIZE = 1024 * 1024


class ExtractError(RuntimeError):
    """Base error for safe archive acquisition failures."""


class PermanentHTTPError(ExtractError):
    """Raised for an HTTP response that must not be retried."""


class UnsupportedRangeError(ExtractError):
    """Raised when the source cannot provide the requested exact byte range."""


class IntegrityError(ExtractError):
    """Raised when archive bytes, source evidence or tar structure are invalid."""


class ExistingArchiveConflict(ExtractError):
    """Raised when an existing archive cannot be proven to match this source."""


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry and backoff settings for HTTP requests."""

    max_attempts: int = 4
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 30.0
    jitter_fraction: float = 0.2

    def __post_init__(self) -> None:
        """Reject unsafe or unbounded retry settings."""
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("Retry delays must be non-negative")
        if not math.isfinite(self.base_delay_seconds) or not math.isfinite(self.max_delay_seconds):
            raise ValueError("Retry delays must be finite")
        if not 0 <= self.jitter_fraction <= 1:
            raise ValueError("jitter_fraction must be between zero and one")


@dataclass(frozen=True)
class SourceInfo:
    """Remote archive size and representation validators observed by a range probe."""

    total_bytes: int
    etag: str | None
    last_modified: str | None
    range_supported: bool

    @property
    def resume_validator(self) -> tuple[str, str] | None:
        """Return an HTTP If-Range validator, preferring a strong ETag."""
        if self.etag and not self.etag.strip().lower().startswith("w/"):
            return "ETag", self.etag
        if self.last_modified:
            return "Last-Modified", self.last_modified
        return None


def _source_url_digest(url: str) -> str:
    """Hash the complete source URL without persisting possible query credentials."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _require_approval(url: str, approval: Mapping[str, Any] | None) -> None:
    """Require explicit academic approval for the exact source before network access."""
    if (
        not isinstance(approval, Mapping)
        or approval.get("academic_approval") != "confirmed_by_user"
        or approval.get("archive_url") != url
    ):
        raise PermissionError("BLOCKED: explicit approval for this archive URL is required")


def _safe_source_url(url: str) -> str:
    """Return a log/metadata-safe URL without userinfo, query or fragment."""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _parse_content_range(value: str | None) -> tuple[int, int, int]:
    """Parse a strict byte Content-Range header and require a known total."""
    match = CONTENT_RANGE_PATTERN.fullmatch(value or "")
    if not match:
        raise UnsupportedRangeError("Missing or malformed Content-Range")
    start, end, total = (int(part) for part in match.groups())
    if total <= 0 or start > end or end >= total:
        raise UnsupportedRangeError("Invalid Content-Range bounds")
    return start, end, total


def _retry_after_seconds(value: str | None, now: datetime | None = None) -> float | None:
    """Parse Retry-After as bounded seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        current = now or datetime.now(UTC)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        return max(0.0, (retry_at - current).total_seconds())


def _transient_reason(error: BaseException) -> bool:
    """Classify transport failures that are safe to retry."""
    if isinstance(error, HTTPError):
        return error.code in RETRYABLE_HTTP_STATUSES or 500 <= error.code <= 599
    if isinstance(error, URLError):
        return isinstance(
            error.reason,
            (TimeoutError, socket.timeout, ConnectionError, OSError, http.client.HTTPException),
        )
    return isinstance(
        error,
        (
            TimeoutError,
            socket.timeout,
            ConnectionError,
            OSError,
            http.client.HTTPException,
        ),
    )


def _retry_delay(
    attempt: int,
    policy: RetryPolicy,
    retry_after: str | None,
    random_value: Callable[[], float],
) -> float:
    """Calculate capped exponential delay with bounded jitter and Retry-After."""
    base = min(policy.max_delay_seconds, policy.base_delay_seconds * (2**attempt))
    jitter = base * policy.jitter_fraction * random_value()
    requested = _retry_after_seconds(retry_after)
    return min(policy.max_delay_seconds, max(requested or 0.0, base + jitter))


def _open_with_retry(
    url: str,
    headers: dict[str, str],
    policy: RetryPolicy,
    *,
    opener: Callable[..., Any] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
    timeout_seconds: float = 60.0,
) -> Any:
    """Open an HTTP request with finite retries and sanitized error messages."""
    for attempt in range(policy.max_attempts):
        try:
            return opener(Request(url, headers=headers), timeout=timeout_seconds)
        except HTTPError as error:
            if error.code == 416:
                return error
            if not _transient_reason(error) or attempt + 1 >= policy.max_attempts:
                raise PermanentHTTPError(
                    f"HTTP {error.code} while requesting archive (attempt {attempt + 1})"
                ) from None
            delay = _retry_delay(attempt, policy, error.headers.get("Retry-After"), random_value)
            LOGGER.warning(
                "extract retry attempt=%s status=%s delay_seconds=%.3f",
                attempt + 1,
                error.code,
                delay,
            )
            sleeper(delay)
        except (
            URLError,
            TimeoutError,
            ConnectionError,
            OSError,
            http.client.HTTPException,
        ) as error:
            if not _transient_reason(error) or attempt + 1 >= policy.max_attempts:
                raise ExtractError(
                    f"Transient transport failed after {attempt + 1} attempt(s): "
                    f"{type(error).__name__}"
                ) from None
            delay = _retry_delay(attempt, policy, None, random_value)
            LOGGER.warning(
                "extract retry attempt=%s error=%s delay_seconds=%.3f",
                attempt + 1,
                type(error).__name__,
                delay,
            )
            sleeper(delay)
    raise ExtractError("HTTP request exhausted its retry policy")


def probe_source(
    url: str,
    retry_policy: RetryPolicy = RetryPolicy(),
    *,
    approval: Mapping[str, Any] | None = None,
    opener: Callable[..., Any] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
    timeout_seconds: float = 60.0,
) -> SourceInfo:
    """Discover archive length and validators without downloading a full archive."""
    _require_approval(url, approval)
    for attempt in range(retry_policy.max_attempts):
        response = _open_with_retry(
            url,
            {"Range": "bytes=0-0"},
            retry_policy,
            opener=opener,
            sleeper=sleeper,
            random_value=random_value,
            timeout_seconds=timeout_seconds,
        )
        try:
            status = getattr(response, "status", response.getcode())
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
            if status == 206:
                start, end, total = _parse_content_range(response.headers.get("Content-Range"))
                try:
                    probe_body = response.read(2)
                except (
                    URLError,
                    TimeoutError,
                    ConnectionError,
                    OSError,
                    http.client.HTTPException,
                ):
                    if attempt + 1 >= retry_policy.max_attempts:
                        raise ExtractError(
                            f"Range probe timed out after {attempt + 1} attempt(s)"
                        ) from None
                    sleeper(_retry_delay(attempt, retry_policy, None, random_value))
                    continue
                if (start, end) != (0, 0) or len(probe_body) != 1:
                    raise UnsupportedRangeError("Range probe returned an invalid byte response")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and content_length != "1":
                    raise UnsupportedRangeError("Range probe Content-Length must be one")
                return SourceInfo(total, etag, last_modified, True)
            if status == 200:
                content_length = response.headers.get("Content-Length")
                try:
                    total = int(content_length or "")
                except ValueError as exc:
                    raise UnsupportedRangeError(
                        "Source without byte ranges must provide Content-Length"
                    ) from exc
                if total <= 0:
                    raise UnsupportedRangeError("Source Content-Length must be positive")
                return SourceInfo(total, etag, last_modified, False)
            raise UnsupportedRangeError(f"Unexpected HTTP status for range probe: {status}")
        finally:
            response.close()
    raise ExtractError("Source range probe exhausted its retry policy")


def _same_source(metadata: dict[str, Any], url: str, info: SourceInfo) -> bool:
    """Check that persisted partial evidence binds the current source representation."""
    return (
        metadata.get("schema_version") == PARTIAL_METADATA_SCHEMA
        and metadata.get("source_url_sha256") == _source_url_digest(url)
        and metadata.get("expected_total_bytes") == info.total_bytes
        and metadata.get("etag") == info.etag
        and metadata.get("last_modified") == info.last_modified
    )


def _file_sha256(path: Path) -> str:
    """Hash a file incrementally without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(IO_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any] | None:
    """Read optional JSON evidence, returning None for malformed or unavailable files."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _partial_candidates(target: Path) -> list[tuple[Path, Path]]:
    """Find canonical and generation-specific partial files without modifying them."""
    canonical = target.with_name(f"{target.name}.part")
    candidates = [(canonical, canonical.with_name(f"{canonical.name}.json"))]
    candidates.extend(
        (path, path.with_name(f"{path.name}.json"))
        for path in sorted(target.parent.glob(f"{target.name}.resume-*.part"))
    )
    return candidates


def _verified_partial(target: Path, url: str, info: SourceInfo) -> tuple[Path, Path, int] | None:
    """Return a source-bound partial only if size, committed offset and digest all match."""
    if info.resume_validator is None:
        return None
    for part, metadata_path in _partial_candidates(target):
        if not part.is_file():
            continue
        metadata = _read_object(metadata_path)
        if metadata is None or not _same_source(metadata, url, info):
            continue
        try:
            size = part.stat().st_size
        except OSError:
            continue
        if (
            size <= 0
            or size >= info.total_bytes
            or metadata.get("committed_bytes") != size
            or metadata.get("partial_sha256") != _file_sha256(part)
        ):
            continue
        return part, metadata_path, size
    return None


def _partial_metadata(url: str, info: SourceInfo, path: Path) -> dict[str, Any]:
    """Build evidence for a durably committed prefix of the current source."""
    return {
        "schema_version": PARTIAL_METADATA_SCHEMA,
        "source_url_sha256": _source_url_digest(url),
        "source_url": _safe_source_url(url),
        "expected_total_bytes": info.total_bytes,
        "etag": info.etag,
        "last_modified": info.last_modified,
        "committed_bytes": path.stat().st_size,
        "partial_sha256": _file_sha256(path),
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }


def _read_stream(
    response: Any,
    destination: Path,
    *,
    expected_bytes: int,
    append: bool = False,
) -> int:
    """Stream exactly the expected byte count into a new file or append target."""
    mode = "ab" if append else "wb"
    received = 0
    with Path(destination).open(mode) as output:
        while True:
            chunk = response.read(IO_CHUNK_SIZE)
            if not chunk:
                break
            if received + len(chunk) > expected_bytes:
                raise IntegrityError("HTTP response exceeded its expected byte range")
            output.write(chunk)
            received += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    return received


def _request_range(
    url: str,
    start: int,
    end: int,
    info: SourceInfo,
    policy: RetryPolicy,
    *,
    opener: Callable[..., Any],
    sleeper: Callable[[float], None],
    random_value: Callable[[], float],
    timeout_seconds: float,
) -> Any:
    """Open a strict requested range, including If-Range when resuming."""
    headers = {"Range": f"bytes={start}-{end}"}
    if start > 0 and info.resume_validator is not None:
        headers["If-Range"] = info.resume_validator[1]
    return _open_with_retry(
        url,
        headers,
        policy,
        opener=opener,
        sleeper=sleeper,
        random_value=random_value,
        timeout_seconds=timeout_seconds,
    )


def _validate_range_response(response: Any, start: int, end: int, info: SourceInfo) -> None:
    """Require exact HTTP status, byte range, total and available validators."""
    status = getattr(response, "status", response.getcode())
    if status != 206:
        raise UnsupportedRangeError(f"Expected HTTP 206 for range; received {status}")
    actual_start, actual_end, actual_total = _parse_content_range(
        response.headers.get("Content-Range")
    )
    if (actual_start, actual_end, actual_total) != (start, end, info.total_bytes):
        raise UnsupportedRangeError("Content-Range does not match the requested exact range")
    content_length = response.headers.get("Content-Length")
    if content_length is not None and content_length != str(end - start + 1):
        raise UnsupportedRangeError("Range Content-Length does not match requested bounds")
    response_etag = response.headers.get("ETag")
    response_modified = response.headers.get("Last-Modified")
    if info.etag and response_etag and info.etag != response_etag:
        raise IntegrityError("Source ETag changed between probe and range response")
    if info.last_modified and response_modified and info.last_modified != response_modified:
        raise IntegrityError("Source Last-Modified changed between probe and range response")


def _commit_partial(
    destination: Path,
    metadata_path: Path,
    url: str,
    info: SourceInfo,
    temp_body: Path,
    *,
    base_path: Path | None,
) -> None:
    """Atomically commit only a validated prefix and update its digest evidence."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".pending", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            if base_path is not None:
                with base_path.open("rb") as base:
                    shutil.copyfileobj(base, output, IO_CHUNK_SIZE)
            with temp_body.open("rb") as body:
                shutil.copyfileobj(body, output, IO_CHUNK_SIZE)
            output.flush()
            os.fsync(output.fileno())
        if temporary.stat().st_size >= info.total_bytes:
            raise IntegrityError("Partial checkpoint reached or exceeded total archive length")
        os.replace(temporary, destination)
        atomic_write_json(metadata_path, _partial_metadata(url, info, destination))
    finally:
        temporary.unlink(missing_ok=True)


def _source_matches_response(info: SourceInfo, response: Any) -> bool:
    """Check returned representation validators against the initial source probe."""
    etag, modified = response.headers.get("ETag"), response.headers.get("Last-Modified")
    return not (
        (info.etag and etag and info.etag != etag)
        or (info.last_modified and modified and info.last_modified != modified)
    )


def _full_response_source_info(response: Any, previous: SourceInfo) -> SourceInfo:
    """Bind a full-body HTTP 200 response to its observed length and validators."""
    if getattr(response, "status", response.getcode()) != 200:
        raise UnsupportedRangeError("Expected HTTP 200 for a full archive response")
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        if not _source_matches_response(previous, response):
            raise IntegrityError("Changed source response has no Content-Length evidence")
        total = previous.total_bytes
    else:
        try:
            total = int(content_length)
        except ValueError as error:
            raise IntegrityError("Full response Content-Length is invalid") from error
    if total <= 0:
        raise IntegrityError("Full response Content-Length must be positive")
    source_changed = not _source_matches_response(previous, response)
    if total != previous.total_bytes and not source_changed:
        raise IntegrityError("Full response size changed without a source validator change")
    return SourceInfo(
        total,
        response.headers.get("ETag") or previous.etag,
        response.headers.get("Last-Modified") or previous.last_modified,
        False,
    )


def _stream_full_fresh(
    url: str,
    info: SourceInfo,
    destination: Path,
    *,
    policy: RetryPolicy,
    opener: Callable[..., Any],
    sleeper: Callable[[float], None],
    random_value: Callable[[], float],
    timeout_seconds: float,
    ranged: bool,
) -> None:
    """Download a fresh representation to scratch without touching existing partials."""
    headers = {"Range": f"bytes=0-{info.total_bytes - 1}"} if ranged else {}
    for attempt in range(policy.max_attempts):
        response = _open_with_retry(
            url,
            headers,
            policy,
            opener=opener,
            sleeper=sleeper,
            random_value=random_value,
            timeout_seconds=timeout_seconds,
        )
        try:
            status = getattr(response, "status", response.getcode())
            if status == 206 and ranged:
                _validate_range_response(response, 0, info.total_bytes - 1, info)
            elif status == 200 and not ranged:
                content_length = response.headers.get("Content-Length")
                if content_length is not None and content_length != str(info.total_bytes):
                    raise IntegrityError("Full response Content-Length differs from probed total")
                if not _source_matches_response(info, response):
                    raise IntegrityError("Source validator changed during full response")
            else:
                raise UnsupportedRangeError(f"Unexpected HTTP {status} for fresh archive transfer")
            received = _read_stream(response, destination, expected_bytes=info.total_bytes)
            if received != info.total_bytes:
                raise http.client.IncompleteRead(b"", info.total_bytes - received)
            return
        except (
            URLError,
            TimeoutError,
            ConnectionError,
            OSError,
            http.client.HTTPException,
        ) as error:
            destination.unlink(missing_ok=True)
            if attempt + 1 >= policy.max_attempts:
                raise ExtractError(
                    f"Full archive stream failed after {attempt + 1} attempt(s): "
                    f"{type(error).__name__}"
                ) from None
            sleeper(_retry_delay(attempt, policy, None, random_value))
        finally:
            response.close()
    raise ExtractError("Full archive transfer exhausted its retry policy")


def _fetch_sequential_part(
    url: str,
    target: Path,
    info: SourceInfo,
    *,
    approval: Mapping[str, Any],
    retry_policy: RetryPolicy,
    opener: Callable[..., Any],
    sleeper: Callable[[float], None],
    random_value: Callable[[], float],
    timeout_seconds: float,
) -> tuple[Path, SourceInfo]:
    """Fetch a full archive, resuming only from a hash-verified source-bound prefix."""
    valid_partial = _verified_partial(target, url, info)
    if valid_partial is None:
        canonical = target.with_name(f"{target.name}.part")
        canonical_meta = canonical.with_name(f"{canonical.name}.json")
        legacy_exists = canonical.exists() and _read_object(canonical_meta) is None
        generation = uuid.uuid4().hex
        if legacy_exists or any(part.exists() for part, _ in _partial_candidates(target)):
            part = target.with_name(f"{target.name}.resume-{generation}.part")
            metadata_path = part.with_name(f"{part.name}.json")
        else:
            part, metadata_path = canonical, canonical_meta
        base_path = None
        offset = 0
    else:
        part, metadata_path, offset = valid_partial
        base_path = part

    if not info.range_supported:
        scratch = target.with_name(f".{target.name}.{uuid.uuid4().hex}.pending")
        try:
            _stream_full_fresh(
                url,
                info,
                scratch,
                policy=retry_policy,
                opener=opener,
                sleeper=sleeper,
                random_value=random_value,
                timeout_seconds=timeout_seconds,
                ranged=False,
            )
            return scratch, info
        except BaseException:
            scratch.unlink(missing_ok=True)
            raise

    if offset == 0:
        atomic_write_json(
            metadata_path,
            {
                "schema_version": PARTIAL_METADATA_SCHEMA,
                "source_url_sha256": _source_url_digest(url),
                "source_url": _safe_source_url(url),
                "expected_total_bytes": info.total_bytes,
                "etag": info.etag,
                "last_modified": info.last_modified,
                "committed_bytes": 0,
                "partial_sha256": hashlib.sha256(b"").hexdigest(),
                "updated_at_utc": datetime.now(UTC).isoformat(),
            },
        )

    for attempt in range(retry_policy.max_attempts):
        valid_partial = _verified_partial(target, url, info)
        if valid_partial is not None:
            part, metadata_path, offset = valid_partial
            base_path = part
        start = offset
        end = info.total_bytes - 1
        tail = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tail")
        try:
            response = _request_range(
                url,
                start,
                end,
                info,
                retry_policy,
                opener=opener,
                sleeper=sleeper,
                random_value=random_value,
                timeout_seconds=timeout_seconds,
            )
            status = getattr(response, "status", response.getcode())
            if status == 200:
                refreshed_info = _full_response_source_info(response, info)
                response.close()
                fresh = target.with_name(f".{target.name}.{uuid.uuid4().hex}.pending")
                try:
                    _stream_full_fresh(
                        url,
                        refreshed_info,
                        fresh,
                        policy=retry_policy,
                        opener=opener,
                        sleeper=sleeper,
                        random_value=random_value,
                        timeout_seconds=timeout_seconds,
                        ranged=False,
                    )
                except BaseException:
                    fresh.unlink(missing_ok=True)
                    raise
                return fresh, refreshed_info
            if status == 416:
                response.close()
                refreshed_info = probe_source(
                    url,
                    retry_policy,
                    approval=approval,
                    opener=opener,
                    sleeper=sleeper,
                    random_value=random_value,
                    timeout_seconds=timeout_seconds,
                )
                fresh = target.with_name(f".{target.name}.{uuid.uuid4().hex}.pending")
                try:
                    _stream_full_fresh(
                        url,
                        refreshed_info,
                        fresh,
                        policy=retry_policy,
                        opener=opener,
                        sleeper=sleeper,
                        random_value=random_value,
                        timeout_seconds=timeout_seconds,
                        ranged=False,
                    )
                except BaseException:
                    fresh.unlink(missing_ok=True)
                    raise
                return fresh, refreshed_info
            if not _source_matches_response(info, response):
                response.close()
                refreshed_info = probe_source(
                    url,
                    retry_policy,
                    approval=approval,
                    opener=opener,
                    sleeper=sleeper,
                    random_value=random_value,
                    timeout_seconds=timeout_seconds,
                )
                fresh = target.with_name(f".{target.name}.{uuid.uuid4().hex}.pending")
                try:
                    _stream_full_fresh(
                        url,
                        refreshed_info,
                        fresh,
                        policy=retry_policy,
                        opener=opener,
                        sleeper=sleeper,
                        random_value=random_value,
                        timeout_seconds=timeout_seconds,
                        ranged=False,
                    )
                except BaseException:
                    fresh.unlink(missing_ok=True)
                    raise
                return fresh, refreshed_info
            _validate_range_response(response, start, end, info)
            try:
                received = _read_stream(
                    response,
                    tail,
                    expected_bytes=info.total_bytes - start,
                )
            except (URLError, TimeoutError, ConnectionError, OSError, http.client.HTTPException):
                response.close()
                if tail.exists() and tail.stat().st_size:
                    _commit_partial(
                        part,
                        metadata_path,
                        url,
                        info,
                        tail,
                        base_path=base_path,
                    )
                if attempt + 1 >= retry_policy.max_attempts:
                    raise ExtractError(
                        f"Archive stream interrupted after {attempt + 1} attempt(s)"
                    ) from None
                delay = _retry_delay(attempt, retry_policy, None, random_value)
                sleeper(delay)
                continue
            finally:
                response.close()
            if received < info.total_bytes - start:
                if tail.stat().st_size:
                    _commit_partial(
                        part,
                        metadata_path,
                        url,
                        info,
                        tail,
                        base_path=base_path,
                    )
                if attempt + 1 >= retry_policy.max_attempts:
                    raise ExtractError("Range response ended before the exact requested byte count")
                sleeper(_retry_delay(attempt, retry_policy, None, random_value))
                continue
            if received != info.total_bytes - start:
                raise IntegrityError("Range response exceeded the exact requested byte count")
            descriptor, assembled_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".pending", dir=target.parent
            )
            assembled = Path(assembled_name)
            try:
                with os.fdopen(descriptor, "wb") as output:
                    if base_path is not None:
                        with base_path.open("rb") as base:
                            shutil.copyfileobj(base, output, IO_CHUNK_SIZE)
                    with tail.open("rb") as body:
                        shutil.copyfileobj(body, output, IO_CHUNK_SIZE)
                    output.flush()
                    os.fsync(output.fileno())
                if assembled.stat().st_size != info.total_bytes:
                    raise IntegrityError("Resumed archive length differs from source total")
                return assembled, info
            except BaseException:
                assembled.unlink(missing_ok=True)
                raise
        finally:
            tail.unlink(missing_ok=True)
    raise ExtractError("Sequential archive transfer exhausted its retry policy")


def _verify_tar(path: Path) -> dict[str, Any]:
    """Validate tar readability and required Food-101 metadata members."""
    required_members = {
        "food-101/meta/classes.txt",
        "food-101/meta/train.txt",
        "food-101/meta/test.txt",
        "food-101/license_agreement.txt",
    }
    try:
        with tarfile.open(path, mode="r:*") as archive:
            names = {member.name.rstrip("/") for member in archive.getmembers()}
    except (OSError, tarfile.TarError, EOFError) as error:
        raise IntegrityError(
            f"Archive is not a readable tar file: {type(error).__name__}"
        ) from None
    missing = sorted(required_members - names)
    if missing:
        raise IntegrityError(f"Archive is missing required Food-101 members: {missing}")
    return {"readable": True, "required_members_present": True}


def _archive_metadata_path(target: Path) -> Path:
    """Return the sidecar path used for complete archive provenance."""
    return target.with_name(f"{target.name}.metadata.json")


def _validate_complete_archive(
    target: Path, url: str, official_sha256: str | None
) -> dict[str, Any] | None:
    """Validate an existing archive and its sidecar or fail closed on ambiguity."""
    if not target.exists():
        return None
    metadata_path = _archive_metadata_path(target)
    metadata = _read_object(metadata_path)
    if (
        metadata is None
        or metadata.get("schema_version") != ARCHIVE_METADATA_SCHEMA
        or metadata.get("complete") is not True
        or metadata.get("source_url_sha256") != _source_url_digest(url)
    ):
        raise ExistingArchiveConflict(
            "Existing archive lacks complete source-bound metadata; refusing overwrite"
        )
    try:
        actual_size = target.stat().st_size
        digest = _file_sha256(target)
        tar_evidence = _verify_tar(target)
    except OSError as error:
        raise ExistingArchiveConflict(
            f"Existing archive cannot be verified: {type(error).__name__}"
        ) from None
    if (
        actual_size != metadata.get("expected_total_bytes")
        or digest != metadata.get("local_sha256")
        or (official_sha256 is not None and digest.lower() != official_sha256.lower())
        or metadata.get("official_sha256") != official_sha256
        or metadata.get("official_checksum_verified") is not (official_sha256 is not None)
        or metadata.get("evidence_fingerprint")
        != checkpoint_digest(
            {key: value for key, value in metadata.items() if key != "evidence_fingerprint"}
        )
    ):
        raise ExistingArchiveConflict("Existing archive checksum or source metadata conflicts")
    return {**metadata, "tar_evidence": tar_evidence, "state": "SKIPPED"}


def _publish_archive(
    assembled: Path,
    target: Path,
    url: str,
    info: SourceInfo,
    official_sha256: str | None,
    run_id: str,
) -> dict[str, Any]:
    """Verify complete bytes and tar evidence before publishing archive and metadata."""
    if assembled.stat().st_size != info.total_bytes:
        raise IntegrityError("Archive total length does not match source probe")
    local_sha256 = _file_sha256(assembled)
    if official_sha256 is not None and local_sha256.lower() != official_sha256.lower():
        raise IntegrityError("Official SHA-256 checksum mismatch")
    tar_evidence = _verify_tar(assembled)
    if target.exists():
        raise ExistingArchiveConflict(f"Archive target appeared during transfer: {target.name}")
    os.replace(assembled, target)
    metadata = {
        "schema_version": ARCHIVE_METADATA_SCHEMA,
        "source": "Food-101 official release",
        "source_url": _safe_source_url(url),
        "source_url_sha256": _source_url_digest(url),
        "etag": info.etag,
        "last_modified": info.last_modified,
        "expected_total_bytes": info.total_bytes,
        "local_sha256": local_sha256,
        "official_sha256": official_sha256,
        "official_checksum_verified": official_sha256 is not None,
        "tar_evidence": tar_evidence,
        "completion_state": "PASS",
        "complete": True,
        "run_id": run_id,
        "verified_at_utc": datetime.now(UTC).isoformat(),
    }
    metadata["evidence_fingerprint"] = checkpoint_digest(metadata)
    atomic_write_json(_archive_metadata_path(target), metadata)
    return {**metadata, "state": "PASS", "archive": str(target)}


def download_sequential(
    url: str,
    target: Path,
    *,
    approval: Mapping[str, Any] | None = None,
    official_sha256: str | None = None,
    retry_policy: RetryPolicy = RetryPolicy(),
    lock_timeout_seconds: float = 0,
    opener: Callable[..., Any] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Download sequentially with validated Range resume and immutable publication."""
    _require_approval(url, approval)
    if official_sha256 is not None and (
        len(official_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in official_sha256)
    ):
        raise ValueError("official_sha256 must be null or a 64-digit SHA-256")
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    LOGGER.info("extract run_id=%s mode=sequential event=start target=%s", run_id, target.name)
    with resource_lock(target, lock_timeout_seconds):
        existing = _validate_complete_archive(target, url, official_sha256)
        if existing is not None:
            return existing
        info = probe_source(
            url,
            retry_policy,
            approval=approval,
            opener=opener,
            sleeper=sleeper,
            random_value=random_value,
            timeout_seconds=timeout_seconds,
        )
        assembled, final_info = _fetch_sequential_part(
            url,
            target,
            info,
            approval=approval,
            retry_policy=retry_policy,
            opener=opener,
            sleeper=sleeper,
            random_value=random_value,
            timeout_seconds=timeout_seconds,
        )
        try:
            result = _publish_archive(assembled, target, url, final_info, official_sha256, run_id)
            LOGGER.info("extract run_id=%s mode=sequential event=%s", run_id, result["state"])
            return result
        finally:
            assembled.unlink(missing_ok=True)


def _chunk_metadata_path(chunk_path: Path) -> Path:
    """Return the sidecar path for one verified parallel chunk."""
    return chunk_path.with_name(f"{chunk_path.name}.json")


def _matching_chunk(
    chunk_path: Path,
    url: str,
    info: SourceInfo,
    start: int,
    end: int,
) -> bool:
    """Require exact boundaries, source validators, expected size and content digest."""
    if info.resume_validator is None:
        return False
    metadata = _read_object(_chunk_metadata_path(chunk_path))
    if metadata is None or not chunk_path.is_file():
        return False
    expected_size = end - start + 1
    try:
        size = chunk_path.stat().st_size
    except OSError:
        return False
    return (
        metadata.get("schema_version") == PARTIAL_METADATA_SCHEMA
        and metadata.get("source_url_sha256") == _source_url_digest(url)
        and metadata.get("expected_total_bytes") == info.total_bytes
        and metadata.get("range_start") == start
        and metadata.get("range_end") == end
        and metadata.get("etag") == info.etag
        and metadata.get("last_modified") == info.last_modified
        and metadata.get("expected_bytes") == expected_size
        and size == expected_size
        and metadata.get("sha256") == _file_sha256(chunk_path)
    )


def _fetch_chunk(
    url: str,
    chunk_path: Path,
    start: int,
    end: int,
    info: SourceInfo,
    *,
    retry_policy: RetryPolicy,
    opener: Callable[..., Any],
    sleeper: Callable[[float], None],
    random_value: Callable[[], float],
    timeout_seconds: float,
) -> Path:
    """Reuse a verified chunk or fetch it to scratch and publish with digest metadata."""
    if _matching_chunk(chunk_path, url, info, start, end):
        return chunk_path
    if not info.range_supported:
        raise UnsupportedRangeError("Parallel download requires HTTP byte-range support")
    for attempt in range(retry_policy.max_attempts):
        response = _request_range(
            url,
            start,
            end,
            info,
            retry_policy,
            opener=opener,
            sleeper=sleeper,
            random_value=random_value,
            timeout_seconds=timeout_seconds,
        )
        temporary: Path | None = None
        try:
            status = getattr(response, "status", response.getcode())
            if status == 200:
                raise UnsupportedRangeError("Server ignored the Range request")
            if status == 416:
                raise UnsupportedRangeError("Server rejected a parallel byte range")
            _validate_range_response(response, start, end, info)
            if not _source_matches_response(info, response):
                raise IntegrityError("Source validator changed during chunk transfer")
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{chunk_path.name}.", suffix=".pending", dir=chunk_path.parent
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as output:
                received = 0
                while True:
                    chunk = response.read(IO_CHUNK_SIZE)
                    if not chunk:
                        break
                    if received + len(chunk) > end - start + 1:
                        raise IntegrityError("Chunk response exceeded expected chunk size")
                    output.write(chunk)
                    received += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if received != end - start + 1:
                raise http.client.IncompleteRead(b"", end - start + 1 - received)
            digest = _file_sha256(temporary)
            os.replace(temporary, chunk_path)
            temporary = None
            atomic_write_json(
                _chunk_metadata_path(chunk_path),
                {
                    "schema_version": PARTIAL_METADATA_SCHEMA,
                    "source_url_sha256": _source_url_digest(url),
                    "source_url": _safe_source_url(url),
                    "expected_total_bytes": info.total_bytes,
                    "range_start": start,
                    "range_end": end,
                    "expected_bytes": end - start + 1,
                    "etag": info.etag,
                    "last_modified": info.last_modified,
                    "sha256": digest,
                    "verified": True,
                },
            )
            return chunk_path
        except (
            URLError,
            TimeoutError,
            ConnectionError,
            OSError,
            http.client.HTTPException,
        ) as error:
            if attempt + 1 >= retry_policy.max_attempts:
                raise ExtractError(
                    f"Chunk transfer failed after {attempt + 1} attempt(s): {type(error).__name__}"
                ) from None
            sleeper(_retry_delay(attempt, retry_policy, None, random_value))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            response.close()
    raise ExtractError("Chunk transfer exhausted its retry policy")


def download_parallel(
    url: str,
    target: Path,
    *,
    approval: Mapping[str, Any] | None = None,
    official_sha256: str | None = None,
    retry_policy: RetryPolicy = RetryPolicy(),
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    workers: int = DEFAULT_WORKERS,
    lock_timeout_seconds: float = 0,
    opener: Callable[..., Any] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Download, checksum and combine exact source-bound ranges under the shared archive lock."""
    _require_approval(url, approval)
    if official_sha256 is not None and (
        len(official_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in official_sha256)
    ):
        raise ValueError("official_sha256 must be null or a 64-digit SHA-256")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or chunk_size <= 0
        or workers <= 0
    ):
        raise ValueError("chunk_size and workers must be positive")
    target = Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    LOGGER.info("extract run_id=%s mode=parallel event=start target=%s", run_id, target.name)
    with resource_lock(target, lock_timeout_seconds):
        existing = _validate_complete_archive(target, url, official_sha256)
        if existing is not None:
            return existing
        info = probe_source(
            url,
            retry_policy,
            approval=approval,
            opener=opener,
            sleeper=sleeper,
            random_value=random_value,
            timeout_seconds=timeout_seconds,
        )
        if not info.range_supported:
            raise UnsupportedRangeError("Parallel download requires HTTP byte-range support")
        chunk_root = target.with_name(f"{target.name}.chunks")
        chunk_root.mkdir(parents=True, exist_ok=True)
        ranges = [
            (index, start, min(info.total_bytes - 1, start + chunk_size - 1))
            for index, start in enumerate(range(0, info.total_bytes, chunk_size))
        ]

        def fetch_range(entry: tuple[int, int, int]) -> Path:
            """Fetch or validate a range using a stable index-based chunk name."""
            index, start, end = entry
            return _fetch_chunk(
                url,
                chunk_root / f"{index:06d}.part",
                start,
                end,
                info,
                retry_policy=retry_policy,
                opener=opener,
                sleeper=sleeper,
                random_value=random_value,
                timeout_seconds=timeout_seconds,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            chunk_paths = list(executor.map(fetch_range, ranges))
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".pending", dir=target.parent
        )
        assembled = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                for chunk_path, (_, start, end) in zip(chunk_paths, ranges, strict=True):
                    if not _matching_chunk(chunk_path, url, info, start, end):
                        raise IntegrityError("Chunk evidence changed before archive combine")
                    with chunk_path.open("rb") as source:
                        shutil.copyfileobj(source, output, IO_CHUNK_SIZE)
                output.flush()
                os.fsync(output.fileno())
            result = _publish_archive(assembled, target, url, info, official_sha256, run_id)
            LOGGER.info("extract run_id=%s mode=parallel event=%s", run_id, result["state"])
            return result
        finally:
            assembled.unlink(missing_ok=True)
