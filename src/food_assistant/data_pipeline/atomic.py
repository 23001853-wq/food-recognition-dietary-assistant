"""Atomic file publication and resource-scoped cross-process locks."""

import csv
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ATOMIC_REPLACE_ATTEMPTS = 6
ATOMIC_REPLACE_BACKOFF_SECONDS = 0.2

ATOMIC_REPLACE_ATTEMPTS = 6
ATOMIC_REPLACE_BACKOFF_SECONDS = 0.2


def atomic_write_text(path: Path, value: str, encoding: str = "utf-8") -> None:
    """Publish text through a flushed sibling temporary file without truncating prior output."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".pending", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt + 1 == ATOMIC_REPLACE_ATTEMPTS:
                    raise
                time.sleep(ATOMIC_REPLACE_BACKOFF_SECONDS * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    """Serialize JSON and atomically replace the destination file."""
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    """Write a CSV table to a sibling temporary file and atomically publish it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".pending", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, target)
                break
            except PermissionError:
                if attempt + 1 == ATOMIC_REPLACE_ATTEMPTS:
                    raise
                time.sleep(ATOMIC_REPLACE_BACKOFF_SECONDS * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def resource_lock(resource: Path, timeout_seconds: float = 0) -> Iterator[None]:
    """Hold a stable OS-level lock file scoped to one archive, raw tree, or manifest."""
    lock_path = Path(resource).parent / f".{Path(resource).name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        started = time.monotonic()
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() - started >= timeout_seconds:
                    raise RuntimeError(f"Resource is locked: {resource}") from exc
                time.sleep(min(0.05, max(timeout_seconds, 0.05)))
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def checkpoint_digest(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 for a JSON-serializable checkpoint payload."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
