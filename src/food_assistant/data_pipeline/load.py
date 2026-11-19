"""Secure staged extraction and first-publication-only raw directory publishing."""

import shutil
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from food_assistant.data_pipeline.atomic import resource_lock


def _safe_member_path(name: str, staging_root: Path) -> Path:
    """Resolve a POSIX archive member under staging and reject unsafe spellings."""
    if not name or "\\" in name or ":" in name or name.startswith("/"):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    parts = name.rstrip("/").split("/")
    if any(part in {"", ".", ".."} or part.endswith((".", " ")) for part in parts):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10)
    }
    if any(part.split(".", 1)[0].upper() in reserved for part in parts):
        raise ValueError(f"Reserved Windows archive member path: {name!r}")
    member = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in member.parts):
        raise ValueError(f"Unsafe archive member path: {name!r}")
    target = staging_root.joinpath(*member.parts).resolve()
    if not target.is_relative_to(staging_root.resolve()):
        raise ValueError(f"Archive member escapes staging: {name!r}")
    return target


def extract_archive_staged(
    archive: Path,
    target: Path,
    validator: Callable[[Path], bool],
    max_files: int = 110_000,
    max_unpacked_bytes: int = 100_000_000_000,
    lock_timeout_seconds: float = 0,
    strip_prefix: str | None = None,
) -> Path:
    """Safely extract and validate a tar archive before publishing a new directory."""
    archive_path, destination = Path(archive).resolve(), Path(target).resolve()
    if max_files <= 0 or max_unpacked_bytes <= 0:
        raise ValueError("Archive limits must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with resource_lock(destination, lock_timeout_seconds):
        if destination.exists():
            raise FileExistsError(f"Refusing to replace existing dataset directory: {destination}")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
        )
        try:
            seen: set[str] = set()
            files = 0
            unpacked_bytes = 0
            with tarfile.open(archive_path, "r:*") as archive_file:
                for member in archive_file:
                    _safe_member_path(member.name.rstrip("/"), staging)
                    relative = PurePosixPath(member.name).as_posix().rstrip("/")
                    if strip_prefix is not None:
                        prefix = f"{strip_prefix.rstrip('/')}/"
                        if relative == strip_prefix.rstrip("/"):
                            relative = ""
                        elif not relative.startswith(prefix):
                            raise ValueError(f"Unexpected archive root: {member.name}")
                        else:
                            relative = relative[len(prefix) :]
                    if not relative:
                        relative = "."
                    identity = relative.casefold()
                    if identity in seen:
                        raise ValueError(f"Duplicate archive member: {member.name}")
                    seen.add(identity)
                    if len(seen) > max_files:
                        raise ValueError("Archive exceeds configured entry limit")
                    if relative == ".":
                        if not member.isdir():
                            raise ValueError(f"Invalid archive root entry: {member.name}")
                        continue
                    target_path = _safe_member_path(relative, staging)
                    if member.isdir():
                        target_path.mkdir(parents=True, exist_ok=True)
                        if not target_path.is_dir():
                            raise ValueError(
                                f"Archive directory collides with a file: {member.name}"
                            )
                        continue
                    if not member.isfile():
                        raise ValueError(f"Unsupported archive entry type: {member.name}")
                    files += 1
                    unpacked_bytes += member.size
                    if files > max_files or unpacked_bytes > max_unpacked_bytes:
                        raise ValueError("Archive exceeds configured extraction limits")
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    source = archive_file.extractfile(member)
                    if source is None:
                        raise ValueError(f"Cannot read archive member: {member.name}")
                    with source, target_path.open("xb") as output:
                        shutil.copyfileobj(source, output)
                        output.flush()
                if files == 0 or not validator(staging):
                    raise ValueError("Extracted dataset failed its validation contract")
            staging.rename(destination)
            return destination
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
