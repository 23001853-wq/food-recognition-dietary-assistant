"""Nonblocking per-model locks, including detection of older runs without lock files."""

import json
import os
from contextlib import contextmanager
from pathlib import Path


def running_training(script_name, output_root):
    import psutil

    excluded = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    matches = []
    for process in psutil.process_iter(["pid", "cmdline", "cwd"]):
        try:
            if process.pid in excluded:
                continue
            argv = process.info["cmdline"] or []
            if not any(Path(arg).name == script_name for arg in argv):
                continue
            cwd = Path(process.info["cwd"] or ".")
            index = argv.index("--output-root") if "--output-root" in argv else None
            root = Path(argv[index + 1]) if index is not None else Path("artifacts/models")
            root = root if root.is_absolute() else cwd / root
            if root.resolve() == Path(output_root).resolve():
                matches.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    return matches


@contextmanager
def training_lock(model_name, output_root, check_processes=True):
    if check_processes:
        running = running_training(f"train_{model_name}.py", output_root)
        if running:
            raise RuntimeError(
                f"Training already active for {model_name}: PIDs {running}. "
                "Do not start a duplicate."
            )
    folder = Path(output_root) / ".locks"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{model_name}.lock"
    with path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b" ")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another {model_name} run holds {path}") from exc
        try:
            handle.seek(1)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "model": model_name}).encode())
            handle.flush()
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
