"""Read-only inspection: no downloads, training, test suites or container mutations."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "run_poc_experiment.py",
    "download_food101.py",
    "download_food101_parallel.py",
    "train_hog_svm.py",
    "train_resnet18.py",
    "train_efficientnet_b0.py",
    "evaluate_models.py",
    "verify_dataset.py",
    "create_splits.py",
}


def active_tasks():
    result = []
    for process in psutil.process_iter(["pid", "cmdline", "cwd", "create_time"]):
        try:
            argv = process.info["cmdline"] or []
            cwd = Path(process.info["cwd"] or ".").resolve()
            if cwd != ROOT:
                continue
            script = next((Path(arg).name for arg in argv if Path(arg).name in TASKS), None)
            module = argv[argv.index("-m") + 1] if "-m" in argv else None
            if script or module in {"pytest", "pip"}:
                result.append(
                    {
                        "pid": process.pid,
                        "parent_pid": process.ppid(),
                        "task": script or module,
                        "elapsed_seconds": round(time.time() - process.create_time()),
                        "cpu_seconds": round(process.cpu_times().user),
                        "rss_mb": round(process.memory_info().rss / 2**20),
                    }
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    return result


def inspect_project(include_docker=False):
    root = ROOT / "data/raw/food-101"
    metadata = root / "download_metadata.json"
    dataset = json.loads(metadata.read_text()) if metadata.exists() else {}
    partial = ROOT / "data/raw/food-101.tar.gz.part"
    dataset = {
        "completed_marker": metadata.exists(),
        "actual_image_count": len(list((root / "images").glob("*/*.jpg"))),
        "archive_sha256": dataset.get("download_sha256"),
        "retained_sequential_partial_bytes": partial.stat().st_size if partial.exists() else 0,
        "note": (
            "Retained partial/chunk files do not mean the completed dataset is still downloading."
        ),
    }
    manifest = ROOT / "data/splits/v1/dataset_metadata.json"
    models = []
    for path in (ROOT / "artifacts/models").glob("*/*/model_config.json"):
        config = json.loads(path.read_text())
        models.append(
            {
                "name": config["model_name"],
                "run_id": config["run_id"],
                "trained": config.get("trained", False),
                "smoke_only": config.get("smoke_only"),
                "sealed": (path.parent / "checksum.sha256").exists(),
                "test_evaluated": (path.parent / "metrics_test.json").exists(),
            }
        )
    result = {
        "root": str(ROOT),
        "active_tasks": active_tasks(),
        "dataset": dataset,
        "poc": json.loads(manifest.read_text()) if manifest.exists() else None,
        "models": models,
        "production_pointer_exists": (ROOT / "artifacts/production/current.json").exists(),
    }
    if include_docker:
        try:
            completed = subprocess.run(
                ["docker", "compose", "ps"], cwd=ROOT, capture_output=True, text=True, timeout=15
            )
            result["docker"] = {
                "exit_code": completed.returncode,
                "status": completed.stdout.strip(),
                "error": completed.stderr.strip(),
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["docker"] = {"error": type(exc).__name__}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker", action="store_true", help="Read docker compose ps only")
    args = parser.parse_args()
    print(json.dumps(inspect_project(args.docker), indent=2, ensure_ascii=True))
