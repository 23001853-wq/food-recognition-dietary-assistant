"""Dataset verification, preparation and frozen-candidate evaluation commands."""

import argparse
import csv
import hashlib
import platform
import time
from pathlib import Path

import yaml
from PIL import Image

from food_assistant.ml.datasets import (
    ManifestDataset,
    audit_manifest,
    prepare_food101,
    read_json,
    write_json,
)
from food_assistant.services.model_registry import Release


def create_splits():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/food-101"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/v1"))
    parser.add_argument("--classes", type=Path, default=Path("configs/classes_poc.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = yaml.safe_load(args.classes.read_text(encoding="utf-8"))
    print(prepare_food101(args.root, args.output, config["classes"], config["version"], args.seed))


def verify_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/splits/v1"))
    args = parser.parse_args()
    metadata = read_json(args.manifest_dir / "dataset_metadata.json")
    with (args.manifest_dir / "dataset_manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    stats = audit_manifest(rows, metadata["root"])
    for row in rows:
        with Image.open(Path(metadata["root"]) / row["relative_path"]) as image:
            image.verify()
    write_json(args.manifest_dir / "quality_report.json", stats)
    print(stats)


def evaluate_models():
    import torch

    from food_assistant.ml.metrics import export_metrics
    from food_assistant.ml.train_utils import seal_artifact

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/splits/v1"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    release = Release(args.artifact, args.device)
    frozen = None
    if args.protocol:
        from food_assistant.ml.protocol import verify_frozen_candidate

        frozen = verify_frozen_candidate(args.protocol, args.artifact, args.device)
    elif not args.smoke and release.config.get("dataset_version") == "poc-v1":
        parser.error("Freeze all three POC candidates first and provide --protocol")
    data = ManifestDataset(args.manifest_dir, args.split, limit_per_class=2 if args.smoke else None)
    fingerprint = hashlib.sha256(
        (args.manifest_dir / "dataset_manifest.csv").read_bytes()
    ).hexdigest()
    if fingerprint != release.config["manifest_sha256"] or data.labels != release.labels:
        raise ValueError("Evaluation manifest or labels differ from training")
    if frozen and (
        frozen["manifest_sha256"] != fingerprint
        or len(data) != frozen["test_samples"]
        or args.split != "test"
        or args.smoke
    ):
        raise ValueError("Evaluation split/sample count differs from frozen protocol")
    output = args.artifact / ("evaluation_smoke" if args.smoke else f"evaluation_{args.split}")
    if (output / "metrics.json").exists():
        raise ValueError("Evaluation already exists; refusing repeated test evaluation")
    truth, predictions, timings, details = [], [], [], []
    # Warm up using validation image; test images never tune any hyperparameters.
    warmup = ManifestDataset(args.manifest_dir, "val", limit_per_class=1)
    with Image.open(warmup.root / warmup.rows[0]["relative_path"]) as image:
        for _ in range(3):
            release.predict(image.convert("RGB"), 3)
    for row in data.rows:
        with Image.open(data.root / row["relative_path"]) as image:
            start = time.perf_counter()
            result = release.predict(image.convert("RGB"), 3)
            timings.append((time.perf_counter() - start) * 1000)
        truth.append(int(row["class_id"]))
        predictions.append(result[0].class_id)
        details.append(
            {
                "image_path": row["relative_path"],
                "true_class": row["class_key"],
                "predicted_class": result[0].class_key,
                "score": result[0].score,
                "error_group": "UNREVIEWED"
                if result[0].class_id != int(row["class_id"])
                else "CORRECT",
            }
        )
    metrics = export_metrics(output, truth, predictions, data.labels["labels"], timings)
    metrics.update(
        model_name=release.config["model_name"],
        run_id=release.config["run_id"],
        dataset_version=release.config["dataset_version"],
        manifest_sha256=fingerprint,
        split=args.split,
        smoke_only=args.smoke or release.config["smoke_only"],
        artifact_size_mb=sum(p.stat().st_size for p in args.artifact.glob("model.*")) / 2**20,
        device=str(release.device),
        batch_size=1,
        warmup_iterations=3,
        latency_scope="image preprocessing plus inference; excludes file decode",
        num_classes=len(data.labels["labels"]),
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest()
        if args.protocol
        else None,
        hardware={
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
    )
    write_json(output / "metrics.json", metrics)
    write_json(output / "predictions.json", details)
    write_json(
        output / "evaluation_metadata.json",
        {
            **metrics,
            "labels": data.labels,
            "preprocessing": release.preproc,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    )
    if not args.smoke:
        write_json(args.artifact / f"metrics_{args.split}.json", metrics)
        seal_artifact(args.artifact)
    write_json(output / "evaluation_complete.json", {"complete": True, "sample_count": len(data)})
    print(metrics)


def analyze_failures():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in read_json(args.predictions) if r["error_group"] != "CORRECT"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "true_class", "predicted_class", "score", "error_group"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Exported {len(rows)} failures for manual taxonomy review; no cause inferred automatically"
    )


def select_production():
    from food_assistant.ml.release_selection import rank_candidates

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--pointer", type=Path, default=Path("artifacts/production/current.json"))
    parser.add_argument(
        "--comparison", type=Path, default=Path("artifacts/reports/experiment_comparison.csv")
    )
    args = parser.parse_args()
    if len(args.artifacts) < 2:
        parser.error("At least two independently trained methods are required")
    candidates = []
    for directory in args.artifacts:
        release = Release(directory)
        val = read_json(directory / "metrics_val.json")
        test = read_json(directory / "metrics_test.json")
        candidates.append(
            {
                "config": release.config,
                "test": test,
                "val": val,
                "labels": release.labels["labels"],
                "artifact": directory,
            }
        )
    # Rank by validation macro-F1, then measured latency; test remains a final report.
    chosen = rank_candidates(candidates)[0]
    root = Path("artifacts/models").resolve()
    relative = chosen["artifact"].resolve().relative_to(root)
    args.comparison.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_name",
        "run_id",
        "dataset_version",
        "num_classes",
        "sample_count",
        "validation_macro_f1",
        "top1_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "p50_ms",
        "p95_ms",
        "artifact_size_mb",
    ]
    with args.comparison.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for candidate in candidates:
            row = {
                **candidate["config"],
                **candidate["test"],
                "validation_macro_f1": candidate["val"]["macro_f1"],
            }
            writer.writerow({key: row.get(key) for key in columns})
    report = [
        "# Measured POC comparison",
        "",
        "Selection uses validation macro-F1; official test was evaluated after freezing.",
        "",
        "| Model | Run | Val macro-F1 | Test Top-1 | Test macro-F1 | CPU P95 ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        cfg, val, test = candidate["config"], candidate["val"], candidate["test"]
        report.append(
            f"| {cfg['model_name']} | {cfg['run_id']} | {val['macro_f1']:.4f} | "
            f"{test['top1_accuracy']:.4f} | {test['macro_f1']:.4f} | {test['p95_ms']:.2f} |"
        )
    report.extend(
        [
            "",
            f"Selected artifact: `{relative.as_posix()}`.",
            "",
            "POC 10 classes only. Scores are not calibrated probabilities. "
            "Review per-class failures and convergence/training budget before presenting results.",
        ]
    )
    args.comparison.with_suffix(".md").write_text("\n".join(report) + "\n", encoding="utf-8")
    args.pointer.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.pointer,
        {
            "artifact": relative.as_posix(),
            "reason": (
                "highest validation macro-F1; latency breaks ties; "
                "final test metrics reported separately"
            ),
        },
    )
    print(f"Production pointer: {relative}")
