"""Offline training and artifact exports. Smoke runs are explicitly ineligible for production."""

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader

from food_assistant.config import PROJECT_ROOT
from food_assistant.ml.datasets import ManifestDataset, write_json
from food_assistant.ml.hog_features import extract_hog
from food_assistant.ml.metrics import compute_metrics
from food_assistant.ml.model_factory import create_model
from food_assistant.ml.transforms import build_transform, preprocessing
from food_assistant.utils.process_lock import training_lock


def device_for(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; explicitly choose CPU")
    return torch.device(name)


def accumulation_window(index, batches, accumulation):
    """Use the actual last window size so a partial accumulation is not underweighted."""
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    start = (index // accumulation) * accumulation
    size = min(accumulation, batches - start)
    return size, (index + 1) % accumulation == 0 or index + 1 == batches


def seal_artifact(output):
    checksums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(output).iterdir())
        if p.is_file() and p.name != "checksum.sha256"
    }
    write_json(Path(output) / "checksum.sha256", checksums)


def export_contract(output, name, run_id, dataset, config, smoke, preproc, pretrained):
    output.mkdir(parents=True, exist_ok=False)
    labels = dataset.labels
    write_json(output / "labels.json", labels)
    mapping = {row["class_key"]: row["class_id"] for row in labels["labels"]}
    write_json(output / "class_to_idx.json", mapping)
    write_json(output / "idx_to_class.json", {str(v): k for k, v in mapping.items()})
    manifest = Path(config["manifest_dir"]) / "dataset_manifest.csv"
    write_json(
        output / "model_config.json",
        {
            "model_name": name,
            "run_id": run_id,
            "num_classes": len(mapping),
            "dataset_version": labels["dataset_version"],
            "smoke_only": smoke or dataset.metadata.get("is_synthetic", False),
            "pretrained": pretrained,
            "trained": False,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "dataset_root": dataset.metadata["root"],
            "source": dataset.metadata.get("source"),
            "torch_version": torch.__version__,
            "device": "cpu" if name == "hog_svm" else str(device_for(config.get("device", "auto"))),
        },
    )
    write_json(output / "preprocessing.json", preproc)
    (output / "train_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    (output / "source_commit.txt").write_text(
        commit.stdout.strip() or "unavailable", encoding="utf-8"
    )
    changed = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    sources = [
        *PROJECT_ROOT.joinpath("src").rglob("*.py"),
        *PROJECT_ROOT.joinpath("configs").rglob("*.yaml"),
        PROJECT_ROOT / "pyproject.toml",
    ]
    write_json(
        output / "source_state.json",
        {
            "working_tree_dirty": bool(changed.stdout.strip()),
            "file_sha256": {
                str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sources
            },
        },
    )


def run_training(name):
    parser = argparse.ArgumentParser(description=f"Train {name} offline")
    parser.add_argument("--config", type=Path, default=Path(f"configs/{name}.yaml"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/splits/v1"))
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Architecture smoke only; not transfer learning",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--head-epochs", type=int)
    parser.add_argument("--fine-tune-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if args.resume and name == "hog_svm":
        parser.error("HOG/SVM has no mid-fit resume; use a fresh run after the active fit finishes")
    with training_lock(name, args.output_root):
        started = time.perf_counter()
        torch.hub.set_dir(str(PROJECT_ROOT / "artifacts/pretrained"))
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        for key in ("head_epochs", "fine_tune_epochs", "batch_size"):
            value = getattr(args, key)
            if value is not None:
                if value < 1:
                    parser.error(f"{key} must be positive")
                config[key] = value
        config["manifest_dir"] = str(args.manifest_dir)
        if args.device:
            config["device"] = args.device
        if name != "hog_svm":
            device_for(config.get("device", "auto"))
        run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,69}", run_id):
            parser.error("run-id must be a safe identifier of at most 70 characters")
        output = args.output_root / name / run_id
        if output.exists():
            parser.error(f"Run folder already exists: {output}. Keep it and choose a new run-id.")
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.set_num_threads(2)
        if args.no_pretrained and not args.smoke:
            parser.error("--no-pretrained is allowed only with --smoke")
        limit = 2 if args.smoke else None
        preproc = config if name == "hog_svm" else preprocessing(name)
        train = ManifestDataset(
            args.manifest_dir,
            "train",
            None if name == "hog_svm" else build_transform(preproc, True),
            limit,
        )
        val = ManifestDataset(
            args.manifest_dir, "val", None if name == "hog_svm" else build_transform(preproc), limit
        )
        if config.get("expected_num_classes") is not None:
            if len(train.labels["labels"]) != config["expected_num_classes"]:
                raise ValueError("Unexpected class count for this experiment")
        if config.get("expected_dataset_version") is not None:
            if train.labels["dataset_version"] != config["expected_dataset_version"]:
                raise ValueError("Unexpected dataset version for this experiment")
        if args.resume:
            previous = read_resume_contract(args.resume, name, train, preproc, config)
        else:
            previous = None
        export_contract(
            output,
            name,
            run_id,
            train,
            config,
            args.smoke,
            preproc,
            previous["pretrained"] if previous else not args.no_pretrained and name != "hog_svm",
        )
        history = []
        if name == "hog_svm":

            def feature_matrix(dataset):
                first, _ = dataset[0]
                matrix = np.empty((len(dataset), len(extract_hog(first, config))), dtype=np.float32)
                for index, (image, _) in enumerate(dataset):
                    matrix[index] = extract_hog(image, config)
                    if (index + 1) % 500 == 0:
                        print(f"HOG features: {index + 1}/{len(dataset)}", flush=True)
                return matrix

            train_x = feature_matrix(train)
            train_y = [y for _, y in train]
            val_x = feature_matrix(val)
            val_y = [y for _, y in val]
            best_score, best_model = -1, None
            for c in config["c_candidates"]:
                model = make_pipeline(
                    StandardScaler(),
                    LinearSVC(
                        C=c,
                        random_state=config["seed"],
                        dual=config.get("dual", False),
                        max_iter=config.get("max_iter", 2000),
                        tol=config.get("tol", 0.001),
                    ),
                )
                print(f"Fitting LinearSVC C={c}, dual={config.get('dual', False)}", flush=True)
                fit_started = time.perf_counter()
                with warnings.catch_warnings(record=True) as fit_warnings:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(train_x, train_y)
                for warning in fit_warnings:
                    print(f"SVM warning: {warning.message}", flush=True)
                metric = compute_metrics(val_y, model.predict(val_x), len(train.labels["labels"]))
                history.append(
                    {
                        "epoch": len(history),
                        "trial_C": c,
                        "train_loss": "",
                        "val_macro_f1": metric["macro_f1"],
                        "fit_seconds": time.perf_counter() - fit_started,
                        "iterations": int(model[-1].n_iter_),
                        "converged": not any(
                            issubclass(w.category, ConvergenceWarning) for w in fit_warnings
                        ),
                    }
                )
                print(f"C={c}: validation macro-F1={metric['macro_f1']:.4f}", flush=True)
                if metric["macro_f1"] > best_score:
                    best_score, best_model = metric["macro_f1"], model
                    joblib.dump(best_model, output / "best_candidate.joblib")
                write_json(output / "svm_trials.json", history)
            joblib.dump(best_model, output / "model.joblib")
            best_metric = compute_metrics(
                val_y, best_model.predict(val_x), len(train.labels["labels"])
            )
        else:
            device = device_for(config.get("device", "auto"))
            use_amp = device.type == "cuda" and config.get("amp", False)
            accumulation = config.get("gradient_accumulation_steps", 1)
            if not isinstance(accumulation, int) or accumulation < 1:
                raise ValueError("gradient_accumulation_steps must be a positive integer")
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            model = create_model(
                name,
                len(train.labels["labels"]),
                pretrained=not args.no_pretrained and args.resume is None,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config["head_lr"], weight_decay=config["weight_decay"]
            )
            loader = DataLoader(
                train,
                batch_size=2 if args.smoke else config["batch_size"],
                shuffle=True,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            validation = DataLoader(
                val,
                batch_size=2 if args.smoke else config["batch_size"],
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            start_epoch, best_score, stale = 0, -1, 0
            best_metric, best_state = None, None
            if args.resume:
                resume = torch.load(args.resume, map_location=device, weights_only=True)
                if (
                    resume["manifest_sha256"]
                    != json.loads((output / "model_config.json").read_text())["manifest_sha256"]
                ):
                    raise ValueError("Resume manifest mismatch")
                model.load_state_dict(resume["model"])
                optimizer.load_state_dict(resume["optimizer"])
                start_epoch, best_score = resume["epoch"] + 1, resume["best_score"]
                best_metric, best_state = resume["best_metric"], resume["best_model"]
                if resume.get("scaler"):
                    scaler.load_state_dict(resume["scaler"])
                history = resume.get("history", [])
                if "torch_rng" in resume:
                    torch.set_rng_state(resume["torch_rng"].cpu())
                    random.setstate(resume["python_rng"])
                    state = resume["numpy_rng"]
                    np.random.set_state(
                        (
                            state[0],
                            np.array(state[1], dtype=np.uint32),
                            state[2],
                            state[3],
                            state[4],
                        )
                    )
                    if device.type == "cuda" and resume.get("cuda_rng"):
                        torch.cuda.set_rng_state_all([state.cpu() for state in resume["cuda_rng"]])
                stale = resume.get("stale", 0)
                save_checkpoint(best_state, output / "model.pth")
            epochs = 1 if args.smoke else config["head_epochs"] + config["fine_tune_epochs"]
            if start_epoch >= epochs:
                raise ValueError("Resume checkpoint has already completed configured epochs")
            for epoch in range(start_epoch, epochs):
                head_only = epoch < config["head_epochs"] and not args.smoke
                for key, parameter in model.named_parameters():
                    parameter.requires_grad_(
                        not head_only or key.startswith(("fc.", "classifier."))
                    )
                model.train()
                if head_only:
                    # Freeze backbone batch normalization as well as its parameters.
                    for module in model.modules():
                        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                            module.eval()
                for group in optimizer.param_groups:
                    group["lr"] = config["head_lr"] if head_only else config["fine_tune_lr"]
                losses = []
                optimizer.zero_grad(set_to_none=True)
                for batch_index, (images, targets) in enumerate(loader):
                    images = images.to(device, non_blocking=device.type == "cuda")
                    targets = targets.to(device, non_blocking=device.type == "cuda")
                    window, step = accumulation_window(batch_index, len(loader), accumulation)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        loss = torch.nn.functional.cross_entropy(model(images), targets)
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            "Non-finite training loss; checkpoint not marked trained"
                        )
                    scaler.scale(loss / window).backward()
                    if step:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    losses.append(loss.item())
                    if len(losses) % 50 == 0:
                        print(
                            f"epoch={epoch} batch={len(losses)}/{len(loader)} "
                            f"loss={loss.item():.4f}",
                            flush=True,
                        )
                truth, predicted = [], []
                model.eval()
                with torch.inference_mode():
                    for images, targets in validation:
                        predicted.extend(model(images.to(device)).argmax(1).cpu().tolist())
                        truth.extend(targets.tolist())
                metric = compute_metrics(truth, predicted, len(train.labels["labels"]))
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": float(np.mean(losses)),
                        "val_macro_f1": metric["macro_f1"],
                    }
                )
                print(
                    f"epoch={epoch} loss={np.mean(losses):.4f} "
                    f"val_macro_f1={metric['macro_f1']:.4f}",
                    flush=True,
                )
                if metric["macro_f1"] > best_score:
                    best_score, stale, best_metric = metric["macro_f1"], 0, metric
                    save_checkpoint(model.state_dict(), output / "model.pth")
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                else:
                    stale += 1
                save_checkpoint(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_score": best_score,
                        "best_metric": {key: float(value) for key, value in best_metric.items()},
                        "best_model": best_state,
                        "scaler": scaler.state_dict(),
                        "history": history,
                        "stale": stale,
                        "torch_rng": torch.get_rng_state(),
                        "python_rng": random.getstate(),
                        "numpy_rng": numpy_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        "manifest_sha256": json.loads((output / "model_config.json").read_text())[
                            "manifest_sha256"
                        ],
                    },
                    output / "resume.pth",
                )
                if stale >= config.get("patience", 5):
                    break
        with (output / "training_history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "trial_C",
                    "train_loss",
                    "val_macro_f1",
                    "fit_seconds",
                    "iterations",
                    "converged",
                ],
            )
            writer.writeheader()
            writer.writerows(history)
        write_json(output / "metrics_val.json", best_metric)
        metadata = json.loads((output / "model_config.json").read_text())
        metadata.update(trained=True, training_samples=len(train), validation_samples=len(val))
        metadata.update(
            training_seconds=time.perf_counter() - started,
            amp_enabled=name != "hog_svm" and use_amp,
            cuda_version=torch.version.cuda,
        )
        if name != "hog_svm" and device.type == "cuda":
            metadata.update(
                gpu_name=torch.cuda.get_device_name(device),
                peak_gpu_memory_mb=torch.cuda.max_memory_allocated(device) / 2**20,
            )
        write_json(output / "model_config.json", metadata)
        seal_artifact(output)
        print(f"TRAINED {'SMOKE ONLY' if args.smoke else 'CANDIDATE'}: {output}")


def numpy_rng_state():
    state = np.random.get_state()
    return (state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4]))


def save_checkpoint(value, target):
    pending = target.with_name(target.name + ".pending")
    torch.save(value, pending)
    pending.replace(target)


def read_resume_contract(checkpoint, name, dataset, preproc, config):
    root = checkpoint.parent
    previous = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
    labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    prior_preproc = json.loads((root / "preprocessing.json").read_text(encoding="utf-8"))
    prior_config = yaml.safe_load((root / "train_config.yaml").read_text(encoding="utf-8"))
    if previous["model_name"] != name or labels != dataset.labels or prior_preproc != preproc:
        raise ValueError("Resume architecture, labels or preprocessing mismatch")
    if previous.get("smoke_only"):
        raise ValueError("Smoke checkpoints cannot be resumed into a formal experiment")
    if prior_config["head_epochs"] != config["head_epochs"]:
        raise ValueError("Resume head_epochs must stay fixed; only extend fine_tune_epochs")
    return previous
