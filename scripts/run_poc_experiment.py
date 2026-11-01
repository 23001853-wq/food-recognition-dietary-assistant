"""Run the declared short CPU POC protocol after the approved archive is ready."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment_budget.yaml"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly execute long jobs; default only prints commands",
    )
    args = parser.parse_args()
    budget = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.run_id = args.run_id or budget["experiment"]
    root = Path("data/raw/food-101")
    if args.execute and not (root / "download_metadata.json").exists():
        raise SystemExit("BLOCKED: official dataset download/extraction has not completed")
    logdir = Path("artifacts/reports") / args.run_id
    if args.execute:
        from food_assistant.utils.process_lock import running_training

        for model in ["hog_svm", "resnet18", "efficientnet_b0"]:
            running = running_training(f"train_{model}.py", Path("artifacts/models"))
            if running:
                raise SystemExit(f"BLOCKED: {model} already running: {running}")
        logdir.mkdir(parents=True, exist_ok=True)

    def run(script, *arguments):
        if not args.execute:
            print(
                "& " + subprocess.list2cmdline([sys.executable, "-u", script, *map(str, arguments)])
            )
            return
        with (logdir / (Path(script).stem + ".log")).open("a", encoding="utf-8") as output:
            result = subprocess.run(
                [sys.executable, "-u", script, *map(str, arguments)],
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise SystemExit(f"Failed: {script}; see {logdir}")

    if not Path("data/splits/v1/dataset_manifest.csv").exists():
        run("scripts/create_splits.py")
    run("scripts/verify_dataset.py")
    for model in ["hog_svm", "resnet18", "efficientnet_b0"]:
        artifact = Path("artifacts/models") / model / args.run_id
        if not (args.skip_existing and (artifact / "checksum.sha256").exists()):
            arguments = ["--run-id", args.run_id]
            if model != "hog_svm":
                arguments += [
                    "--device",
                    budget["device"],
                    "--head-epochs",
                    str(budget["cnn_head_epochs"]),
                    "--fine-tune-epochs",
                    str(budget["cnn_fine_tune_epochs"]),
                    "--batch-size",
                    str(budget["batch_size"]),
                ]
            if args.execute:
                print(f"Training {model}; see {logdir}", flush=True)
            run(f"scripts/train_{model}.py", *arguments)
    # Candidates are now frozen. Do not optimize hyperparameters based on these test results.
    artifacts = [
        Path("artifacts/models") / name / args.run_id
        for name in ["hog_svm", "resnet18", "efficientnet_b0"]
    ]
    protocol = logdir / "frozen_protocol.json"
    if not protocol.exists():
        run("scripts/freeze_poc.py", "--output", protocol, "--artifacts", *artifacts)
    for artifact in artifacts:
        if not (args.skip_existing and (artifact / "metrics_test.json").exists()):
            run(
                "scripts/evaluate_models.py",
                "--artifact",
                artifact,
                "--split",
                "test",
                "--protocol",
                protocol,
            )
    run("scripts/select_production_model.py", "--artifacts", *artifacts)
    print(
        "Execution finished; inspect actual metrics."
        if args.execute
        else "DRY RUN ONLY: no download, training or evaluation was started.",
        flush=True,
    )


if __name__ == "__main__":
    main()
