"""Run or preflight the configured Food-101 ELT orchestration DAG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from food_assistant.data_pipeline.pipeline import STAGES, run_pipeline


def main() -> int:
    """Parse the orchestration CLI, run the selected DAG prefix, and return its status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/food101_elt.yaml"),
        help="Validated Food-101 ELT configuration.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", *STAGES[:-1]),
        default="all",
        help="Final stage requested; all prerequisites still run in DAG order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a read-only stage plan without creating state, locks, or reports.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the newest matching run, or select it with --run-id/--report-dir.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Run report directory under artifacts/reports/elt (otherwise generated per run).",
    )
    parser.add_argument("--run-id", help="Stable safe run identifier for report selection/resume.")
    arguments = parser.parse_args()
    result = run_pipeline(
        arguments.config,
        stage=arguments.stage,
        dry_run=arguments.dry_run,
        resume=arguments.resume,
        report_dir=arguments.report_dir,
        run_id=arguments.run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"PASS", "DRY_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
