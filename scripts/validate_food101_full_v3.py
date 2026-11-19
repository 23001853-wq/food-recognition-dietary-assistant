"""Run read-only validation of the existing full Food-101 v3 manifest."""

import argparse
import json
from pathlib import Path

from food_assistant.data_pipeline.splits import REPORT_PATH
from food_assistant.data_pipeline.validate_v3 import (
    FULL_V3_FINAL_REPORT,
    validate_full_v3,
    write_validation_report,
)


def main() -> int:
    """Validate v3, save a separate immutable report, and print the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/food101_elt_full_v3.yaml"),
        help="Full-scope v3 configuration.",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=REPORT_PATH,
        help="Existing Phase 2.5 full verification report.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=FULL_V3_FINAL_REPORT,
        help="New immutable report path; an existing report is never overwritten.",
    )
    arguments = parser.parse_args()
    result = validate_full_v3(arguments.config, arguments.baseline_report)
    report_path = write_validation_report(result, arguments.report)
    print(json.dumps({**result, "report_path": str(report_path)}, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
