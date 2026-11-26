"""Select a full-v3 candidate using validation evidence only."""

import argparse
from pathlib import Path

from food_assistant.ml.full_v3_release import read_candidate, write_selection_record


def main():
    """Validate candidate artifacts, rank them, and write a separate selection record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.artifacts) < 2:
        parser.error("At least two independently trained methods are required")
    candidates = [read_candidate(path, args.protocol) for path in args.artifacts]
    selection = write_selection_record(candidates, args.protocol, args.output)
    print(
        "Selected by validation only: "
        f"{selection['selected_model_name']}/{selection['selected_run_id']}"
    )


if __name__ == "__main__":
    main()
