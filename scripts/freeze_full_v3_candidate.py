"""Freeze the winner from a validated full-v3 validation selection record."""

import argparse
import json
from pathlib import Path

from food_assistant.ml.full_v3_release import freeze_selected_record, read_candidate


def main():
    """Recompute validation ranking and freeze the unchanged selected artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    candidates = [
        read_candidate(row["artifact_dir"], args.protocol)
        for row in selection.get("ranked_candidates", [])
    ]
    result = freeze_selected_record(candidates, args.selection, args.protocol, args.output)
    print(f"Candidate freeze status: {result['status']}")


if __name__ == "__main__":
    main()
