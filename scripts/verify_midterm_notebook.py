"""Verify executed notebook output and report evidence, without executing its experiments."""

import ast
import hashlib
import json
from pathlib import Path

import nbformat
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def verify():
    primary = ROOT / "notebooks/midterm_image_processing_demo.ipynb"
    compatibility = ROOT / "notebooks/notebooks/midterm_image_processing_demo.ipynb"
    notebook = nbformat.read(primary, as_version=4)
    nbformat.validate(notebook)
    assert primary.read_bytes() == compatibility.read_bytes(), "Notebook copies differ"
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    embedded, errors = 0, []
    for cell in cells:
        ast.parse(cell.source)
        assert cell.execution_count is not None, "Unexecuted code cell"
        for output in cell.outputs:
            if output.output_type == "error":
                errors.append(output)
            embedded += "image/png" in output.get("data", {})
    assert not errors and embedded == 13
    reports = sorted(
        (ROOT / "artifacts/reports/midterm_visualizations").glob("run_*/execution_summary.json")
    )
    assert reports, "Missing executed report"
    summary = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert summary["status"] == "PASS" and summary["sample_count"] == 4
    assert summary["sample_splits"] == ["val"] and summary["test_images_opened"] == 0
    assert not summary["training_performed"] and not summary["test_evaluation_performed"]
    assert summary["hog_features"] == 8100
    assert summary["hog_visualization_matches_pipeline"]
    assert summary["cnn_normalization_matches_pipeline"] and summary["noise_repeatable"]
    assert summary["protected_files_unchanged"]
    for name in summary["figures"]:
        with Image.open(ROOT / name) as figure:
            figure.verify()
    for name in ("noise_metrics.csv", "saved_model_comparison.csv"):
        assert (reports[-1].parent / name).stat().st_size > 0
    print(
        json.dumps(
            {
                "status": "PASS",
                "executed_code_cells": len(cells),
                "embedded_figures": embedded,
                "exported_figures": len(summary["figures"]),
                "identical_notebook_copies": True,
                "notebook_sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
                "report": str(reports[-1].relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    verify()
