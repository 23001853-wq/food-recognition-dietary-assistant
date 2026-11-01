"""Execute only the midterm demonstration in this interpreter, no training or evaluation."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook_path = ROOT / "notebooks/midterm_image_processing_demo.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    temporary_root = ROOT / "storage/temp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="midterm_kernel_", dir=temporary_root) as directory:
        spec_root = Path(directory)
        assert spec_root.resolve().is_relative_to(temporary_root.resolve())
        spec_dir = spec_root / "project18-midterm"
        spec_dir.mkdir()
        (spec_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Project 18 CPU (temporary)",
                    "language": "python",
                    "env": {
                        "CUDA_VISIBLE_DEVICES": "",
                        "OMP_NUM_THREADS": "1",
                        "IPYTHONDIR": str(spec_root / "ipython"),
                        "MPLCONFIGDIR": str(spec_root / "matplotlib"),
                    },
                }
            ),
            encoding="utf-8",
        )
        manager = KernelManager(
            kernel_name="project18-midterm",
            connection_file=str(spec_root / "connection.json"),
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(spec_root)]),
        )

        def progress(cell, cell_index, **kwargs):
            print(f"Executing code cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

        client = NotebookClient(
            notebook,
            km=manager,
            timeout=180,
            resources={"metadata": {"path": str(ROOT)}},
            allow_errors=False,
            on_cell_execute=progress,
        )
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output.output_type == "error"
    ]
    assert not errors
    embedded = sum(
        "image/png" in output.get("data", {})
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    )
    assert embedded == 13, f"Expected 13 embedded PNG figures, found {embedded}"
    nbformat.validate(notebook)
    targets = [notebook_path, ROOT / "notebooks/notebooks/midterm_image_processing_demo.ipynb"]
    for target in targets:
        nbformat.write(notebook, target)
    print(
        f"PASS: {sum(c.cell_type == 'code' for c in notebook.cells)} code cells executed; "
        "both notebook paths saved with outputs."
    )


if __name__ == "__main__":
    main()
