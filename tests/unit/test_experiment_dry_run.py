import runpy
import subprocess
import sys
from pathlib import Path


def test_default_experiment_only_prints_commands(monkeypatch, capsys):
    run_id = "unit-dry-run-no-execution"
    monkeypatch.setattr(sys, "argv", ["run_poc_experiment.py", "--run-id", run_id])

    def forbid_spawn(*args, **kwargs):
        raise AssertionError("Dry-run must not spawn a task")

    monkeypatch.setattr(subprocess, "run", forbid_spawn)
    runpy.run_path("scripts/run_poc_experiment.py", run_name="__main__")
    output = capsys.readouterr().out
    assert "DRY RUN ONLY" in output
    assert "train_resnet18.py" in output
    assert not (Path("artifacts/reports") / run_id).exists()
    assert not (Path("artifacts/models/resnet18") / run_id).exists()
