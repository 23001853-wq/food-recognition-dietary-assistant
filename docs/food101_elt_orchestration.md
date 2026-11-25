# Food-101 ELT orchestration (Phase 5)

## Run commands

```powershell
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --help
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --stage all --dry-run
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --config configs\food101_elt_poc_v2.yaml --stage all --dry-run
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --stage all
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --stage transform --resume --run-id <run-id>
```

`--stage` selects the final stage, not a way around prerequisites: requesting Transform still runs Inspect, Extract/Load decisions, and the Quality gate first. `all` runs the complete DAG. Dry-run prints a plan and does not create run directories, locks, checkpoints, or reports. A normal run gets a unique `elt-<UTC>-<suffix>` id. Resume accepts `--run-id`, a `--report-dir` whose final component is the run id, or discovers the newest valid state with the same config hash.

| Config | Scope/version | Transform route | Compatibility |
|---|---|---|---|
| `configs/food101_elt.yaml` | POC / `v2` | None | Deliberately `BLOCKED`; retained unchanged because other tools use it as a legacy default. |
| `configs/food101_elt_poc_v2.yaml` | POC / `poc-v2` | `build_poc_transform` | Supported; recommended explicit config for POC orchestration. |
| `configs/food101_elt_full_v2.yaml` | Full / `v2` | `build_clean_split` | Supported when source quality evidence passes. |
| `configs/food101_elt_full_v3.yaml` | Full / `v3` | `build_training_split_v3` | Supported when source v2 and quality evidence validate. |

Use `--config configs/food101_elt_poc_v2.yaml` to run/preflight the supported POC contract. The runner never silently changes scope or version when a config is incompatible.

Reports and the versioned `run_state.json` are stored in `artifacts/reports/elt/<run-id>/`. The state contains stage status, attempt count, timestamps, duration, failure reason, config hash, available source/split fingerprints, and skip reasons. Updates use atomic JSON writes under a run-scoped resource lock. A stale or malformed state is not trusted. Completed non-skip stages are re-run through their existing idempotent validators/builders on resume; immutable transform builders validate matching existing outputs and refuse conflicts.

## Safety and stage behavior

- The pipeline never initiates network access when the raw tree is absent. With downloads disabled, Extract reports `BLOCKED`; Load and downstream stages do not run.
- If raw directories already exist, Inspect validates them against the Phase 2.5 checkpoint (including current file signatures and source/inventory fingerprints). Extract and Load are skipped only after this evidence matches. Transform remains gated on the full existing verifier returning `PASS`.
- The quality adapter calls `audit_food101(..., mode="full", reject_cross_split_duplicates=False)`. This keeps raw integrity separate from the official split leakage warning; duplicate leakage is reported, not hidden or silently resolved here.
- Transform dispatch supports the implemented POC `poc-v2`, clean full `v2`, and full `v3` entry points. Unsupported scope/version pairs fail closed. In particular, the legacy default `configs/food101_elt.yaml` currently says POC + `v2`, which has no matching builder; use the explicit versioned config for the intended transform.
- The Validate stage uses the full-v3 validator for v3, or `ManifestDataset` for other published manifest versions. No training, model inference, warehouse, or scheduler is part of this command.

Extract and Load use the existing `download_sequential` and safe `scripts.download_food101.safe_extract` implementations. Extract reaches the network only when the validated config says `approved_only` and the configured approval record confirms the exact archive URL. With the checked-in `disabled` policy, missing raw returns `BLOCKED` without a network call. Load refuses an incompatible raw target name and relies on the safe extractor's full structural, count, and decode checks before first publication.
