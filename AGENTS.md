# Food Recognition — Codex Working Agreement

## Project
Repository: food-recognition-dietary-assistant
Architecture: Food-101 ELT + ML Image Processing

## Working rules
- Read repository guardrails and relevant Phase reports before editing.
- Preserve all pre-existing uncommitted changes.
- Do not overwrite or delete existing user work.
- Do not commit or push unless explicitly requested.
- Prefer minimal code changes.
- Investigate root causes rather than hiding failing tests.

## Protected data
Do not modify:
- data/raw/**
- data/splits/v1/**
- data/splits/v2/**
- data/splits/v3/**
- data/splits/poc-v2/**
- Published manifests and official metadata
- Existing model weights and preprocessing artifacts

Do not download, resplit, rehash full Food-101,
train models, or run full-dataset jobs without approval.

## Testing
- Prefer Python 3.11 in .venv-311 when available.
- Run targeted tests before broader regression tests.
- Run Ruff when applicable.
- Never claim PASS for tests that were not executed.
- Distinguish synthetic, sample and full-data evidence.

## Completion report
Always report:
- Baseline and scope
- Files changed
- Root cause and implementation
- Commands and tests executed
- PASS / FAIL / INCOMPLETE
- Data safety
- Remaining risks
- Recommended next action

Stop after the assigned task.