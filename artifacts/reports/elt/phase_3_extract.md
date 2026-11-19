# Phase 3 — Extract: retry, resume, integrity and shared lock

## BASELINE

- Worktree already contained uncommitted work from earlier phases: edits to `.gitignore`, `DATASET_GUIDE.md`, both Food-101 downloaders, `scripts/status_project.py`, `src/food_assistant/ml/cli.py`, and `src/food_assistant/ml/datasets.py`, plus untracked Phase 1–2.7 configs, reports, pipeline modules, scripts and tests. These were preserved.
- Python 3.11 environment: `.venv-311`; no dependency installation or environment changes.
- Existing raw Food-101 and a legacy `data/raw/food-101.tar.gz.part` (4,996,278,331 bytes) were present. Approval config sets `official_sha256: null`; therefore no official archive checksum can be claimed.
- ELT download policy remains `disabled`.

## FILES REVIEWED

- `promts/food101_elt_prompt_pack/prompts/00_GUARDRAILS.md`, `00_MASTER.md`, `04_PHASE_3_EXTRACT.md`.
- `artifacts/reports/elt/baseline.md`, `docs/food101_elt_architecture.md`, Phase 2.5/2.6/2.7 evidence and current ELT/approval configs.
- `scripts/download_food101.py`, `scripts/download_food101_parallel.py`.
- `src/food_assistant/data_pipeline/{atomic.py,config.py,load.py,splits.py}`, `src/food_assistant/ml/food101_verifier.py`, and existing resource-lock / data-safety tests.

## FILES CHANGED

- Added `src/food_assistant/data_pipeline/extract.py`: shared verified sequential/parallel HTTP acquisition.
- Updated `scripts/download_food101.py` and `scripts/download_food101_parallel.py` to use the shared Extract module and preserve their CLI entry points/options.
- Added `tests/unit/test_food101_extract.py` using scripted HTTP responses and synthetic tar archives only.
- Added this report. No config, raw image, existing archive, manifest, model, or approval file was changed.

## BEHAVIOR BEFORE → AFTER

- Sequential resume previously appended to the destination based mainly on file length and a partial `Content-Range` prefix. It now probes source length/validators, resumes only from a partial bound to the exact URL digest/ETag/Last-Modified/total with a matching partial SHA-256, and validates exact status and range boundaries before accepting bytes.
- HTTP retries are finite, classify transient transport errors and selected 429/5xx statuses, use exponential backoff with jitter, and honor bounded `Retry-After`. 404 and authorization failures are permanent. Errors and logs omit the full URL; each extraction logs a run ID.
- A server returning 200 for a resumed range or 416 causes a scratch full transfer or fresh source probe. Existing partial bytes are not appended to a full response. Changed ETags invalidate old partial evidence; new representation bytes are fetched separately while the previous partial is preserved.
- Parallel and sequential modes now use the same archive-target resource lock. Parallel chunks are reusable only with source/boundary/length metadata and a recomputed matching SHA-256; same-size corrupted chunks are fetched again. Chunks combine into a sibling temporary archive.
- Before publication, exact byte length, local SHA-256, tar readability and required Food-101 metadata members are checked. The complete archive is atomically published only when the target is absent; a versioned sidecar binds provenance and evidence. If `official_sha256` is null, metadata records `official_checksum_verified: false` and the local digest separately.
- Both CLIs require `--download`, `download.policy: approved_only`, and a matching `academic_approval: confirmed_by_user` HTTPS source record before network access. `--dry-run` makes no lock, staging or output writes. A raw tree is skipped only after its Phase 2.5 checkpoint/inventory/source fingerprint validate.

## TESTS EXECUTED

- `.venv-311/Scripts/ruff.exe check src/food_assistant/data_pipeline/extract.py scripts/download_food101.py scripts/download_food101_parallel.py tests/unit/test_food101_extract.py tests/unit/test_food101_data_safety.py tests/unit/test_food101_elt_config.py` — PASS.
- `.venv-311/Scripts/python.exe -m pytest tests/unit/test_food101_extract.py -q` — 27 passed. Cases cover 500 and timeout retry/resume (including probe read timeout), permanent 404/401/403, Range ignored, incorrect range start/end/total, 416 recovery, ETag changes, corrupt same-size chunk, verified chunk reuse, digest/tar failure, combine I/O failure, archive skip/conflict, Retry-After, parallel combine/range rejection, cross-process shared lock, explicit approval, disabled config, raw skip and dry-run.
- `.venv-311/Scripts/python.exe -m pytest tests/unit/test_food101_extract.py tests/unit/test_food101_data_safety.py tests/unit/test_food101_verifier.py tests/unit/test_food101_elt_config.py tests/unit/test_food101_clean_split.py tests/unit/test_food101_split_v3.py tests/unit/test_poc_preprocessing.py tests/unit/test_process_lock.py -q` — 103 passed.
- `.venv-311/Scripts/python.exe scripts/download_food101.py --dry-run` — PASS; reported `network_requested: false`, `writes_performed: false`, policy disabled.
- `.venv-311/Scripts/python.exe scripts/download_food101_parallel.py --help` — PASS; compatibility options shown; no network.
- `.venv-311/Scripts/python.exe scripts/download_food101.py` — PASS / `SKIPPED`; verified existing raw against Phase 2.5 evidence: 101,000 images, inventory SHA-256 `b91c327114b45c4fe2e626404a86c449891e6e258c832e67a7f2350aeb59adb9`, source fingerprint `f5cfdb7480c2650ed94c8624652ac595e4a7f038d660f4886869836632c56ed1`. No download/extraction was attempted.
- Real network download: NOT RUN, as required.

## SAFETY CHECK

- Raw verifier reported `SKIPPED` only after current file signatures, Phase 2.5 checkpoint and fingerprints matched. No raw/archive path was written by that command.
- The legacy `.part` remains present at its baseline size; no extraction, download or partial cleanup command ran. A legacy partial without trusted sidecar evidence is never reused solely by size; it remains preserved while any new attempt uses a separate generation path.
- No code path in this phase writes `data/splits/v1`, v2/v3 manifests, or model artifacts. No download, external network, training or model-weight operation was run.
- Downloader modes share the canonical archive resource lock; extraction to raw continues through Phase 2 staged loader and its independent raw lock.

## ACCEPTANCE

PASS for implementation and synthetic verification. Real acquisition was intentionally not run. Existing raw was verified read-only and Extract skipped.

## RISKS/OPEN ITEMS

- The approval record has no official SHA-256. A future successful download can prove its local SHA-256, expected length, HTTP representation evidence and tar structure, but cannot claim official checksum verification.
- The pre-existing 4.996 GB `.part` has no trusted Phase 3 resume metadata, so it will not be appended or accepted on size alone. It is preserved and may require substantial additional disk space if a future approved download is run.
- A failure after archive rename but before sidecar publication leaves an archive without complete provenance; the next run fails closed with a conflict rather than treating that file as complete.

## NEXT PHASE

Phase 4 — Transform.
