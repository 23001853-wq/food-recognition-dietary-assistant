# Food-101 ELT runbook

This runbook covers the repository's offline ELT runner. It does not authorize a download, a full-data rewrite, model training, or replacing a published split. The checked-in ELT configs disable downloads.

## Components and stage contracts

The runner executes the requested DAG prefix in this order:

```text
Inspect → Extract → Load → Quality → Transform → Validate → Report
```

| Stage | Contract | Implementation / evidence |
|---|---|---|
| Inspect | Validate config, roots and existing source/checkpoint evidence | `src/food_assistant/data_pipeline/pipeline.py`, `config.py` |
| Extract | Reuse a source-bound archive or fetch only under explicit approved policy; bounded retry/Range resume/checksum | `extract.py`; HTTP behavior tested with fake responses |
| Load | Safely extract into a new staging tree and publish only after validation | `scripts/download_food101.py::safe_extract`, `load.py`, `atomic.py` |
| Quality | Verify raw layout, metadata and image integrity; fail closed on invalid/incomplete evidence | `src/food_assistant/ml/food101_verifier.py` |
| Transform | Build only a compatible, versioned output from validated inputs | `transform.py`, `splits.py`, `split_v3.py` |
| Validate | Read published manifests with their matching validator/consumer | `validate_v3.py`, `ManifestDataset` |
| Report | Record stage status, evidence paths and fingerprints atomically | `pipeline.py`, report directory under `artifacts/reports/elt/` |

No data warehouse is required. Train-time resizing, tensor conversion and normalization happen in the ML Dataset/transform path; ELT does not create a second copy of every image.

## Legacy script to current module map

| Existing entry point | Current module/contract | Notes |
|---|---|---|
| `scripts/download_food101.py` | `data_pipeline.extract.download_sequential`, `load.py`, `safe_extract` | Direct legacy entry point. Follow approval config; do not target an existing raw tree for a fresh extraction. |
| `scripts/download_food101_parallel.py` | `data_pipeline.extract.download_parallel` | Parallel Range chunks are source-bound and verified before archive publication. |
| `scripts/verify_dataset.py` | `food_assistant.ml.food101_verifier.audit_food101` | Select audit scope carefully; a full audit can read/decode/hash the complete dataset. |
| `scripts/build_clean_food101_split.py` | `data_pipeline.splits.build_clean_split` | Full clean v2 builder; output is immutable/versioned. |
| `scripts/build_food101_split_v3.py` | `data_pipeline.split_v3.build_training_split_v3` | Full v3 builder; validates v2 source and writes only a new v3 target. |
| `scripts/validate_food101_full_v3.py` | `data_pipeline.validate_v3` | Validation of an existing full-v3 artifact. |
| `scripts/transform_food101.py` | `data_pipeline.transform.build_poc_transform` or `splits.build_clean_split` | Builder depends on scope and version; runner refuses unsupported pairs. |
| `scripts/run_food101_elt.py` | `data_pipeline.pipeline.run_pipeline` | Orchestrates the stages and writes run state/report. |

Supported routes are explicit: `configs/food101_elt_poc_v2.yaml` → POC `poc-v2`; `configs/food101_elt_full_v2.yaml` → full `v2`; `configs/food101_elt_full_v3.yaml` → full `v3`. The legacy default `configs/food101_elt.yaml` declares POC + `v2`, which has no compatible Transform builder and must remain `BLOCKED`; there is no silent fallback.

## Windows / Python 3.11 setup

Use the repository environment if it already exists:

```powershell
.\.venv-311\Scripts\python.exe --version
.\.venv-311\Scripts\python.exe -m pytest --version
.\.venv-311\Scripts\python.exe -m ruff --version
```

Expected Python is 3.11. If the environment is missing, create a separate `.venv-311` and install dependencies only through the repository's approved dependency process. Do not replace another environment, install into system Python, or download dataset/model files as part of setup.

## Preflight and execution

Read-only CLI checks:

```powershell
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --help
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --config configs\food101_elt_poc_v2.yaml --stage all --dry-run
```

The default config is intentionally expected to block Transform:

```powershell
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --stage all --dry-run
```

A real run may write versioned output and report/state files. Before starting one, check the selected config, source evidence, output target, download policy and available resources; obtain the data-operation approval required for the target. Example commands below are operational examples, not runs performed during Phase 6:

```powershell
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --config configs\food101_elt_poc_v2.yaml --stage all
.\.venv-311\Scripts\python.exe scripts\run_food101_elt.py --config configs\food101_elt_poc_v2.yaml --stage all --resume --run-id <run-id>
```

The command does not enable downloading by itself. A missing raw tree with download policy disabled returns `BLOCKED`. Never change the approval file just to bypass this gate.

## Resume, retry and recovery

- Retry is bounded by config. Extract validates status, range boundaries, source validators and chunk digest before reuse. A same-sized partial is not sufficient evidence.
- Run state is versioned and tied to config/run identity; malformed, stale or mismatched evidence fails closed.
- Resume validates completed-stage evidence/fingerprints. A filename's existence does not make a stage complete.
- Output directories are staged on the target filesystem, validated, then published only if the version target does not exist. Same evidence may be validated and skipped; different evidence at the same version is a conflict.
- On `BLOCKED`, `FAIL`, stale checkpoint or conflict, preserve the report and inspect the stated stage/reason. Do not delete lock, staging, checkpoint or output files manually while a process may still be active.
- After confirming all writers have exited, retain the diagnostic artifacts. A retry should use the same validated input/config for an idempotent attempt, or a new output version for changed inputs/config. Do not overwrite a published version.
- Rollback means selecting the prior known-good version/config in the consumer. It does not mean deleting or rewriting raw, v1, v2, v3, POC manifests, or model artifacts.

## Artifacts and reports

- Raw source: `data/raw/food-101/`; immutable after publication.
- Versioned manifests: `data/splits/<version>/`; do not replace an existing version.
- Source approval/checkpoint and verifier evidence: under configured `configs/` and `data/reports/elt/` paths.
- ELT execution state and JSON/Markdown report: `artifacts/reports/elt/<run-id>/`.

Review the report's config hash, source fingerprint, stage states, skip reasons, quality findings, output fingerprint and any failure traceback. `PASS` is a stage/report result only for the scope actually executed; synthetic tests are not full Food-101 validation.

## Testing and known limitations

### Phase 6.1 regression commands

From the repository root in PowerShell:

```powershell
.\.venv-311\Scripts\python.exe -m pytest tests/unit -q
.\.venv-311\Scripts\python.exe -m pytest tests/integration -q -rs
.\.venv-311\Scripts\python.exe -m pytest tests/integration/test_food101_elt_http.py -q
.\.venv-311\Scripts\python.exe -m ruff check scripts src tests
```

The loopback integration binds an ephemeral server to `127.0.0.1`, builds a 1-class/3-image archive under pytest's temporary workspace, and uses the real `scripts/run_food101_elt.py` entry point plus real downloader, safe extractor, verifier, POC builder, manifest reader and report writer. It injects a bounded HTTP 503 and, in a separate case, a truncated response to exercise retry and Range resume. A malicious archive case verifies that traversal is rejected before raw publication. The fixture approval is patched only inside the test process; it does not alter `configs/dataset_approval.json` or relax the production HTTPS/approval policy.

### PostgreSQL integration setup

PostgreSQL tests require a **dedicated local test database** whose name ends in `_test`; the session fixture refuses other database names. It connects to the `postgres` maintenance database to create the test database if needed, runs Alembic migrations, and tears down test transactions. Use a local test role with only the needed privileges (`CONNECT`, schema permissions, and `CREATEDB` only if the fixture must create the database). Never point this variable at production, shared, or application data.

Set the URL from a local secret source for the current PowerShell session only; do not put a real password in repository files, command history, screenshots or reports:

```powershell
$env:TEST_DATABASE_URL = "postgresql+psycopg://food_test_user:<URL-ENCODED-SECRET>@127.0.0.1:5432/food_assistant_test"
.\.venv-311\Scripts\python.exe -m pytest tests/integration -q -rs
Remove-Item Env:TEST_DATABASE_URL
```

If PostgreSQL requires credentials that are unavailable, tests are **BLOCKED**, not passed. Docker is an alternative only when its daemon is already available and the test service is isolated; do not install or reconfigure PostgreSQL/Docker as part of a routine test run.

#### Windows and DBeaver workflow

1. In DBeaver, connect only to the local development server (`127.0.0.1`, port `5432`) using an already authorized session. Do not copy a saved password into chat, a report, or a checked-in file.
2. Confirm the server and current connection with read-only queries:

   ```sql
   SELECT version(), current_user, current_database();
   SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname;
   ```

3. Use or create the dedicated `food_assistant_test_elt` database. Do not point the tests at an application or production database. The integration fixture requires a database name ending in `_test`; this name satisfies that guard.
4. Use a dedicated login that owns the test database and can run its migrations. If the fixture must create the database itself, the login also needs `CREATEDB`; otherwise an administrator can create the isolated database and assign ownership, avoiding that extra privilege. Do not grant `SUPERUSER` or `CREATEROLE`.
5. In DBeaver, inspect the test database's schemas and tables after the migration test. The fixture applies Alembic migrations and wraps per-test data in rollback transactions. Keep the test database for reruns; no cleanup command should drop it.
6. Set `TEST_DATABASE_URL` only in the PowerShell process that runs pytest. Keep the password in a local secret source and never echo the expanded URL:

   ```powershell
   $env:TEST_DATABASE_URL = '<driver URL for the dedicated local test role/database>'
   .\.venv-311\Scripts\python.exe -m pytest tests/integration -q -rs
   Remove-Item Env:TEST_DATABASE_URL
   ```

Authentication errors should be resolved through DBeaver's normal credential prompt or the local PostgreSQL administrator. Do not retrieve saved passwords, change an existing account's password, or disable server authentication. For migration errors, confirm the URL database name ends in `_test`, the role owns that database, and the expected schema is writable; then rerun only against that isolated database.

### Recovery details

- Safe Load creates a sibling `.staging-*` directory while holding its resource-scoped OS lock. A retry removes orphan staging directories with that exact reserved prefix only after acquiring the lock and confirming the destination is absent. Symlinked or non-directory entries are left untouched.
- A failed archive validation or traversal error removes the current staging tree and leaves the destination unpublished. The orchestration regression suite now hard-stops a child process after the real POC Transform atomically publishes its manifest but before `run_state.json` records `PASS`. Resume reruns the Transform builder, validates the existing fingerprint/config/manifest, returns `SKIPPED`, validates the dataset reader, and leaves the published manifest bytes unchanged. The process synchronization uses an explicit stdout handshake, not a timing sleep.
- The downloader removes the source-bound partial prefix it consumed only after the completed archive has been published and verified. Untrusted legacy partial files without matching sidecar evidence are preserved.
- Lock files are stable lock resources; their existence alone does not mean the lock is held. Check process ownership by attempting the lock through the supported code path, not by deleting the lock file.
- Rollback uses the prior known-good manifest/config selected by the consumer. Do not delete the new or prior published output to simulate rollback.

Targeted unit and integration tests use temporary files, synthetic archives and fake HTTP responses. PostgreSQL-backed application tests require `TEST_DATABASE_URL`; Phase 6.1 will report their actual execution state separately. Do not point tests at a user/application database.

Run the post-publication crash/recovery test with:

```powershell
.\.venv-311\Scripts\python.exe -m pytest tests/unit/test_food101_pipeline_orchestration.py::test_resume_recovers_after_transform_publish_before_run_state_update -q
```

The test uses only `tmp_path`; it does not kill the PostgreSQL service or touch published Food-101 data. This verifies one publication/state boundary. It does not claim crash coverage for every atomic report replacement or for full Food-101 execution.

Phase 6.1 uses Python 3.11 on Windows. Full Food-101 read-only verification is not included; a full audit can decode/hash all images and remains outside this release-blocker remediation. The legacy default POC+v2 config remains unsupported for Transform; choose a supported versioned config explicitly.
