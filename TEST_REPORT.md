# Test report — 2026-10-31

## Latest midterm notebook checks — actual execution

Executed in existing CPU .venv, no package installation:

```powershell
& .\.venv\Scripts\python.exe scripts/build_midterm_notebook.py
# Subsequent rebuilds used --overwrite only for the newly-created notebook during development.
& .\.venv\Scripts\python.exe scripts/run_midterm_notebook.py
& .\.venv\Scripts\python.exe scripts/verify_midterm_notebook.py
& .\.venv\Scripts\python.exe -m ruff check scripts/build_midterm_notebook.py scripts/run_midterm_notebook.py scripts/verify_midterm_notebook.py
```

Final PASS: 13/13 code cells executed, 0 cell errors, 13 embedded PNG figures and 13
exported readable PNGs, 2 nonempty CSV tables, identical canonical/compatibility notebook,
valid notebook schema and code syntax. Four validation samples; no test image decoding.
HOG visualization vector exactly matches production extract_hog, dimension 8100.
CNN tensor normalization matches production transform; noise repeats with fixed seed.
Protected model/config/pointer/.env hashes unchanged before/after; Ruff passed.

Final execution evidence:
`artifacts/reports/midterm_visualizations/run_20261031T144001_929466Z/execution_summary.json`.
Notebook SHA256: `b19b732abaa1a6ff5275d07edd69a9a8362b23dad07c733f44d7c6c0ea2c7bc8`.

Controlled experiment on the selected hamburger validation reference: Gaussian noise
PSNR 22.4048 dB → Gaussian filter 30.6626 dB; salt-and-pepper 15.5393 dB → median
33.0317 dB. Full MSE/SSIM and alternative-filter measurements are in noise_metrics.csv.
These are image-restoration demonstration metrics, not new classifier evaluation metrics.
Saved comparison metrics were checked against the existing release JSON files.

Final plots were visually inspected. Initial development used the first sorted hamburger
validation image, which was an advertising statue; replaced it with a visually checked
food photo and reran all cells. No filtering by model correctness. Initial kernel history
profile warning was resolved using temporary workspace-local IPython/Matplotlib folders.
All final figures are explicitly embedded PNGs, independent of a GUI Matplotlib backend.

Not performed: training, test-set Evaluate, GPU inference, full app regression suite,
database connection/modification or production preprocessing changes. This task only
adds standalone notebook/scripts/report files; application source was reused read-only.

## Latest food photography / camera / dashboard verification

Actual final command:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_nutrition.py tests/unit/test_nutrition_catalog.py tests/unit/test_inference.py tests/integration/test_automatic_nutrition.py tests/integration/test_camera_flow.py tests/integration/test_diary_dashboard.py tests/integration/test_diary_presentation.py tests/integration/test_frontend_flow.py tests/integration/test_meal_repository.py tests/unit/test_meal_service.py tests/smoke/test_end_to_end.py -q
```

**39 passed, 0 skipped, 1 warning in 15.73 seconds.** Targeted Ruff check/format passed.
Existing warning: Starlette TestClient/httpx deprecation; no dependencies changed.

New coverage: injected JPEG camera capture follows normal inference and diary save;
missing/corrupt capture; source switching clears stale prediction/save state; no duplicate
prediction is posted; summary/daily macro aggregation matches persisted snapshots;
manual-energy-only and unknown foods retain null macros; empty aggregate totals remain
null. Existing Home/navigation/banner, recognition, diary, charts and model-info AppTest
flows passed. Existing test of production CPU inference through nutrition/diary also passed.
An initial new aggregate assertion compared numeric dashboard JSON to decimal-string
meal JSON; corrected the assertion to compare Decimal values, preserving API serialization.

Integrity checked: banner asset hash equals supplied original `.streamlit/3.jpg`; `.env`,
production pointer, production EfficientNet weights and checksum manifest unchanged.
PostgreSQL tests use the isolated `_test` database and rollback transactions.

Not performed: physical camera capture, browser permission prompt, browser viewport
screenshots, live deployment restart, Docker rebuild, full repository test suite, training
or Food-101 test-set evaluation. No Playwright/browser automation tool was available;
no new browser/dependency download was started. See UI_GUIDE.md for manual checks.

## Latest UI / Vietnamese labels / diary macro check

Actual commands from the repository root:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/integration/test_frontend_flow.py tests/integration/test_automatic_nutrition.py tests/integration/test_diary_dashboard.py -q
# 12 passed, 1 warning, 16.10s
& .\.venv\Scripts\python.exe -m pytest tests/integration/test_diary_presentation.py tests/integration/test_frontend_flow.py tests/integration/test_diary_dashboard.py -q
# 3 passed, 1 warning, 6.33s
```

The groups overlap. The new diary presentation test creates an isolated test meal with
a sourced curry item and an unmapped item, reads it back from PostgreSQL, asserts all
three persisted macros match rendered values, and checks unknowns remain unavailable.
It also checks Vietnamese portion/meal labels, that editing meal metadata still sends
the original backend enums and preserves item snapshots, dashboard/about rendering,
and the full app Home/hero/navigation rendering. Existing recognition save/change-food
regressions also pass. Targeted Ruff check/format passed. Only the existing Starlette
TestClient httpx deprecation warning remains.

No schema/data import or app DB edits were needed. No training, model evaluation or
Docker operations were performed. Browser visual inspection, mobile viewport inspection
and a restart of the user's existing frontend were not performed; styling and responsive
rules were implemented and the component flow was checked with Streamlit AppTest.

## Current Nutrition & Portion verification

Actually executed in the CPU `.venv`:

```powershell
& .\.venv\Scripts\python.exe -m pytest tests/unit/test_nutrition.py tests/unit/test_nutrition_catalog.py tests/unit/test_inference.py tests/integration/test_automatic_nutrition.py tests/integration/test_diary_dashboard.py tests/integration/test_frontend_flow.py tests/integration/test_meal_repository.py tests/unit/test_meal_service.py tests/smoke/test_end_to_end.py -q
```

**36 passed, 0 skipped, 1 warning, 19.31 seconds.** Warning: the installed Starlette
TestClient reports a deprecation warning for httpx; no dependency changes were made.
Targeted Ruff check over modified nutrition/API/UI/seed/scripts/test code also passed.

Coverage: exact manifest mapping and USDA source/unit extraction; idempotent catalog
rebuild; immutable reference merges; real-sourced Hamburger/Kem/Pizza S/M/L/custom
formula and macros; manual input provenance; null/unreviewed/missing references; invalid
numbers; seed version identity and preservation of existing default mappings; immutable
diary snapshots even after a reference changes; dashboard known/missing/manual counts;
unknown-only Home display; Streamlit food/portion/gram changes and save confirmation;
read-only verification script exercised against PostgreSQL TestClient API; mock pipeline;
single real production upload → nutrition → saved meal on CPU using a synthetic image.

The real smoke did not run training or evaluation and does not estimate accuracy.
Tests use the configured `_test` PostgreSQL database with transactions rolled back;
no application meals were rewritten. No Docker services were started/restarted.
Catalog preparation also ran successfully: 10 available classes, 0 unavailable.

Initial UI test failed because it implicitly selected production when the pointer became
available, and the initial default 3-second AppTest wait expired. The regression test now
explicitly selects mock and uses a 15-second timeout; separate CPU smoke checks real AI.
Final tests above pass. Protected production/.env files retained identical SHA256 hashes.

Not run: full repository test suite, training, Food-101 test-set evaluation, GPU inference,
post-import live deployed UI verification, or Docker rebuild. Application DB import and
restart commands are supplied in [NUTRITION_GUIDE.md](NUTRITION_GUIDE.md).

## Earlier test sessions

## Current POC/GPU code checks

Executed only the selected short tests, not a full suite or a food-model training job:

```powershell
.venv/Scripts/python.exe -m pytest tests/unit/test_poc_preprocessing.py tests/unit/test_gpu_training_helpers.py tests/unit/test_poc_protocol.py tests/unit/test_activation_config.py tests/unit/test_release_selection.py tests/smoke/test_end_to_end.py tests/integration/test_frontend_flow.py -q
```

**23 passed, 1 Starlette deprecation warning, 11.70 seconds.** Covers exact transform
matching to official torchvision presets without downloading weights, EXIF/RGB loader,
tiny CPU AMP-disabled accumulation math, weights-only-safe RNG checkpoint serialization,
atomic checkpoint replacement, candidate freeze/tamper detection, environment backup,
validation ranking, and Mock API/UI/PostgreSQL flows. No actual CNN was trained here.

`scripts/audit_poc.py` actually passed for all 10000 manifest rows (no full image-byte
hash/decode sweep in this run). `check_training_device.py` reported torch/torchvision
CPU builds and CUDA unavailable. No CUDA execution, official test evaluation or
real-model end-to-end result is claimed. Ruff passed. Long checks remain user-run.

## Latest user-directed short checks

Long/full suites are now user-run; no full suite or training was launched for this update.
New guards/ranking/dry-run are covered by the targeted commands recorded in
IMPLEMENTATION_STATUS.md. Historical full-suite results below do not imply that the
entire latest codebase has been retested. Full commands are in RUN_COMMANDS.md.

Executed in this update:

```powershell
.venv/Scripts/python.exe -m pytest tests/unit/test_process_lock.py tests/unit/test_release_selection.py tests/unit/test_experiment_dry_run.py -q
```

Result: **11 passed in 0.19s**. Ruff passed. Python AST and PowerShell parser checks
passed. Dry-run did not spawn jobs; the legacy-process guard detected the active HOG.
No full-suite result is claimed for these latest changes.

Results are execution evidence, not claims of food recognition quality.

| Executed command/check | Observed result |
|---|---|
| `pytest -q` (first integrated suite) | 38 passed, 1 Starlette TestClient deprecation warning |
| `pytest -m 'not integration' -q` (later suite) | 33 passed, 8 deselected |
| `ruff check src scripts tests migrations` | Passed before subsequent implementation edits; rerun at final verification |
| `alembic upgrade head` | Applied business schema to project DB |
| `alembic check` | No new upgrade operations detected |
| PostgreSQL seed run twice | Successful idempotent catalog/reference seed |
| Streamlit AppTest, all five pages, host API | Passed |
| Host Uvicorn `/health` | 200, database=true |
| Host Streamlit `/_stcore/health` | 200 |
| `docker compose ps` | PostgreSQL, backend, frontend all healthy |
| Live `scripts/smoke_test.py --model mock` against Docker | Passed upload, metadata retrieval, sourced nutrition, meal save, dashboard, update/delete |
| Docker frontend health | 200 |

Unit coverage includes invalid/corrupt/MIME-mismatched/oversize/pixel-limited images,
EXIF rotation, crop validation, unknown vs zero kcal, per-100g scaling/macros, invalid
grams, deterministic mock, production mock rejection, missing models and JSON contracts.
PostgreSQL coverage includes migrations, indexes, idempotent seed, rollback,
idempotency conflicts, invalid patches, missing nutrition and dashboard totals.

All three model methods have run train/save/load/evaluate smoke tests on synthetic
fixtures. CNN smoke tests used random initialization and one epoch, with artifacts
confined to temporary test directories. They do not count as pretrained food training.

Initial Docker attempts exposed an installed-package root path bug, then Docker Engine
stopped during parallel export. Root path was fixed, memory limits were added, and a
single shared image was rebuilt sequentially with locked runtime dependencies.
The later three-service health and HTTP smoke checks passed. No database reset occurred.

Dataset-dependent experiments and real checkpoint acceptance are reported separately
in MODEL_EVALUATION_REPORT.md and IMPLEMENTATION_STATUS.md when actually executed.

## Test groups

```powershell
# Requires ML extras for ML smoke tests; no network/pretrained weights used by them.
.venv/Scripts/python.exe -m pytest -m "not integration" -q
# Requires separate TEST_DATABASE_URL ending in _test.
.venv/Scripts/python.exe -m pytest -m integration -q
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ruff check src scripts tests migrations
```

The test DB is created if absent and migrations are applied; each test rolls back its
own data using savepoints. No user database is dropped or reset. Tests not run are
not marked pass. Native Windows PostgreSQL and CUDA training remain unverified.
