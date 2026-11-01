# Implementation status

## CURRENT MIDTERM NOTEBOOK — executed, CPU-only

Status: DONE. The user-linked nested notebook was empty (0 bytes). Filled it and added
the canonical `notebooks/midterm_image_processing_demo.ipynb`; both are identical with
executed outputs. Editing workflow and copy behavior are documented in notebooks/README.md.

- Seven sections: image representation/color spaces, geometry/pixel normalization,
  controlled denoising, faithful HOG visualization, existing HOG/SVM inference, saved
  CNN comparison, and Vietnamese midterm conclusions/limits.
- Exactly four validation samples used by the notebook. Hamburger 1010695 was selected
  after visual review; first sorted image was an advertising statue. Selection was by
  visual content, not model predictions. No manifest/split changes or test images opened.
- Reused production ImageService/CropService, extract_hog and build_transform. Asserted
  HOG visualize feature equality (8,100 dimensions), CNN normalization equality, fixed
  seed noise repeatability and before/after protected file hashes.
- Read existing comparison CSV and model metrics JSON; verified consistency. No model
  training, test-set evaluation, configuration change or database connection.
- Actual final run: all 13 code cells in a 23-cell notebook completed without cell errors.
  Verification script passed: 13 embedded PNG figures, 13 readable exported PNGs,
  two CSV tables, valid JSON summary, identical notebook copies and Python syntax.
- Final output: `artifacts/reports/midterm_visualizations/run_20261031T144001_929466Z/`.
  Figure descriptions and sources are in notebooks/README.md and the export README.
  Exported at 300 dpi; visually inspected denoising/HOG/comparison figures. Earlier
  development output folders remain; final handoff is explicitly identified.
- Files added: canonical notebook, filled compatibility notebook; build/run/verify
  scripts; notebook/export README files; new visualization outputs and this/test report.
- No library installs needed. CPU .venv already has nbformat/nbclient/ipykernel and
  image libraries. Initial runner attempted IPython history on a read-only user profile;
  fixed by using a workspace-local temporary kernel/profile/cache. Final run has no
  such error. Explicit PNG display ensures figures are embedded without a GUI backend.
- Morphology/segmentation/SIFT are not implemented in this notebook: scope is the HOG
  classical-recognition branch. Filters remain standalone demonstrations. No claim that
  denoising increases recognition accuracy or that POC results apply to all 101 classes.

Actual commands: `scripts/build_midterm_notebook.py`, `scripts/run_midterm_notebook.py`,
`scripts/verify_midterm_notebook.py` using `.venv/Scripts/python.exe`; targeted Ruff passed.
Next user step: open canonical notebook, select Python 3.11 .venv kernel and view existing
outputs or Run All. No application/backend restart is required for this notebook-only work.

## CURRENT UI PHOTO / CAMERA / DASHBOARD UPDATE

Status: DONE for implementation and targeted tests. Based on user reference images
`.streamlit/1.jpg`, `2.jpg`, `3.jpg`. Reference 3 is displayed as original food photography;
references 1/2 inform the card/chart layout only. No reference UI was reproduced wholesale.

- Home: photo hero, accessible start-recognition link, mint/white surfaces and warm accent,
  responsive stacked photo/copy layout. Source photo is copied byte-for-byte into
  `ui/assets/food_photography.jpg`; SHA256 matches original 3.jpg. Package data includes
  JPG/SVG assets so the installed app and Docker COPY src retain the banner.
- Recognition: upload/camera selector, native Streamlit camera input, responsive preview
  and guidance columns. Both inputs use the same decoder/upload/inference contracts.
  Switching sources or removing an image clears prior predictions/save confirmation.
- Dashboard: four kcal/protein/carbohydrate/fat cards with per-nutrient coverage counts;
  interactive energy trend, true meal-type kcal-share donut, and daily macro bars.
  No invented intake goals or progress percentages. Missing dates/values retain gaps.
- Backend: additive `nutrients` in dashboard summary/daily responses, totals/known/missing
  counts computed from saved meal snapshots. Existing API fields retained. No schema,
  Nutrition Engine, historical meal or model changes. No migration/seed required.
- Files changed: `ui/theme.py`, `ui/pages/{recognize,dashboard}.py`, photo asset and asset
  provenance README, `services/dashboard_service.py`, setuptools package-data config,
  camera integration test, dashboard tests, API docs, UI_GUIDE.md and status/test reports.
- Actual final verification: 39 passed, 0 skipped, 1 existing deprecation warning in 15.73s;
  Ruff passed. Includes injected JPEG camera capture, corrupt capture rejection, no-capture
  state, source-switch invalidation, diary save, macro snapshot UI, missing/manual
  nutrition, API CRUD and one real production CPU serving smoke using a synthetic image.
- Protected file SHA256 unchanged: `.env`, production pointer, EfficientNet model.pth
  and checksum.sha256. No training, test-set evaluation, downloads or container operations.
- Remaining deployment/manual checks: user restart backend (new aggregate fields) and
  frontend, inspect real browser desktop/mobile layout and actual camera permission/device
  capture. AppTest covers component flow; no physical camera/browser viewport was tested.
  See UI_GUIDE.md. Application DB and services were not modified/restarted by the agent.

## CURRENT UI UPDATE — mint theme, Vietnamese labels and diary macros

Status: DONE for repository implementation and targeted verification. Running services
were not restarted; refresh/restart Streamlit to load the new UI and theme configuration.

- Added light mint/cream healthcare-food visual style, responsive Home hero with original
  offline SVG illustration, soft cards, nutrition energy accent, visible keyboard focus
  and reduced-motion preference support. Functional pages retain a clean background.
- Vietnamese navigation and shared display labels for meal types and portions across
  recognition, diary filters/add/edit forms, saved rows and dashboard charts. API/database
  enum values are unchanged. Model technical JSON is now collapsed in an expander.
- Confirmed `meal_items.protein_g`, `carbohydrate_g`, `fat_g` and MealService persistence
  already exist. Diary now reads those stored snapshots, without recalculating historical
  values. Missing macros show unavailable; explicit zero stays zero. No migration or seed.
- Files: `.streamlit/config.toml`, `app.py`, `ui/theme.py`, `ui/labels.py`,
  `ui/assets/food_hero.svg`, home/recognize/diary/dashboard/about pages, nutrition editor,
  `tests/integration/test_diary_presentation.py`, this status and TEST_REPORT.md.
- Actually ran existing frontend/automatic-nutrition/diary regression group: 12 passed
  in 16.10s. Then new diary presentation + frontend + diary-dashboard: 3 passed in 6.33s.
  Groups overlap; these counts are not additive. Ruff check and format passed.
- New test verifies persisted macros and unknown values in the rendered diary table,
  Vietnamese enum options/headings, metadata edit preserving snapshots and original API
  enum values, dashboard/about rendering, and full app navigation/Home hero rendering.
  Verification used Streamlit AppTest and isolated PostgreSQL test transactions, not a
  browser screenshot or manual viewport inspection. No application meals were changed.
- No training, test-set evaluation, downloads, checkpoint/pointer/.env changes or Docker
  operations. Next: restart only the running frontend if needed, then inspect desktop/
  mobile layout and horizontal scrolling of the diary table in the browser.

## CURRENT — Nutrition & Portion Engine, 2026-10-31

This section supersedes the historical CUDA/training status below. Read-only inspection
confirmed production pointer `efficientnet_b0/poc10-rtx3050-v1` and its already-saved test
accuracy 0.898. No training or test-set evaluation was performed during nutrition work.

| Phase | Status | Actual evidence / remaining action |
|---|---|---|
| Inspect nutrition / labels / schema | DONE | Exact 10 manifest labels; existing SQLAlchemy schema supports required snapshots. Initially only pizza had a nutrition mapping. |
| Sourced catalog | DONE | 7 USDA SR Legacy + 3 FNDDS records extracted from official archives, source IDs/versions/URLs and recipe caveats retained. All 10 have kcal and macros. |
| Portion and automatic calculation | DONE | Per-food S/M/L, explicitly design assumptions; custom grams; Decimal rounding; missing values stay null; manual source user_input. |
| Backend / diary / dashboard | DONE | Existing tables reused; no new migration. Version-aware append-only seed, historical snapshots retained, known/missing/manual counts exposed. |
| Streamlit | DONE | Reactive estimate, source/record/caveats, medium default, reset stale grams/manual values on food/portion change, confirmation before save, unknown total shown explicitly. Tested with AppTest. |
| Targeted integration verification | DONE | 36 passed in 19.31s; actual PostgreSQL test database, rollback isolation, mock flow and one CPU production inference upload → nutrition → diary. Ruff passed. |
| Apply catalog / restart running application | NOT STARTED | Intentionally left as user-run commands in NUTRITION_GUIDE.md. Application DB and running containers not mutated. |

Created files: `configs/nutrition_poc.json`, `src/food_assistant/nutrition/{__init__,catalog,prepare}.py`,
`services/portion_service.py`, `scripts/prepare_nutrition.py`, `scripts/seed_nutrition.py`,
`scripts/check_nutrition.py`, `data/nutrition/{coverage,label_catalog}.json`,
`tests/unit/test_nutrition_catalog.py`, `tests/integration/test_automatic_nutrition.py`,
`NUTRITION_GUIDE.md`.

Updated: nutrition catalog/mappings; persistence seed and seed_database script; nutrition
service; API and response schemas; dashboard service; recognize/nutrition editor/diary/
dashboard/home UI and disclaimer; frontend/smoke regression tests; README, API docs,
nutrition assumptions and this/test status report. Existing unrelated work was preserved.

Commands actually executed: `scripts/prepare_nutrition.py` (10 available, 0 unavailable),
the selected pytest command recorded verbatim in TEST_REPORT.md (36 passed, 0 skipped),
and targeted Ruff check/format. Initial UI test exposed an implicit production selection
and the default 3s AppTest timeout; fixed the test to explicitly choose mock and allow
15s. Final real-serving check is separate and explicitly CPU-only, using one synthetic
image rather than Food-101 test data. No model-quality metric was calculated.

Dependencies: current catalog is ready without re-downloading anything. Rebuilding it
requires the local USDA source archives. Official FNDDS nutrition archive (~4 MB) was
obtained to fill the three missing composite-food references; no Food-101 download.
Historical pizza source record and original provenance file were retained.

Limitations: all 10 classes have representative references, not recipe-specific measured
composition. Caesar reference excludes dressing; ice cream is vanilla; curry excludes
rice; sushi is unspecified. Other recipes/extra toppings may differ. All S/M/L weights
are project assumptions, never image measurements or official USDA serving sizes.
Out-of-catalog foods or missing mass remain unavailable. No application DB import or
browser/live-service verification after restart has been performed in this phase.

Integrity: SHA256 before/after confirmed production pointer, EfficientNet checkpoint,
checkpoint checksum manifest and `.env` unchanged. No new migration, model rewrite,
evaluation, dataset download, container operation or historical meal rewrite.

Next: follow NUTRITION_GUIDE.md to seed nutrition, restart the existing deployment and
run the read-only 12-case Hamburger/Kem/Pizza verifier, then confirm/save on the UI.

## Historical implementation snapshots

The following entries describe earlier sessions, before the production model was trained
and activated. They are retained as history, not current model status.

CUDA setup follow-up: user-created .venv-cuda is isolated (include-system-site-packages=false),
but its installed torch is a CPU build. No package install was performed by the agent.
The prior unqualified torch==2.9.1 install could retain CPU wheels; the guide now pins
torch==2.9.1+cu126 / torchvision==0.24.1+cu126 explicitly and supplies a user-run repair
command for the already-installed environment. Device check now prints interpreter/package
paths and distinguishes a CPU wheel from a CUDA-wheel driver failure. GPU remains unverified.

## CURRENT AI POC CHECKPOINT — 2026-10-31

Scope: finish Food-101 POC 10 classes before full-101. Long work remains user-run.
Read POC_REAL_AI.md for exact RTX 3050 Ti commands and independent steps.

- Read-only inspection: original HOG worker 10308/wrapper 2992 still active, >100 min
  elapsed at this inspection; no completed validation result, trained/sealed artifact
  or production pointer. No duplicate training or checkpoint overwrite performed.
- Fast POC audit actually passed: 6000 train / 1500 validation / 2500 test, 10 classes,
  fixed 0..9 mapping, exact official test membership, train/val from original train,
  consistent split CSVs and stored cross-split checksums. This run did not decode/hash
  all images again. Saved data/splits/v1/poc_contract_audit.json.
- GPU hardware read: RTX 3050 Ti Laptop 4096MiB, driver 566.36. Current environment:
  torch 2.9.1+cpu / torchvision 0.24.1+cpu; CUDA unavailable. No CUDA install attempted.
- Implemented GPU POC profiles: batch 4, accumulation 4, AMP, 2 head + max 8 fine-tune
  epochs, validation early stopping. Dedicated .venv-cuda commands preserve active CPU env.
- Implemented atomic CNN checkpoints, optimizer/scaler/best/history/RNG resume and
  resume-contract checks. Image loader applies EXIF orientation before RGB transforms.
- Future HOG runs log per-C fit time/iterations/convergence and save a pending best
  candidate; existing active worker keeps its previously loaded code unchanged.
- Implemented frozen three-candidate protocol with immutable weights/labels/preprocess/
  validation metadata before test; CPU inference evaluation shared across all methods.
- Evaluation exports measured metrics, confusion matrices, per-class reports, metadata,
  predictions and completion marker; selection exports comparison CSV/Markdown and pointer.
- Prepared explicit activation script with .env backup and mock disabled only after a
  measured full-POC release is available. API/UI production alias/provenance updated.
  Real HTTP smoke now requires a real validation/local photo.
- NOT RUN: real CNN training, CUDA/AMP GPU execution, official evaluation, activation,
  real checkpoint end-to-end. No actual accuracy/F1 or winning model is claimed.
- Short tests only: initial targeted run 14 passed (7.09s); final result recorded below.

Verified current code checks: 23 selected tests passed in 11.70s (one Starlette warning),
covering preprocessing equality to official weight transforms, EXIF, tiny optimizer math,
RNG/atomic saves, freeze integrity, activation backup, selection and Mock end-to-end.
Ruff passed; no full suite/real training was run. Optional HOG64 profile is prepared for
a new user-chosen run only; it never replaces or modifies the active HOG128 worker.
Use scripts/poc_ai.ps1 for independent stages, POC_REAL_AI.md for exact CUDA commands.

Historical CPU instructions/checkpoints below do not override this POC/GPU workflow.

## CURRENT EXECUTION POLICY — user update 2026-10-31

Do not autonomously launch downloads, training, full test suites or deployment builds.
Focus on code fixes and independent scripts; provide PowerShell commands for the user.
Do not duplicate existing processes. `RUN_COMMANDS.md` is the ordered command reference.

Current inspection: Docker postgres/backend/frontend all healthy; dataset has 101,000
local images; POC manifests already prepared. No real trained/sealed artifact or
production pointer exists at this checkpoint. HOG v2 is still fitting, not TRAINED.
Orchestrator PIDs 11868/15760 were stopped to prevent automatic CNN/evaluation chaining;
the active HOG worker PID 10308 and wrapper 2992 were deliberately preserved.
An older pip installer was observed (3540/26680); it was not restarted or terminated.
These PIDs are a snapshot; use `scripts/status_project.py --docker` for current state.

New code: `scripts/project.ps1` defaults to read-only Status; per-model process/file
locks prevent duplicate training; the aggregate experiment CLI is dry-run by default,
requiring explicit `--execute` for long work. Release selection validates matching
test provenance and exports the measured comparison CSV. Changes do not modify code
already loaded by the currently running legacy HOG process.

The historical checkpoints below are retained; this section governs future execution.

### Code prepared for user-run steps

- `scripts/status_project.py`: read-only process/dataset/artifact report; optional Docker ps.
- `scripts/project.ps1`: individual PowerShell steps, Status by default; no background launch.
- `scripts/run_poc_experiment.py`: default dry-run; long chain requires explicit --execute.
- `utils/process_lock.py`: kernel lock + legacy process detection, scoped by model/output root.
- Training rejects an existing run folder before reading dataset; HOG resume explicitly rejected;
  CNN resume avoids unnecessary ImageNet download. Active legacy workers remain unchanged.
- `ml/release_selection.py`: common test provenance/labels/sample-count checks and validation ranking.
- `select_production_model.py`: writes measured comparison CSV along with production pointer.
- Full command order and current skip conditions are in RUN_COMMANDS.md.
- No downloads, training, full tests, Docker starts or builds were launched in this update.
- Executed short verification: `pytest tests/unit/test_process_lock.py
  tests/unit/test_release_selection.py tests/unit/test_experiment_dry_run.py -q`:
  **11 passed in 0.19s**. Ruff and Python/PowerShell syntax checks passed.
- Dry-run command verified to print commands without spawning subprocesses or creating
  a run directory. Process guard correctly detected existing HOG PIDs 2992/10308.
- `git diff --check` passed (Git emitted only LF/CRLF normalization notices).
- Full suite and real training/evaluation were not re-run; user commands are ready.

Updated: 2026-10-31 (Asia/Ho_Chi_Minh). Solo developer; no team handoff.

## Initial inspection

- Existing repository is a committed scaffold; Git working tree was clean.
- PLAN.md, README, docs, diagrams, configs, Python modules and tests inspected.
- Five placeholder Streamlit pages; no API, business ORM, implemented pipelines or executable tests.
- Architecture change authorized by the new master prompt: Streamlit calls FastAPI over HTTP.
- User confirmed Food-101 academic approval in this conversation. Official source checked;
  usage/attribution will be recorded before download. No food images downloaded yet.
- Python 3.11.0; RAM 15.6 GiB; workspace disk free 77.2 GiB.
- NVIDIA RTX 3050 4 GiB present. Existing torch 2.9.1 is CPU-only (CUDA unavailable).
- Docker installed; engine initially stopped. No local PostgreSQL executable found.

## Progress

| Phase | Status | Evidence / next action |
|---|---|---|
| 1 Environment | IN PROGRESS | Project .venv created using installed system packages; dependency installation running after sandbox network rejection |
| 2 Database | NOT STARTED | Implement schema, migration and isolated integration DB |
| 3 Dataset | NOT STARTED | Food-101 academic approval confirmed by user; implement source/rights manifest and downloader |
| 4 Training | NOT STARTED | No model is TRAINED |
| 5 Evaluation | NOT STARTED | No experimental metrics |
| 6 Inference | NOT STARTED | Explicit deterministic mock first; real artifact loader next |
| 7 Backend | NOT STARTED | Implement API, validation, errors, transaction boundaries |
| 8 Nutrition | NOT STARTED | No verified nutrition records yet; unknown values remain NULL |
| 9 Frontend | NOT STARTED | Replace placeholder pages with API client flows |
| 10 Integration | NOT STARTED | Run upload → mock → confirm → save → dashboard → update/delete |
| 11 QA | NOT STARTED | Replace placeholder tests with executed assertions |
| 12 Deployment | NOT STARTED | Compose postgres/backend/frontend and Windows host commands |
| 13 Documentation | IN PROGRESS | This status tracks observed evidence, not promised completion |

## Commands executed

- `git status --short`: clean initially.
- `nvidia-smi`: RTX 3050 4096 MiB, driver CUDA capability 12.7.
- `python` psutil/shutil/torch inspection: 15.6 GiB RAM, 77.2 GiB free, CUDA false.
- `docker info`: failed, engine not running (also sandbox config permissions).
- `.venv` dependency install: initial network denied; retried with required escalation.

No application, database, dataset or model acceptance criterion is marked DONE yet.

## Verified progress after implementation

The initial observations above are retained as history; the following supersedes them.

- DONE (code and executed tests): PostgreSQL migration `cba28587a601`, 8 business tables,
  constraints/indexes, isolated test DB, idempotent seed, transactional meal CRUD.
- DONE (mock integration): FastAPI upload/prediction/foods/nutrition/meals/dashboard/models,
  versioned Pydantic + JSON Schema, explicit mock flags, production mock prohibition.
- DONE (tested implementation): image decode/MIME/pixel/EXIF checks, normalized manual crop,
  snapshot kcal/macros, unknown values, retry-safe meal creation and rollback.
- DONE (live host checks): Uvicorn started on 8000, Streamlit on 8501, `/health` returned 200,
  Streamlit `/_stcore/health` returned 200. All five UI pages passed AppTest with live API.
- Nutrition: one real USDA SR Legacy profile FDC 170317, 268 kcal/100g; sourced and extracted
  from the official CSV archive. Mapping scoped to cooked regular-crust cheese pizza;
  S/M/L masses are explicitly declared demo conventions, never image measurements.
- Training/evaluation code: all three pipelines completed train → save → load → evaluate
  smoke tests on synthetic, non-food test fixtures. NOT food-trained and NOT production eligible.
- `pytest -q`: 38 passed, 1 Starlette deprecation warning; includes PostgreSQL E2E and 3 ML smoke runs.
- `alembic check`: no new upgrade operations detected.
- IN PROGRESS: full Compose backend/frontend build and deployment verification.
- IN PROGRESS: official Food-101 HTTP Range download (149 resumable chunks); sequential
  attempt timed out after ~54 MiB. No complete real dataset or formal trained model yet.
- Academic approval: explicitly confirmed by user; official source/citation recorded in configs.
- BLOCKED for formal training/evaluation until the real dataset is fully downloaded and audited.

Files: `src/food_assistant/{api,domain,services,persistence,ml,ui}`, `scripts/`, `tests/`,
`contracts/`, migration, Dockerfile/Compose, nutrition provenance and configs.

## Deployment acceptance checkpoint

- DONE: three-service Compose mock deployment. `docker compose ps`: all three healthy.
- DONE: live HTTP `scripts/smoke_test.py --model mock`, including real sourced pizza
  estimate, PostgreSQL meal creation, dashboard, metadata edit and own-record deletion.
- DONE: Docker frontend health returned 200.
- Runtime dependency closure recorded in requirements-runtime-lock.txt (63 packages).
- Docker recovery: application root path fixed; Engine stopped during a parallel build;
  recovered without volume reset; shared image rebuilt sequentially using the lockfile.
- Remaining formal ML phases are in progress; dataset download/extraction must finish.

## Complete-suite and dependency checkpoint

- Latest complete suite: `pytest -q`: **41 passed**, one Starlette deprecation warning,
  167.20 seconds, including PostgreSQL integration and all three synthetic ML pipelines.
- Installed the declared `[ml,dev]` extras into the project venv. Imports verified;
  full installed dependency closure recorded in requirements-lock.txt (97 packages).
- Runtime Docker uses a separate 63-package runtime lock. The development venv reuses
  pre-existing system packages; the clean Docker runtime was also built and smoke-tested.
- Actual dataset archive has downloaded; extraction is in progress. Do not confuse
  expected source counts with local audit completion.
- Declared CPU experiment budget before any real test evaluation: POC 10 classes,
  two CNN epochs (one head + one fine-tune), batch 4, seed 42. Not a full-101 benchmark.

## Real dataset / experiment checkpoint

- DONE: official archive downloaded/extracted, **101,000 actual local JPEG images**.
- Archive SHA256: `d97d15e438b7f4498f96086a4f7e2fa42a32f2712e87d3295441b2b6314053a4`.
  No official SHA256 was published, so independent official checksum verification is false.
- Archive `license_agreement.txt` inspected: scientific fair use only; other uses require
  negotiating with image owners. No redistribution permission is claimed.
- DONE: POC manifest + labels + metadata + quality report: **10,000 images / 10 classes**;
  6,000 train, 1,500 validation, 2,500 unchanged official test. All selected images decode.
- Exact duplicate count: 1 within a split; cross-split checksum leakage check passed.
- IN PROGRESS: actual HOG+SVM training on all 6,000 POC train samples; validation C tuning.
- Next: pretrained ResNet18/EfficientNet-B0 training, then frozen-candidate test evaluation.
- Run logs: `artifacts/reports/poc-v1-cpu-budget-v1/`; orchestrator:
  `python -u scripts/run_poc_experiment.py --run-id poc-v1-cpu-budget-v1`.
- CUDA remains unavailable in this installed CPU build; the declared short CPU protocol
  is used without silently changing architectures or touching the official test split.

## Baseline solver adjustment before test evaluation

- Initial dual SVM attempt halted after >6 minutes of CPU work and ~1.95GB RSS,
  before any C validation result or test evaluation. Incomplete artifact retained as
  untrained under run `poc-v1-cpu-budget-v1`; never eligible for production.
- New declared run `poc-v1-cpu-budget-v2-primal` uses LinearSVC primal solver,
  max_iter=2000, tol=0.001; same HOG/class set/samples/C grid. CNN budget unchanged.
- This is an optimizer implementation adjustment, not test-driven hyperparameter tuning.
- Added Streamlit upload/confirm/save/image-change integration test: **1 passed**,
  injected JPEG/PNG bytes + PostgreSQL TestClient API; no browser automation claim.
