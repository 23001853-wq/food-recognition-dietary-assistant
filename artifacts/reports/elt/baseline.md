# Phase 0 Baseline — completion attempt

- **Checked:** 2026-11-17 (Asia/Saigon)
- **Repository:** `23001853-wq/food-recognition-dietary-assistant`
- **Branch / SHA:** `main` / `2ead0dbd472a97978205ecb907e8a342b43f9fc9`
- **Acceptance:** **PASS** — required tests, verifier regression tests, real project transforms, and synthetic Dataset/DataLoader checks pass.
- **Scope:** Phase 0 verifier defect fix only. No Phase 1–6 work, raw dataset writes, downloads, or real-data training. The required test suite's synthetic CPU smoke-training cases were run as instructed.

## 1. Executive summary

The current `.venv-311` executable is Python 3.11.9 with pip 24.0; imports succeed for pytest 9.2.0, torch 2.15.0+cpu, torchvision 0.29.2+cpu, Pillow 12.3.0, and NumPy 2.4.6. The required unit suite passes 7 tests. A real project torchvision transform and synthetic `ManifestDataset`/PyTorch DataLoader smoke test also pass. The initial verifier failure was caused by a hard-coded `count == 1000` in the inventory criterion. It now derives the required per-class inventory from the official per-class train/test expectations (750 + 250), with an added regression test proving a missing image still fails the audit. Raw Food-101 and POC v1 evidence from the prior read-only check remains unchanged in this report; the old full audit still concludes `INCOMPLETE`, with two exact duplicate hashes crossing official train/test.

This continuation changed only the verifier implementation, its directly related unit tests, and this report. No configuration, raw data, archive, POC manifest, downloader, or other ML component was changed.

## 2. Environment evidence

| Check | Command | Expected | Actual | Status |
|---|---|---|---|---|
| Git status | `git -c "safe.directory=C:/Users/User/MyProjects/food-recognition-dietary-assistant" status --short` | Preserve and report existing work | Existing modifications: `.gitignore`, `DATASET_GUIDE.md`, `scripts/status_project.py`, `src/food_assistant/ml/datasets.py`; untracked `data/reports/`, `promts/`, `scripts/verify_food101.py`, `src/food_assistant/ml/food101_verifier.py`, `tests/unit/test_food101_verifier.py` | PASS (read only) |
| Branch / SHA | corresponding Git read-only commands with temporary `safe.directory` | Current revision | `main` / `2ead0dbd472a97978205ecb907e8a342b43f9fc9` | PASS |
| Interpreter | `.\.venv-311\Scripts\python.exe --version` | Python >=3.11,<3.12 | Python 3.11.9 | PASS |
| pip | `.\.venv-311\Scripts\python.exe -m pip --version` | venv pip | pip 24.0, `.venv-311` | PASS |
| Dependencies | `.\.venv-311\Scripts\python.exe -m pip show pytest torch torchvision Pillow numpy` | Packages available | pytest 9.2.0; torch 2.15.0+cpu; torchvision 0.29.2+cpu; Pillow 12.3.0; NumPy 2.4.6 | PASS |
| Import verification | `.\.venv-311\Scripts\python.exe -c "import sys, pytest, torch, torchvision, PIL, numpy; ..."` | All imports succeed | Python 3.11.9; imports and versions printed successfully: pytest 9.2.0, torch 2.15.0+cpu, torchvision 0.29.2+cpu, Pillow 12.3.0, NumPy 2.4.6 | PASS |

`pyproject.toml` requires Python `>=3.11,<3.12`; all installed versions satisfy the declared constraints: Pillow `>=10,<13`, torch `>=2,<3`, torchvision `>=0.16,<1`, NumPy `>=1.26,<3`, and pytest `>=8,<10`. The user reported installing dependencies manually. No installation or environment change was performed by this task.

## 3. Dataset evidence

### Current read-only filesystem checks

| Check | Observation | Interpretation |
|---|---|---|
| `data/raw/food-101/` | Exists | Existence only |
| `images/` class folders | 101 directories | Inventory observation |
| JPEG paths under `images/` | 101,000 `.jpg` paths | Count only; not a new decode/integrity check |
| Official metadata | `classes.txt`, `labels.txt`, `train.txt`, `train.json`, `test.txt`, `test.json` all present | Presence only |
| License | `license_agreement.txt` present | Presence only; terms not re-reviewed in this baseline |
| Partial archive | `data/raw/food-101.tar.gz.part`, 4,996,278,331 bytes | Present; left untouched |
| Complete archive | `data/raw/food-101.tar.gz` absent | Current filesystem check |

### Existing audit (not rerun)

`data/reports/food101/food101_report.json` says full audit, started `2026-11-16T05:15:33.661117+00:00`, duration 14.54 seconds. It records 101,000 images, 101 classes, train/test metadata counts 75,750/25,250, 101,000 decoded valid images, zero decode errors, zero missing/unlisted images, and eight duplicate-content groups. The report's `image_decoding_and_sha256` criterion says all 101,000 images were checked and valid and includes per-image duplicate grouping plus SHA-256 values for two train/test duplicate hashes. The two cross-official-split groups are:

- `images/churros/2547960.jpg` and `images/churros/2617186.jpg` — SHA-256 `ec9b9f1437dde19e478d277d3519678d11f64435fbcf6f62ed965d048c9de9fe`.
- `images/paella/2199939.jpg` and `images/paella/2199941.jpg` — SHA-256 `f40544646c0e07fcc3e6539bd03af71949dabb4b79515bfe207ca66e5f1c6f54`.

The report warns these exact duplicates cross official splits. No image was removed. The report says its archive was present during that audit and records 4,996,278,331 bytes and local SHA-256 `d97d15e438b7f4498f96086a4f7e2fa42a32f2712e87d3295441b2b6314053a4`; current check finds only the `.part` path, so archive state changed or the old report's archive evidence is stale. `official_sha256` is null and `official_checksum_verified` is false; the local digest is not an independently verified official checksum. The checkpoint has schema/version 1 and 101,000 file entries, but no independently verifiable content fingerprint beyond what the report/checkpoint implementation records; the report is historical evidence and was not revalidated in this task.

The old report's `overall` is `INCOMPLETE`. Its DataLoader section records Python 3.13.5 / torch 2.11.0+cpu, torchvision missing, and a fallback preprocessing path. That fallback does **not** prove project torchvision transforms work. Its historical loader claims do not substitute for the requested current Python 3.11 synthetic smoke tests.

### POC v1

`data/splits/v1/dataset_metadata.json` records 10,000 images, 10 classes, 6,000 train, 1,500 val, 2,500 test, dataset version `poc-v1`, and preparation timestamp `2026-10-31T06:34:15.687416+00:00`. Current CSV line counts (excluding headers) are 6,000 / 1,500 / 2,500. Its `root` is `D:\food-recognition-dietary-assistant\food-recognition-dietary-assistant\data\raw\food-101`, which is not portable to the current checkout. No v1 file was modified.

### Git tracking and sensitivity

Read-only `git ls-files data/raw data/reports` returns only `data/raw/.gitkeep`; no raw images/archive or report files are tracked. `data/reports/` is already untracked in the pre-existing worktree. Audit JSON contains local absolute paths and image inventory/checksum metadata; avoid committing it without reviewing repository data policy. This task adds only `artifacts/reports/elt/baseline.md`.

## 4. Unit test results

| Command | Result | Counts / diagnosis |
|---|---|---|
| `.\.venv-311\Scripts\python.exe -m pytest tests/unit/test_poc_preprocessing.py tests/unit/test_ml_pipeline.py -q` | **PASS** | Exit code 0; `7 passed in 159.20s (0:02:39)`, 0 failed, 0 skipped. The user had reported this result; this continuation independently reran the command. Suite includes CPU smoke-training paths on synthetic data, without Food-101 or pretrained weights. |
| Verifier tests before fix | **FAIL** | `pytest tests/unit/test_food101_verifier.py -q`: 1 failed, 3 passed in 6.81s. `test_full_audit_resumes_unchanged_images_and_rechecks_changed_file` expected `overall == PASS`, but `image_inventory` used `count == 1000` even though the test configured a one-class, two-image fixture. |
| Verifier tests after fix | **PASS** | `pytest tests/unit/test_food101_verifier.py -q`: 5 passed in 5.64s, exit code 0. Includes a new missing-image regression check. All tests use temporary fixtures; no downloader/network test exists in `tests/`. |

The required test suite passed. The verifier suite now passes, including the regression that removes one expected fixture image and asserts both the audit and `image_inventory` fail. No downloader/network test was run.

### Verifier defect and minimal fix

The verifier's declared Food-101 contract is `EXPECTED_CLASSES = 101`, `EXPECTED_TRAIN_PER_CLASS = 750`, `EXPECTED_TEST_PER_CLASS = 250`, and `EXPECTED_IMAGES = 101_000`. The project config `configs/data.yaml` selects Food-101 and its official `train` / `test` splits. The failing fixture correctly patches these expectations to one class, one train image, one test image, and two total images. However, `src/food_assistant/ml/food101_verifier.py` independently hard-coded `count == 1000` in the `image_inventory` criterion at the former line 227, causing the synthetic fixture to fail.

The minimal fix derives `expected_images_per_class` as `EXPECTED_TRAIN_PER_CLASS + EXPECTED_TEST_PER_CLASS` and compares each class inventory to that value. With production constants, this remains exactly 1,000 images per class and 101,000 total. The new test `test_full_audit_rejects_missing_image_from_configured_class_count` deletes one fixture image and verifies `image_inventory` and overall audit both remain `FAIL`; the count check was not removed or relaxed.

## 5. Real transforms and DataLoader

| Check | Status | Evidence |
|---|---|---|
| `TORCHVISION_IMPORT_PASS` | PASS | Real import succeeded; torchvision `0.29.2+cpu`. |
| `PROJECT_TRANSFORM_PASS` | PASS | Used `preprocessing("resnet18")` and `build_transform(..., train=False)` on a synthetic in-memory RGB image. Output is `torch.Tensor`, shape `(3, 224, 224)`, dtype `torch.float32`, all values finite. |
| Standalone real PyTorch DataLoader | PASS | Imported `torch.utils.data.DataLoader` and used it with the project Dataset on temporary synthetic files. |
| Project synthetic Dataset + DataLoader | PASS | `ManifestDataset` length 2; `__getitem__` returned `(3, 224, 224)` float32 tensor and Python `int` label. DataLoader batch `(2, 3, 224, 224)`, float32 images, int64 labels `[0, 1]`, all values finite. Temporary fixtures were cleaned by `TemporaryDirectory`. |

## 6. Code flow map

```text
Source archive / existing raw tree
       ↓
Download helpers and Food-101 verifier
       ↓
Split/manifest preparation and audit
       ↓
ManifestDataset / OfficialFood101Dataset
       ↓
Project torchvision transforms
       ↓
PyTorch DataLoader (constructed by callers)
       ↓
Training / evaluation scripts
```

| Component | File | Confirmed function/class | Reusable? | Current issue |
|---|---|---|---|---|
| Download | `scripts/download_food101.py` | `download`, `safe_extract`, `main` | Yes, existing utility | Resume checks status/range partially; untouched in Phase 0. |
| Parallel download | `scripts/download_food101_parallel.py` | `main` (nested `fetch`) | Existing utility | Uses network and writes raw/chunks; not run. |
| Verify | `src/food_assistant/ml/food101_verifier.py` | `_fingerprint`, `_load_checkpoint`, `AuditLock`, `_inspect_image`, `audit_food101`, `_write_reports` | Yes, existing audit implementation | Existing report is historical; no new audit was run to avoid expensive full image rehash and report/raw writes. |
| Verify CLI | `scripts/verify_food101.py` | `FallbackValidationTransform`, `verify_loader`, `main` | Partially | Historical verifier permits fallback when torchvision is missing; that path is explicitly not evidence for real torchvision preprocessing. |
| POC verify | `src/food_assistant/ml/cli.py` | `verify_dataset` | Yes | Writes `quality_report.json` into manifest dir, so not invoked. |
| Manifest | `src/food_assistant/ml/datasets.py` | `audit_manifest`, `prepare_food101`, `read_json`, `write_json` | Yes | `prepare_food101` refuses a frozen CSV output, but baseline did not call it. |
| Dataset | `src/food_assistant/ml/datasets.py` | `ManifestDataset`, `OfficialFood101Dataset` | Yes | Requires Pillow/NumPy; root override exists for relocated manifest data. |
| Transform | `src/food_assistant/ml/transforms.py` | `preprocessing`, `build_transform` | Yes | Real transform passed on synthetic RGB input. |
| DataLoader | `scripts/verify_food101.py`, training scripts | `verify_loader` constructs `torch.utils.data.DataLoader` | Yes as an existing smoke path | Synthetic project Dataset/DataLoader smoke test passed. |
| Lock | `src/food_assistant/utils/process_lock.py` | `running_training`, `training_lock` | Potentially reusable for training locks | Current lock is model/output scoped; not tested or changed in this phase. |

## 7. Risks and gaps

| ID | Issue | Evidence | Severity | Suggested phase |
|---|---|---|---|---|
| R0-1 | Resolved: verifier inventory count was hard-coded | `src/food_assistant/ml/food101_verifier.py`: criterion now uses `EXPECTED_TRAIN_PER_CLASS + EXPECTED_TEST_PER_CLASS`; small-fixture pass and missing-image fail both covered | Resolved | Completed in Phase 0 |
| R0-2 | Existing audit found exact duplicate content crossing official train/test | Two paths/hash pairs listed above; audit status `WARN` | High for unbiased evaluation | Phase 1/4 design; do not alter POC v1 silently |
| R0-3 | Current archive is a `.part`, with no complete `.tar.gz` at the checked path | Current filesystem observation; old audit recorded archive present | Medium; reconcile in a later authorized data-safety/extract phase | Phase 2/3 |
| R0-4 | POC metadata root is an old absolute Windows path | `dataset_metadata.json` root field | Medium; compatibility/portability issue; v1 is frozen | Phase 1, with versioned approach if contract changes |
| R0-5 | Existing full audit is historical and overall incomplete | Timestamp 2026-11-16; loader preprocessing blocked; no official archive checksum | Medium; not current end-to-end PASS | Phase 4 audit policy; current transform/DataLoader smoke checks are independently passing |
| R0-6 | Downloader behavior remains untested; there are no matching downloader tests in `tests/` | Test discovery found only `tests/unit/test_food101_verifier.py` for verifier/data-specific extras; network tests not run | Medium | Phase 3/6 using mocks and synthetic fixtures |

## 8. Phase 0 acceptance checklist

| Criterion | Status | Evidence |
|---|---|---|
| Python 3.11 interpreter works | PASS | `.venv-311` reports Python 3.11.9 |
| Required dependencies import | PASS | Python 3.11.9; pytest/torch/torchvision/Pillow/NumPy import and version checks pass |
| Required unit tests pass | PASS | Required suite: 7 passed in 159.20s |
| Real torchvision project transform | PASS | Real project transform on synthetic RGB image yields finite float32 tensor `(3, 224, 224)` |
| Real DataLoader smoke test | PASS | Synthetic project Dataset returns valid sample; real PyTorch DataLoader yields valid finite batch and int64 labels |
| Raw dataset unchanged | PASS | Read-only checks only; `.part` left untouched |
| POC v1 unchanged | PASS | Read-only metadata/CSV counts; no manifest command invoked |
| Downloader unchanged | PASS | No source edits or downloader invocation |
| No serious setup blocker | PASS | Verifier suite is 5/5; inventory still fails when an expected image is missing |
| Baseline report written | PASS | This file |

**Phase 0 result: PASS.** The initial verifier defect is fixed without weakening the Food-101 inventory requirement: production expectations remain 101 classes × (750 train + 250 test) = 101,000 images, and a missing-image regression is explicitly rejected. Required and verifier test suites, real project transforms, and synthetic Dataset/DataLoader smoke checks pass. The historical full audit remains `INCOMPLETE` due to its recorded cross-split duplicates and prior transform check; this is documented as a data-quality risk and was not altered or silently reclassified.

## 9. Recommendations for later phases

- Dependencies are now installed by the user and real imports passed; no reinstall is needed.
- Keep the configured class/train/test totals aligned if Food-101 contracts are ever versioned; the inventory criterion now derives its per-class expectation from those train/test totals.
- Keep transform and DataLoader regression checks synthetic; keep raw and v1 untouched.
- Treat the existing archive report as historical. Any archive reconciliation or fresh full audit belongs to a later phase and must preserve the existing `.part` and raw tree.
- Keep duplicate-content handling explicit in future quality gates; preserve official test metadata and frozen POC v1.
- Address portability with a new/versioned metadata contract rather than silently editing frozen v1.

## 10. Change log

- Updated: `src/food_assistant/ml/food101_verifier.py` to derive expected images per class from the configured official train/test counts.
- Updated: `tests/unit/test_food101_verifier.py` with a regression test confirming a missing expected image fails the inventory criterion.
- Updated: `artifacts/reports/elt/baseline.md` with root cause, fix, before/after evidence, and final acceptance.
- Raw Food-101, the partial archive, POC v1 manifests, downloader, and other ML components were not changed.
- Pre-existing worktree changes listed in Section 2 were preserved.




