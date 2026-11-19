# PHASE: 4.5 — ML Image Processing & Near-Duplicate Analysis

## BASELINE

- Repository: `food-recognition-dietary-assistant`; working tree was already dirty with Phase 2–4 changes and untracked artifacts. Those changes were preserved.
- Python 3.11.9; Pillow 12.3.0; torch 2.15.0+cpu; torchvision 0.29.2+cpu; NumPy 2.4.6; OpenCV 4.14.0; scikit-image 0.26.0.
- Input: existing Food-101 split v3; no full-image scan or rehash was run in this phase.
- Phase 4 evidence: `artifacts/reports/elt/phase_4_full_v3_validation_final.md` reports 101 classes, 100,990 rows, train/val/test 68,166/7,574/25,250, exact hash leakage 0 and overall PASS under exact-SHA policy. That report explicitly says full-v3 near-duplicate audit was NOT_CHECKED and capture-group IDs are unavailable.
- v3 files were inspected read-only: manifest CSV header and a sample row; metadata and quality report. Metadata says schema 3.0, source fingerprint `f5cfdb7480c2650ed94c8624652ac595e4a7f038d660f4886869836632c56ed1`, inventory SHA-256 `b91c327114b45c4fe2e626404a86c449891e6e258c832e67a7f2350aeb59adb9`, v3 fingerprint `aa43cfa766982006fe6a3611a49e0f07d35d33d56e54310e923cb5e58fbdbb59`.

## FILES REVIEWED

- Prompt and project guidance: `promts/food101_elt_prompt_pack/prompts/PHASE_4_5_ML_IMAGE_PROCESSING_PROMPT.md`, `00_GUARDRAILS.md`, `00_MASTER.md`, and repository `AGENTS.md` instructions supplied for this work.
- Reports/config/manifest: Phase 4 full-v3 report, `configs/food101_elt_full_v3.yaml`, `configs/classes_full.yaml`, `configs/data.yaml`, `configs/resnet18.yaml`, `configs/efficientnet_b0.yaml`, `configs/hog_svm.yaml`, `data/splits/v3/dataset_manifest.csv` (header/sample), `dataset_metadata.json`, `quality_report.json`.
- Existing ML code: `src/food_assistant/ml/transforms.py`, `datasets.py`, `hog_features.py`, `train_utils.py`, `src/food_assistant/services/model_registry.py`.
- Existing tests: `tests/unit/test_poc_preprocessing.py` and Phase 4 transform/split/safety/config tests.

## FILES CHANGED

- Added `src/food_assistant/ml/image_ops.py` — explicit, opt-in processing operations.
- Added `src/food_assistant/ml/near_duplicate.py` — bounded small-batch perceptual comparison.
- Added `scripts/demo_food_image_processing.py` — reproducible synthetic demonstration runner.
- Added `tests/unit/test_ml_image_ops.py`, `tests/unit/test_ml_transforms_contract.py`, `tests/unit/test_near_duplicate_detection.py`.
- Added `tests/integration/test_ml_image_pipeline_synthetic.py`.
- Added this report and synthetic outputs under `artifacts/reports/elt/phase_4_5_demos_final2/`.
- Existing transforms, datasets, manifests, configs, model artifacts and data were not edited by these changes.

## FEATURE MATRIX

| Requirement | Code location | Status before | Tests/evidence | Result |
|---|---|---|---|---|
| PIL/array to RGB, grayscale, RGBA, EXIF | `image_ops.to_rgb_array`, `ManifestDataset.__getitem__` | PARTIAL | unit tests incl. EXIF orientation | PASS |
| RGB/BGR, HSV/RGB, HWC/CHW, uint8/float32 | `image_ops.py` | PARTIAL | channel, round-trip, layout and range tests | PASS |
| Brightness/contrast/gamma/normalization/clipping | `image_ops.py` | MISSING | range and invalid-input tests | PASS |
| Resize/crop/flip/opt-in rotation | `image_ops.py`; existing torchvision transforms | PARTIAL | geometry tests and official transform regression | PASS |
| Gaussian/median filtering | `image_ops.py` | MISSING | noisy synthetic fixture, MSE and validation tests | PASS |
| Morphology/edges/regions | `image_ops.py` | MISSING | mask expansion/shrink, connected components, edge tests | PASS |
| pHash/Hamming/SSIM | `near_duplicate.py` | MISSING | synthetic fixture metrics and bounds tests | PASS (fixture scope) |
| CNN train/eval preprocessing | `transforms.py` | ALREADY IMPLEMENTED / MUST PRESERVE | official ResNet18/EfficientNet weight transform regression | PASS |
| HOG RGB→gray path | `hog_features.py` | ALREADY IMPLEMENTED / MUST PRESERVE | configured feature extraction smoke test | PASS |
| ManifestDataset/DataLoader | `datasets.py` | ALREADY IMPLEMENTED / MUST PRESERVE | actual synthetic manifest + DataLoader | PASS (synthetic scope) |

## IMAGE PROCESSING RESULTS

- **Representation:** array operations use explicit RGB ordering for 3-channel arrays; OpenCV conversions require explicit BGR helpers. PIL inputs use EXIF transpose, convert to RGB, and support grayscale/RGBA/palette inputs through Pillow conversion. HWC↔CHW supports arbitrary finite numeric tensors without mistaking the channel axis for image height.
- **Color:** grayscale and OpenCV HSV are explicit. HSV uses OpenCV uint8 convention (H 0–179, S/V 0–255); HSV/RGB synthetic round-trip maximum absolute error is tested within 4 intensity values.
- **Pixel operations:** deterministic brightness, contrast, gamma and channel normalization are separate helpers. uint8 transforms clip and retain uint8; float transforms require [0,1]. Normalization refuses out-of-range/already-normalized input to avoid accidental double-normalization.
- **Geometry:** resize accepts explicit `(height,width)` and named interpolation; crop rejects out-of-bounds; flip is deterministic. Rotation is experimental and requires `enabled=True`. Cropping can remove food context or the subject; no random geometry was added to validation/test/inference.
- **Filtering:** Gaussian and median are opt-in only. On the seed-2026 synthetic scene, Gaussian noise MSE fell from 322.44 to 87.84 after Gaussian blur; salt-and-pepper MSE fell from 463.67 to 37.17 after median filtering. These are fixture-specific measurements, not a universal quality claim.
- **Morphology/regions:** foreground is nonzero/true, square all-ones structuring element; erosion shrinks, dilation expands, opening removes small foreground details, closing fills small gaps. Connected-component bounds and Sobel/Canny are available for synthetic analysis.
- Nothing in `image_ops.py` or near-duplicate helpers is enabled implicitly in production classification.

## NEAR-DUPLICATE RESULTS

- pHash uses grayscale uint8, resize, 2D DCT low-frequency coefficients, median threshold, fixed-width hex. Hamming distance validates equal-width hex input. SSIM uses grayscale images, resizes the second to the first and requires each dimension ≥7; it is not invariant to crop/rotation.
- Demo policy: hash size 8 (64 bits), maximum Hamming distance 12, minimum SSIM 0.72. This threshold is a synthetic demonstration setting, not an approved Food-101 policy.
- Eight synthetic pairs: TP=4, FP=1, TN=1, FN=2; precision=0.80, recall=0.667. Recompression/resize/brightness were found; crop and horizontal flip were missed. The false positive was a same-layout/different-channel-palette synthetic image. This demonstrates that pHash+SSIM evidence needs review and is not a label or split decision by itself.
- No real-v3 image pair was scanned. No claim is made that v3 is near-duplicate-free. Existing Phase 4 exact SHA-256 PASS remains a separate result.
- Pair comparison accepts only explicit caller-provided images and enforces a configurable maximum pair count. No traversal or implicit full-dataset API exists.

## ML PREPROCESSING

- Train transform remains the current `RandomResizedCrop(crop=224)` plus `RandomHorizontalFlip`, then `ToTensor` and ImageNet mean/std normalization. It remains train-only.
- Eval/inference remains `Resize(256)` plus `CenterCrop(224)`, `ToTensor`, and the existing ImageNet mean/std. EfficientNet-B0 keeps bicubic interpolation; ResNet18 keeps bilinear interpolation.
- Existing ResNet18 and EfficientNet-B0 transforms match their torchvision weight enum transforms exactly in the regression test. `Release.predict()` uses `build_transform(self.preproc)` for CNN inference; HOG/SVM continues through the pre-existing `extract_hog` path. No trained model release exists in `artifacts/models` (only `.gitkeep` files), so actual checkpoint inference was not run.
- Random rotations, denoising, color transforms, and near-duplicate detection are not part of the production model profile. No existing preprocessing metadata or model weights were overwritten.

## BEFORE/AFTER EVIDENCE

Artifacts are synthetic-only, seed 2026; the transform demonstration uses seed 2028. No Food-101 file was used by the demo.

- `artifacts/reports/elt/phase_4_5_demos_final2/food_image_color_spaces.png` — RGB, grayscale and OpenCV HSV.
- `.../food_image_pixel_ops.png` — brightness, contrast and gamma examples.
- `.../food_image_geometry.png` — resize and center crop; primary panel also shows flip.
- `.../food_image_processing.png` — Gaussian and salt-and-pepper noise, filters, MSE, and Canny edges.
- `.../food_image_morphology.png` — threshold mask, opening and closing; unit tests also verify erosion and dilation.
- `.../near_duplicate_fixtures.png` and `.../metrics.json` — exact SHA-256 versus pHash evidence, candidate transformations and similarity scores.
- `.../train_augmentation.png` — one existing train-transform sample; demo-only image visualization.
- All new helpers are research/demo-only. Production behavior still uses established transforms and HOG extraction.

## TESTS EXECUTED

- `.venv-311\Scripts\python.exe -m pytest tests/unit/test_ml_image_ops.py tests/unit/test_ml_transforms_contract.py tests/unit/test_near_duplicate_detection.py tests/unit/test_poc_preprocessing.py tests/integration/test_ml_image_pipeline_synthetic.py -q` — **14 passed**.
- `.venv-311\Scripts\python.exe -m pytest tests/unit/test_food101_transform.py tests/unit/test_food101_split_v3.py tests/unit/test_food101_data_safety.py tests/unit/test_food101_elt_config.py -q` — **50 passed**.
- `.venv-311\Scripts\ruff.exe check scripts/demo_food_image_processing.py src/food_assistant/ml/image_ops.py src/food_assistant/ml/near_duplicate.py tests/unit/test_ml_image_ops.py tests/unit/test_ml_transforms_contract.py tests/unit/test_near_duplicate_detection.py tests/integration/test_ml_image_pipeline_synthetic.py` — **All checks passed**.
- `.venv-311\Scripts\python.exe -m scripts.demo_food_image_processing --output artifacts/reports/elt/phase_4_5_demos_final2` — completed and emitted PNGs/JSON metrics. Matplotlib printed a user-profile font-cache permission warning; artifact generation still completed. No package installation, network, download, training or full dataset scan was performed.
- During development, one channel-layout test exposed that the normal image validator rejected a valid CHW tensor; layout validation was separated from image validation and the focused suite then passed.

## DATA SAFETY

- Raw Food-101, official split metadata, `data/splits/v1`, `v2`, `v3`, `poc-v2`, published manifests and model artifacts were not written by this phase.
- No dataset image was opened for demo/test writes; all writes targeted new source/test files and `artifacts/reports/elt/phase_4_5_demos_final2/` plus this report.
- No v1/v2/v3 manifest was regenerated or changed. No training/default model behavior or Phase 4 quality policy was changed.
- Full raw/v3 checksums were intentionally not recomputed; baseline evidence is referenced above rather than representing an independent full-data integrity scan.

## ACCEPTANCE

- Gate A — **PASS**: synthetic color, pixel, geometry, filtering, morphology, edge and region tests plus visual artifacts.
- Gate B — **PASS (synthetic algorithm scope)**: pHash/Hamming/SSIM, reproducibility, fixture metrics and error examples verified; real-v3 near-duplicate scan is NOT RUN and no cleanliness claim is made.
- Gate C — **INCOMPLETE**: ResNet18/EfficientNet preprocessing and HOG feature contracts pass; train-only randomness and deterministic eval pass. Actual `Release.predict()` with trained artifacts is untested because no trained weights are present.
- Gate D — **INCOMPLETE**: v3 schema/metadata and consumer contract were inspected; real `ManifestDataset` construction would audit/hash all manifest rows, so only a real synthetic manifest/DataLoader integration was run. Full-v3 consumer loading was not run.
- Gate E — **PASS**: operations were confined to new code/tests/report/demo output; no network/download/training/full scan.
- Gate F — **PASS**: before/after figures, quantitative MSE, pHash/SSIM metrics, test evidence, feature mapping and production-vs-experimental separation are documented.
- **Overall — INCOMPLETE** because trained-release inference and full-v3 `ManifestDataset` loading were not executed. Algorithm and synthetic integration tests pass; this is not a full-v3 image-quality audit.

## RISKS / OPEN ITEMS

- pHash/SSIM has synthetic false positives and false negatives under the selected demo threshold; real data thresholds require a separately approved, labeled sample review.
- Capture-group leakage is not evaluable from v3 because no capture-group IDs were recorded. Near-duplicate full-v3 scan remains intentionally unrun.
- The report/demo directory may emit a harmless Matplotlib warning when its user-level cache directory is not writable; PNG and JSON files were produced successfully.
- Inference compatibility is established by shared transform code review and transform regression tests, not by a trained checkpoint run.

## NEXT PHASE

None — STOP.
