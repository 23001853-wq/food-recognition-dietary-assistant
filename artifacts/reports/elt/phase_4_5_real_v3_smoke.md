# PHASE: 4.5 — Real Food-101 v3 Smoke Test

## BASELINE VERIFIED

- Phase 4 report: `artifacts/reports/elt/phase_4_full_v3_validation_final.md` reports v3 PASS under exact-SHA policy: 101 classes, 100,990 manifest rows, train/val/test 68,166/7,574/25,250, exact cross-partition duplicates 0. It says full-v3 near-duplicate audit was not checked.
- Existing Phase 4.5 synthetic report: `artifacts/reports/elt/phase_4_5_ml_image_processing.md`; its Gate C/D were incomplete for actual release inference and real-v3 DataLoader.
- v3 schema is `3.0`; manifest columns match `ManifestDataset.FIELDS`: `image_id, relative_path, source_dataset, class_id, class_key, split, checksum, validation_status, dataset_version, capture_group`.
- v3 fingerprint: `aa43cfa766982006fe6a3611a49e0f07d35d33d56e54310e923cb5e58fbdbb59`; source fingerprint: `f5cfdb7480c2650ed94c8624652ac595e4a7f038d660f4886869836632c56ed1`; inventory SHA-256: `b91c327114b45c4fe2e626404a86c449891e6e258c832e67a7f2350aeb59adb9`.
- `ManifestDataset.__init__` calls `audit_manifest()` on every row supplied; this reads/checksums all manifest image paths. A 30-row sample manifest was therefore written outside immutable data directories, with v3 schema, full original labels, original class IDs and portable relative image paths. It is marked `sample_only: true`, points to the v3 fingerprint and is not represented as a full validated dataset.
- Configured raw image root was `data/raw/food-101/images`. No pretrained weights or trained release were used.

## MANIFEST SOURCE

- Source manifest: `data/splits/v3/dataset_manifest.csv`.
- Source metadata/labels/quality evidence: `data/splits/v3/dataset_metadata.json`, `labels.json`, `quality_report.json`.
- Root resolution came from `configs/food101_elt_full_v3.yaml` and was passed to `ManifestDataset` as `root_override`. `audit_manifest` validated sample-relative paths, file existence, sample checksums, split labels and class consistency.
- The sample manifest preserves source class IDs and names; it contains no official test rows.

## SAMPLING

- Policy: read the v3 manifest index; for official class IDs 0–19, choose one train ID per class; for IDs 0–9, choose one validation ID per class. Candidates were sorted by image ID and selected with Python `random.Random(4505)`. This keeps class IDs contiguous for the current consumer's `audit_manifest` rule while covering 20 classes.
- Seed: **4505**. No statistical representativeness is claimed.
- Distinct real images: **30** — train **20**, validation **10**, test **0**.
- Class coverage: **20** classes. The sample manifest and portable sample records are in `artifacts/reports/elt/phase_4_5_real_v3_smoke_final2/`.
- Selected source IDs, source partitions, class IDs/names, relative paths and seed are recorded in `sampling_metadata.json`.
- Final bounded run decoded each of the 30 selected source IDs once through the ResNet18 train/validation DataLoaders. EfficientNet transforms and image-processing checks reused the captured in-memory RGB arrays. During implementation, earlier runs reprocessed the same deterministic 30 IDs; no additional IDs were selected or touched. The successful final run did not decode any v3 image outside this sample.

## FILES REVIEWED

- `artifacts/reports/elt/phase_4_full_v3_validation_final.md`
- `artifacts/reports/elt/phase_4_5_ml_image_processing.md`
- `configs/food101_elt_full_v3.yaml`
- v3 `dataset_manifest.csv`, `dataset_metadata.json`, `labels.json`, `quality_report.json`
- `src/food_assistant/ml/datasets.py`, `transforms.py`, `image_ops.py`, `near_duplicate.py`, `train_utils.py`
- `tests/integration/test_ml_image_pipeline_synthetic.py`, unit preprocessing/transform/safety tests

## FILES CHANGED

- Added `scripts/phase_4_5_real_v3_smoke.py` for deterministic bounded sampling, sample-manifest construction, real DataLoader smoke, in-memory transforms and evidence output.
- Added `artifacts/reports/elt/phase_4_5_real_v3_smoke.md`.
- New outputs: `artifacts/reports/elt/phase_4_5_real_v3_smoke/`, `.../phase_4_5_real_v3_smoke_final/`, and `.../phase_4_5_real_v3_smoke_final2/`. These contain sample-only manifest/evidence/figures; the latest successful bounded run is `final2`.
- No production data, loader, preprocessing, split, model or config files were modified.

## REAL IMAGE PROCESSING RESULTS

- **Color and representation:** all 30 in-memory captured images passed RGB shape/range, grayscale/HSV channel shape, HWC↔CHW round-trip, and uint8→float32 `[0,1]` checks. On the representative `apple_pie/1963597` image (train), RGB HWC was `341×512×3`; BGR round-trip was exact; HSV→RGB max absolute error was 3; float output was `float32` in `[0,1]`; CHW was `3×341×512` and round-trip exact.
- **EXIF and RGB conversion:** `ManifestDataset.__getitem__` runs `ImageOps.exif_transpose(...).convert("RGB")` before the transform for each selected read. The smoke exercised that consumer path. EXIF tag frequency across v3 was not measured.
- **Pixel operations:** brightness ×1.1, contrast ×1.1 and gamma 0.9 returned uint8 values clipped to `[0,255]`. Values are demonstration parameters and did not alter production config.
- **Geometry:** representative resize returned `256×256×3`; center crop `289×435×3`; flip and opt-in 5° rotation retained `341×512×3`. Actual train `RandomResizedCrop` ran in the train DataLoader; validation used deterministic resize/center crop only.
- **Filtering:** Gaussian noise was added to an in-memory copy with seed 4505, and the original decoded image was used only as the experiment reference. MSE vs that reference: noisy 141.97, Gaussian blur 37.56, median 49.90. This single-image result is illustrative, not an image-quality or model-accuracy conclusion.
- **Morphology/edges/regions:** threshold mask used the 65th grayscale percentile. Erosion/dilation/opening/closing, Canny, Sobel and connected regions executed. The threshold mask is not a ground-truth food segmentation annotation.
- Figures: `real_v3_processing.png` and `real_v3_edges_regions.png` under `phase_4_5_real_v3_smoke_final2/`.

## NEAR-DUPLICATE DEMONSTRATION

- One sampled real train image was compared only against in-memory derivatives of itself; no pairwise comparison between distinct v3 images was run.
- pHash: 8×8 DCT hash, 64 bits. Hamming threshold ≤12; SSIM threshold ≥0.72. SSIM and pHash were computed for resize, JPEG recompression (quality 82), brightness ×1.08, 5% border crop and horizontal flip.
- Results: resize `(H=0, SSIM=0.9596)`, JPEG `(H=0, 0.9949)`, brightness `(H=0, 0.9943)` were flagged; crop `(H=20, 0.3328)` and flip `(H=32, 0.3363)` were not flagged. SHA-256 values for the source manifest image and in-memory JPEG derivatives are retained in `smoke_metrics.json`.
- These controlled pairs have known positive lineage to the same source. Crop and flip are false negatives at this threshold. There were no unrelated real negative pairs, so real-sample false-positive rate is **not evaluated**. The earlier synthetic Phase 4.5 report includes negative fixtures and its own false-positive/false-negative counts.
- This does not audit full v3 or establish leakage status; it does not change any split, image or exclusion ledger.
- Figure: `real_v3_near_duplicate.png` in the final2 artifact folder.

## DATALOADER INTEGRATION

- Real `ManifestDataset` and `torch.utils.data.DataLoader` ran on the 20 train + 10 validation sample manifest rows with `num_workers=0`, no shuffle, batch size 4. ResNet18 was the direct DataLoader transform profile; labels matched the source sample manifest and were `int64`.
- ResNet18 train: `[20,3,224,224]`, `torch.float32`, finite; random train augmentation enabled. ResNet18 validation: batches `[4,3,224,224]`, `[4,3,224,224]`, `[2,3,224,224]`, finite; no random augmentation. Repeated eval transform max absolute difference: **0.0**.
- EfficientNet-B0 preprocessing ran on the exact same once-decoded real sample images in memory to stay within the image cap; its train/validation tensors were `[20,3,224,224]` and `[10,3,224,224]`, float32 and finite. Validation repeat max absolute difference: **0.0**. It was not separately passed through a second real-image DataLoader.
- Original official class IDs/names were preserved. The sample spans IDs 0–19 for train and 0–9 for validation; the labels file retains the full v3 101-class mapping. There are no fake checkpoints and no model inference.
- Batch metrics, decoded sizes, label results and determinism evidence are in `smoke_metrics.json`.

## TESTS EXECUTED

- `.venv-311\Scripts\python.exe -m scripts.phase_4_5_real_v3_smoke --output artifacts/reports/elt/phase_4_5_real_v3_smoke_final2` — **PASS**, 30 unique real sample IDs, train 20, validation 10, 20 classes; all selected paths/checksums and loaders validated. Earlier development invocations used the same deterministic IDs; no extra IDs were accessed.
- `.venv-311\Scripts\python.exe -m pytest tests/unit/test_ml_image_ops.py tests/unit/test_ml_transforms_contract.py tests/unit/test_near_duplicate_detection.py tests/unit/test_poc_preprocessing.py tests/integration/test_ml_image_pipeline_synthetic.py -q` — **14 passed**.
- `.venv-311\Scripts\python.exe -m pytest tests/unit/test_food101_transform.py tests/unit/test_food101_split_v3.py tests/unit/test_food101_data_safety.py tests/unit/test_food101_elt_config.py -q` — **50 passed**.
- `.venv-311\Scripts\ruff.exe check scripts/phase_4_5_real_v3_smoke.py scripts/demo_food_image_processing.py src/food_assistant/ml/image_ops.py src/food_assistant/ml/near_duplicate.py tests/unit/test_ml_image_ops.py tests/unit/test_ml_transforms_contract.py tests/unit/test_near_duplicate_detection.py tests/integration/test_ml_image_pipeline_synthetic.py` — **All checks passed** after formatting the smoke script.
- Matplotlib emitted a user-profile font-cache permission warning; PNG and JSON artifacts were generated successfully. No tests were skipped in the listed pytest commands.

## DATA SAFETY

- No writes to `data/raw/**`, `data/splits/v1/**`, `v2/**`, `v3/**`, `poc-v2/**`, official metadata, published manifests, cleanup/exclusion ledgers, configs or model artifacts.
- The sample-only manifest lives under the report output and points to existing raw image paths. Transformed images were held in memory and saved only as report PNGs. No image was copied into raw or split directories.
- Sampling excluded official test rows. No download, install, training, model inference, full-dataset rehash, full-v3 image scan, resplit or relabel was performed.
- The worktree was already dirty before this task; those changes were preserved. Current changes are limited to the new smoke script and report/evidence outputs.

## ACCEPTANCE

- Gate A — **PASS**: real-sample color/representation, pixel, geometry, filtering, morphology and edge operations executed; all 30 captured images passed basic color/layout/range checks.
- Gate B — **PASS (bounded derivative demonstration)**: pHash/Hamming/SSIM ran on one real image and five known in-memory derivatives; misses and limitations are recorded. No real cross-image false-positive evaluation and no full-v3 audit are claimed.
- Gate C — **PASS for sample preprocessing/DataLoader**: train and validation real DataLoaders passed for ResNet18; EfficientNet-B0 transforms passed on the same in-memory real sample; labels/shapes/dtypes/finite values and eval determinism passed. This does not resolve the earlier `Release.predict()` limitation; no trained release exists.
- Gate D — **PASS for selected sample**: schema, paths, original labels and source split IDs validated; train/validation DataLoader smoke passed. No test images or resplitting.
- Gate E — **PASS**: no protected data or artifacts were written; only the bounded selected sample was read and the final run decoded those 30 IDs once.
- Gate F — **PASS**: portable sampling evidence, before/after PNGs, metrics, loader results, tests and scope limitations are recorded.
- Overall — **REAL V3 SAMPLE SMOKE PASS**. This is not a `FULL V3 PROCESSING PASS`.

## REMAINING LIMITATIONS

- Full-v3 image processing not executed.
- Full-v3 near-duplicate audit not executed.
- Trained model inference not verified; no trained model release was present.
- Real near-duplicate false-positive rate is not evaluated because only positive lineage pairs were constructed.
- Sample selection intentionally covers 20 contiguous official class IDs to preserve the current consumer's contiguous-ID audit rule; it is not statistically representative of all 101 classes.

## NEXT PHASE: NONE — STOP
