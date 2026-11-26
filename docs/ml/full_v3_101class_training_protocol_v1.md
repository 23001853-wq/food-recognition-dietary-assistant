# Food-101 Cleaned v3 — Training and Evaluation Protocol v1.0

**Status:** protocol design frozen for this dataset fingerprint; executable final-test selection tooling remains a Phase 9 precondition.  
**Frozen:** 2026-11-25  
**Protocol scope:** HOG-SVM, ResNet18 and EfficientNet-B0, 101-class cleaned Food-101 v3.

## 1. Immutable dataset binding

| Binding | Value |
|---|---|
| Dataset/version | Food-101 cleaned split v3 / `v3` |
| Dataset root | `data/splits/v3` (metadata-controlled raw root) |
| Source fingerprint | `f5cfdb7480c2650ed94c8624652ac595e4a7f038d660f4886869836632c56ed1` |
| Inventory SHA-256 | `b91c327114b45c4fe2e626404a86c449891e6e258c832e67a7f2350aeb59adb9` |
| v3 fingerprint | `aa43cfa766982006fe6a3611a49e0f07d35d33d56e54310e923cb5e58fbdbb59` |
| `dataset_manifest.csv` SHA-256 | `f135b5492bc3767a5f18268838de67a5535004f8d361c3a659ff9946d4f9f3c4` |
| `labels.json` SHA-256 | `b80ce882769587fdaf2b1ed0534ccfffb1e6c70a8c7b6c3deac45ef167a0ae43` |
| Manifest schema | `3.0` |
| Class map | 101 labels, IDs `0..100` in existing `labels.json` order |
| Counts | train 68,166; validation 7,574; test 25,250; total 100,990 |

The hashes above bind this protocol to the exact published metadata files observed at freeze time. If any digest changes, this protocol does not apply; create a new protocol revision after review. The published v3 report records 100,990 unique IDs, zero missing references, zero exact duplicate content groups within or across partitions, and unchanged official test IDs/order/checksums. This phase independently streamed the CSV manifest and verified its row/split counts, label mapping, safe relative-path syntax, unique IDs and zero cross-split checksum overlaps. It did not hash or decode the underlying 100,990 image files; the previous validation report is the evidence for those checks.

## 2. Split roles and leakage controls

- **Train:** fit model parameters and train-only scalers. CNN train augmentation is enabled only through the existing training transform.
- **Validation:** choose hyperparameters, early stopping and the winning candidate. Use deterministic evaluation transforms. No parameters or statistics are fit from validation.
- **Test:** remain sealed until one model candidate and its inference/evaluation protocol have been frozen. Evaluate that single selected candidate once for final reporting. Do not use test results to rank, tune, stop or replace candidates.
- Any ID or SHA-256 collision across partitions is a hard stop. No split repair is permitted under this protocol; the v3 artifact is immutable.
- Do not apply notebook-only blur, Canny, morphology or other demonstration transforms to model input.

## 3. Model and preprocessing contracts

| Candidate | Contract |
|---|---|
| HOG-SVM | Decode to RGB, convert RGB to grayscale, resize to 128×128, HOG with 9 orientations, 8×8 pixels/cell, 2×2 cells/block, L2-Hys; vector has 8,100 float features. Existing pipeline fits `StandardScaler` followed by linear `LinearSVC`; validation compares C = 0.1, 1, 10 in that order. No test-derived feature scaling. |
| ResNet18 | 101-output classifier; train uses random resized crop and horizontal flip; validation/test/inference resize 256, center crop 224, tensor conversion and ImageNet mean/std; bilinear interpolation. |
| EfficientNet-B0 | 101-output classifier; same normalization and train/eval augmentation boundary; validation/test/inference resize 256, center crop 224; bicubic interpolation. |

CNN model factory receives `pretrained=True` for the proposed full experiment. Training must not start unless the corresponding torchvision weights are already locally available or a separate explicit download approval is granted. The Phase 8 dry-run always constructs the architecture with `weights=None` and performs no network access. The config field `pretrained` is now consumed by the trainer instead of being an ignored declaration.

## 4. Experiment settings (proposals, not optimized settings)

- Common: seed 42; dataset version `v3`; 101 outputs; artifact under `artifacts/models/<model>/<unique-run-id>/`; validation is the only model-selection split.
- HOG-SVM: existing 128×128 HOG descriptor and three-C linear SVM grid. Current implementation materializes complete train and validation feature matrices.
- ResNet18: existing two-stage schedule (5 head epochs + 10 fine-tuning epochs), batch 16, AdamW, head LR 0.001, fine-tune LR 0.0001, weight decay 0.0001, AMP when CUDA is available, pretrained ImageNet initialization.
- EfficientNet-B0: same schedule and optimizer settings, batch 8 to reduce activation memory, AMP when CUDA is available, pretrained ImageNet initialization.
- Values are starting points only. Any tuning is performed using validation and recorded as a distinct run/config. Existing training code does not implement a learning-rate scheduler or class weighting; neither is claimed by these configs.

## 5. Metrics and selection

Report top-1 accuracy, macro precision, macro recall, macro-F1, per-class support, confusion matrix, batch-1 CPU inference p50/p95 latency, and artifact size. Report top-5 accuracy for CNN score vectors. HOG normalized decision scores are ranking scores, not calibrated probabilities.

Primary ranking metric is validation macro-F1 at full stored precision. Deterministic tie-breaks are validation top-1 accuracy, lower validation p95 latency under the common CPU protocol, smaller sealed artifact size, then model name and run ID lexicographically. There is no approved minimum performance threshold; this protocol defines comparisons, not a production quality guarantee.

**Implementation limitation:** current `select_production_model.py` requires test metrics for every candidate and writes `current.json` during selection. That workflow conflicts with this protocol's selected-candidate-only final test. Do not invoke it for full v3. Before Phase 9 reaches test evaluation or publication, implement and test separate validation-only candidate freeze, selected-candidate final evaluation, and explicit release publication. The current POC freeze tool is also hard-coded to 10-class `poc-v1` and cannot be used for v3.

## 6. Reproducibility and checkpoints

Record per run: protocol revision and SHA-256, source commit/dirty state, config file/hash, manifest and labels hashes, source/inventory/v3 fingerprints, class map, model architecture and framework versions, seed, preprocessing, device, train/validation metrics, timings and artifact checksums. Run IDs must be unique safe identifiers; never overwrite an artifact directory.

CNN resume checkpoints currently save model/optimizer/scaler states, epoch, best state, history and Python/NumPy/Torch RNG state once per epoch using a sibling `.pending` file followed by replacement. Resume now rejects changes to training config except a non-decreasing `fine_tune_epochs`; it remains epoch-boundary recovery, not an exact mid-epoch continuation. HOG-SVM has no resume and its complete feature matrices are in memory. Neither an incomplete checkpoint nor a smoke run is a release candidate.

## 7. Evaluation execution order

1. Validate config/dataset fingerprints and inspect compute capacity.
2. Train each approved candidate using train only; monitor validation only.
3. Evaluate candidate metrics on validation and rank under Section 5.
4. Freeze the single winner, exact artifact digest, preprocessing and inference benchmark protocol in a new immutable candidate-freeze record.
5. Evaluate only that winner on the full test split once. Test output is final reporting only.
6. Verify weights, checksums, 101-class labels, finite outputs, preprocessing compatibility and full-v3 manifest binding.
7. Publish via an explicit release action only after review; update the production pointer atomically and only then run the real API inference smoke test.

The existing command line does not currently implement steps 4–7 in this safe order for v3. They are mandatory Phase 9 release-tooling work, not implied by this document.

## 8. Revision policy

This file is protocol v1.0. Do not edit it in place after training or test results are observed. Change to dataset binding, metric definitions, split roles, preprocessing or ranking requires a new versioned protocol and a documented reason before the affected experiment begins.
