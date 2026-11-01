# Training and trusted artifacts

Current POC/GPU workflow: [POC_REAL_AI.md](POC_REAL_AI.md). Use a separate .venv-cuda;
the existing CPU HOG worker is preserved. POC evaluation now requires a protocol
created by freeze_poc.py after all three candidates finish. Generic evaluation examples
below must include `--protocol <frozen_protocol.json>` when dataset_version is poc-v1.
AMP and gradient accumulation are implemented; GPU execution has not yet been verified.

Use Python 3.11 and a compatible Torch/Torchvision pair. Current host pair:
torch 2.9.1+cpu / torchvision 0.24.1. NVIDIA RTX 3050 4GiB is present, but this
installed PyTorch build cannot use CUDA. Do not claim GPU training from nvidia-smi alone.

```powershell
.venv/Scripts/python.exe scripts/train_hog_svm.py --manifest-dir data/splits/v1
.venv/Scripts/python.exe scripts/train_resnet18.py --manifest-dir data/splits/v1 --device cpu
.venv/Scripts/python.exe scripts/train_efficientnet_b0.py --manifest-dir data/splits/v1 --device cpu
```

CNN defaults download ImageNet-1K V1 weights; they are then trained on the actual food
manifest. Epochs/batch size/LR are in configs/*.yaml. Head training freezes backbone
parameters and batch-normalization statistics; fine-tuning unfreezes the backbone.
Seed and manifest digest are recorded. Best checkpoint uses validation macro-F1;
patience defaults to 5. CSV logs contain actual loss/validation F1. No test tuning.

Use `--smoke` for two train and two validation samples per class and one CNN epoch.
`--smoke --no-pretrained` exercises the architecture without downloading weights;
it is not transfer learning and never production eligible. CPU ML tests use synthetic
fixtures, not food data, and clean temporary artifacts are not model releases.

`--resume path/to/resume.pth` resumes optimizer/model epoch with manifest checks.
Use a new run ID. `--run-id` is optional; default UTC timestamp includes microseconds.
Existing run folders are never overwritten. Trusted checkpoints only.

```powershell
.venv/Scripts/python.exe scripts/evaluate_models.py --artifact artifacts/models/hog_svm/RUN --manifest-dir data/splits/v1
.venv/Scripts/python.exe scripts/analyze_failures.py --predictions artifacts/models/hog_svm/RUN/evaluation_test/predictions.json --output artifacts/reports/failure_cases/review.csv
```

Outputs: metrics.json, classification_report.csv, confusion_matrix.csv/png,
evaluation_summary.md and predictions.json. Training history is in the run root.
Label every error manually as visual similarity, mixed dish, plating variation,
lighting/occlusion or domain shift; automatic export uses UNREVIEWED, not invented causes.
Latency is measured on the chosen device with preprocessing included, batch size 1.

```powershell
.venv/Scripts/python.exe scripts/select_production_model.py --artifacts artifacts/models/hog_svm/RUN_A artifacts/models/resnet18/RUN_B artifacts/models/efficientnet_b0/RUN_C
```

All candidates must share a frozen manifest and have completed non-smoke test evaluation.
Rank by validation macro-F1; latency breaks ties. Test metrics are reported separately.
The production pointer is created only at this point. No production pointer currently
exists unless a later status update explicitly reports its verified release.

Select a specific development release via `model_name=architecture/run_id`; production
uses `model_name=production`. Artifact integrity, label indices, class count and dataset
version are verified before inference. SVM scores normalize decision margins for
ranking; CNN scores use softmax. Neither is calibrated probability.
