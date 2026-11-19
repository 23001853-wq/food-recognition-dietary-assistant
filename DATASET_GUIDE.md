# Dataset guide

Food-101 academic eligibility was confirmed by the user on 2026-10-31.
Source/citation: [ETH Zurich Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).
The official release describes 101 classes and the original 750/250 train/test per class.
Those are source specifications, not a claim that all files are present locally.
Archive rights/README must be inspected after download; do not redistribute images.
The official page does not publish a SHA256; local computed SHA256 is integrity evidence,
not an independently verified official checksum.

## Verifying the complete local release

The full audit is read-only for the source images and does not rebuild or overwrite the
frozen POC manifest. It checks the original 101-class metadata, every expected image,
Pillow decoding, SHA256 duplicates, and the retained archive against its local download
marker. Image checks are checkpointed by file size and filesystem timestamps; changed
files are checked again. A lock prevents two full audits from running at once.

```powershell
.venv\Scripts\python.exe scripts\verify_food101.py --mode quick
.venv\Scripts\python.exe scripts\verify_food101.py --mode full --dataloader --batch-size 4
.venv\Scripts\python.exe -m pytest tests\unit\test_food101_verifier.py -q
```

Reports, log, and resumable checkpoint are written under `data/reports/food101/`.
Quick mode checks structure and metadata but does not decode images or compute image
hashes. Full mode is required for a PASS. `--dataloader` checks a batch from the official
101-class train split and the existing POC 10-class manifest separately. A complete
DataLoader PASS also requires the training transform dependency (`torchvision`) to load.
The frozen manifest may contain a machine-specific root; the validation command uses a
temporary root override and does not edit the manifest files.
If a check fails, inspect `food101_report.json` for exact missing, unlisted, or unreadable
paths. The audit never repairs or deletes images. Restore only from a trusted copy or
archive, then rerun full mode; unchanged successful images are skipped using the
checkpoint. Exact duplicate paths and cross-split hashes are reported for review before
using official test metrics.

The project requires Python 3.11. If it is not installed, create the project environment
with `py -3.11 -m venv .venv`, then install the project's ML and development dependencies
before running the commands above. The preprocessing check needs the matching `torch` and
`torchvision` packages.

From repository root with the project venv:

```powershell
.venv/Scripts/python.exe scripts/download_food101.py
# Alternative: resumes bounded Range chunks after a slow/failed sequential transfer
.venv/Scripts/python.exe scripts/download_food101_parallel.py --workers 8
.venv/Scripts/python.exe scripts/create_splits.py --root data/raw/food-101 --output data/splits/v1
.venv/Scripts/python.exe scripts/verify_dataset.py --manifest-dir data/splits/v1
.venv/Scripts/python.exe scripts/audit_near_duplicates.py --manifest-dir data/splits/v1
```

Do not run two downloaders against the same output simultaneously. Existing complete
dataset count is checked before re-downloading. Chunk files are resumable and are not
auto-deleted. Allow at least 20GB free for archive/chunks/extracted files.

The default POC uses the 10 explicit classes in configs/classes_poc.yaml. For each class:
600 internal train, 150 validation and the unchanged 250 official test are intended.
Outputs: dataset_manifest.csv, train.csv, val.csv, test.csv, labels.json,
dataset_metadata.json and quality_report.json after verification. Paths/class IDs,
dataset version and checksum are stable; frozen output is not overwritten.
Duplicate cross-split hashes block preparation. Near-duplicate candidates require
manual review; official test is never silently rewritten to improve results.

## Vietnamese photos

No Vietnamese photos have been collected in this session. Proposed labels are in
configs/classes_vietnamese.yaml. Each catalog row must include:
`relative_path,class_key,capture_group,source_url,rights_confirmed`.
For your own photos source_url can be `self-captured:session-id` and rights_confirmed=true.
The same capture session must use the same capture_group across all related photos.

```powershell
.venv/Scripts/python.exe scripts/prepare_custom_dataset.py --root data/local_eval --catalog photos.csv --classes configs/classes_vietnamese.yaml --output data/splits/vietnamese-v1
```

This checks group separation and class coverage; insufficient groups produce an error.
Review visual similarity/duplicate labels before merging with any Food-101 classes.
Local domain-shift photos remain an independent evaluation dataset, never blended into
Food-101 test metrics. Record camera/lighting/background metadata in the source catalog.
