# PHASE: 2.7 — Training Data Quality & Validation Split v3

## BASELINE

- Raw Food-101: 101 classes / 101,000 images; Phase 2.5 source fingerprint `f5cfdb7480c2650ed94c8624652ac595e4a7f038d660f4886869836632c56ed1`; inventory SHA-256 `b91c327114b45c4fe2e626404a86c449891e6e258c832e67a7f2350aeb59adb9`.
- Clean Split v2 passed with 75,748 train and 25,250 test images; v2 fingerprint `319780d638052e9bfddbf28aa9e3df9bda9d9d78224a220ef7603b3d02bb150e`.
- v2 train contained six exact-duplicate groups: four same-label groups and two cross-label groups. There was no duplicate across v2 train/test.
- v1, raw images, official split metadata and model artifacts were not inputs to any write operation.

## FILES REVIEWED

- `promts/food101_elt_prompt_pack/prompts/00_GUARDRAILS.md`
- Phase 0 baseline, Phase 2.5 verification, and Phase 2.6 reports; Phase 1/2 contracts in `docs/food101_elt_architecture.md`.
- `configs/food101_elt.yaml`, `configs/food101_elt_full_v2.yaml`, `configs/data.yaml`, `configs/classes_poc.yaml`, `configs/classes_full.yaml`.
- `src/food_assistant/data_pipeline/config.py`, `splits.py`, `atomic.py`, `src/food_assistant/ml/datasets.py`, and related CLI/test files.
- Actual v2 metadata, quality report, manifest, labels and Phase 2.5 checkpoint/report.

## FILES CHANGED

- Added `configs/food101_elt_full_v3.yaml`.
- Updated `src/food_assistant/data_pipeline/config.py` with validated v3 source and split-policy fields.
- Added `src/food_assistant/data_pipeline/split_v3.py` and `scripts/build_food101_split_v3.py`.
- Added `tests/unit/test_food101_split_v3.py`; extended `tests/unit/test_food101_elt_config.py`.
- Published new output under `data/splits/v3/` and reports under `data/reports/elt/v3/phase_2_7/`.
- No v1/v2, raw, official metadata or model artifact file was changed by this phase.

## CONFIG V3

- Config: `configs/food101_elt_full_v3.yaml`; full scope with official 101-class order.
- Source manifest: `data/splits/v2`; output: `data/splits/v3`.
- Seed: 42; validation fraction: 0.10; validation method: class-stratified `train_test_split` after SHA-256 groups are collapsed/quarantined, using scikit-learn 1.9.1.
- Download policy disabled; official test preserved; same-label policy keeps lexicographically smallest ID; cross-label policy quarantines every reference; cross-partition duplicates rejected.
- Config hash: `51555b3a95c8dfcf50ea6edec96f6e26edd576710985e47453cd6a1f11863963`.

## DUPLICATE CLEANING

Four same-label groups were deduplicated, one representative per SHA-256 retained:

| Class | Representative | Excluded reference |
|---|---|---|
| dumplings | `dumplings/167000` | `dumplings/180290` |
| hamburger | `hamburger/3578279` | `hamburger/3783650` |
| pho | `pho/4787` | `pho/51112` |
| pho | `pho/51122` | `pho/651427` |

Two cross-label groups were quarantined in full with status `LABEL_CONFLICT_QUARANTINED`:

| SHA-256 | Image IDs | Labels |
|---|---|---|
| `24a7003393e446af26422ce6b4e08224d02ba92f54a86a42ec65ea9044f4e0b7` | `omelette/2619753`, `pancakes/2619752` | omelette / pancakes |
| `a9da5a645cb2c9f28cbe90ede5b09ed31ce27180dd5b42920dad46d8c988077b` | `prime_rib/76271`, `steak/97656` | prime_rib / steak |

Total exclusions: 8 references (4 deduplicated and 4 quarantined). All exclusions include source version, original partition, class, digest, reason, and matching/representative IDs in `data/splits/v3/excluded_images.json`. No files were removed or relabeled. Remaining exact duplicates: 0 within train and 0 across train/validation/test.

## SPLIT V3

- Output: `data/splits/v3/`; manifest schema 3.0 (`dataset_manifest.csv`, plus train/val/test CSVs and labels).
- Train: 68,166; validation: 7,574; test: 25,250; total: 100,990 manifest references.
- All 101 classes remain in train. Validation has all 101 classes. Test counts and order match v2 exactly (250/class).
- Train is 675/class except 674 for `churros`, `dumplings`, `hamburger`, `omelette`, `paella`, `pancakes`, `pho`, `prime_rib`, and `steak`. Validation is 75/class except `pho` with 74. Detailed distributions are persisted in metadata/report.
- Raw source fingerprint: `f5cfdb7480c2650ed94c8624652ac595e4a7f038d660f4886869836632c56ed1`.
- v2 fingerprint: `319780d638052e9bfddbf28aa9e3df9bda9d9d78224a220ef7603b3d02bb150e`.
- v3 fingerprint: `aa43cfa766982006fe6a3611a49e0f07d35d33d56e54310e923cb5e58fbdbb59`.
- `ManifestDataset` read the published train split and audited paths/checksums for the complete manifest successfully.

## QUALITY GATES

- Source Integrity: PASS — current raw evidence matches Phase 2.5; v2 metadata/report/config/source/split fingerprints agree.
- Train Data Quality: PASS — train has no repeated SHA-256; all cross-label duplicate references are quarantined; all 101 classes remain.
- Partition Independence: PASS — no duplicate IDs or hashes span train/validation/test; test IDs, labels, checksums and order equal v2.
- Manifest Quality: PASS — portable `images/...` references, valid mapping, all files/checksums readable, and metadata/report fingerprints agree.
- Overall: PASS; `data/splits/v3/dataset_metadata.json` has `complete: true` only after all gates passed.

## TESTS EXECUTED

- `.venv-311/Scripts/python.exe -m pytest tests/unit/test_food101_elt_config.py tests/unit/test_food101_split_v3.py -q` — 21 passed (including the config tests; subsequent focused v3 run also passed 10 tests after the test-order correction).
- `.venv-311/Scripts/python.exe -m pytest tests/unit/test_food101_verifier.py tests/unit/test_food101_data_safety.py tests/unit/test_food101_elt_config.py tests/unit/test_food101_clean_split.py tests/unit/test_food101_split_v3.py tests/unit/test_poc_preprocessing.py tests/unit/test_process_lock.py -q` — 76 passed.
- `.venv-311/Scripts/ruff.exe check src/food_assistant/data_pipeline/config.py src/food_assistant/data_pipeline/split_v3.py scripts/build_food101_split_v3.py tests/unit/test_food101_elt_config.py tests/unit/test_food101_split_v3.py` — passed.
- `python scripts/build_food101_split_v3.py` — read-only preflight passed against existing raw/v2.
- `python scripts/build_food101_split_v3.py --publish` — published successfully after staging validation and atomic rename.
- Post-publish audit — `ManifestDataset` train load passed; test rows/order, metadata/report consistency, 101 labels and v3 fingerprint assertions passed.
- An initial preflight exposed a test-order issue; it was corrected, covered by synthetic tests, and the subsequent preflight and publish passed. The diagnostic is retained in `failure_report.json` separately from success metadata.

## SAFETY CHECK

- No download or training command was run.
- Raw SHA-256 inventory and source fingerprint were revalidated immediately before publish and matched the approved baseline. Raw images and official metadata were read only.
- v1, v2 and model artifacts were not written. The v3 builder only reads v2; all new split files and reports are confined to versioned v3 paths.
- Output creation used a resource-scoped lock, a same-filesystem staging directory, full validation and atomic rename. Existing output is validated and skipped only when fingerprints match; conflicts are rejected.

## ACCEPTANCE

PASS.

## RISKS/OPEN ITEMS

- The quarantined cross-label groups require manual label review before any later policy considers restoring either reference. No label correctness was inferred.
- A prior failed preflight diagnostic remains at `data/reports/elt/v3/phase_2_7/failure_report.json`; it records the corrected test-order issue and is separate from passing metadata/report.

## NEXT PHASE

Phase 3 — Downloader Retry/Resume.
