# PHASE 2.6 — Clean Split v2 Implementation

**Outcome: PASS** for implementation, synthetic/regression tests, raw baseline preflight, and clean split quality. Official train/test leakage remains **FAIL** by design and is reported independently; only Clean Split v2 is leakage-free.

## Baseline and approval

The initial Phase 2.6 inspection found `dataset_scope: poc`; that configuration was not suitable for resolving full Food-101 leakage. This report supersedes the earlier BLOCKED analysis after the user approved a dedicated full-scope configuration. The existing POC config and v1 remain unchanged.

Phase 2.5 baseline: 101 classes; 101,000 raw images; 75,750 official train IDs; 25,250 official test IDs; 0 missing/unlisted files; 101,000 successfully decoded images; 8 duplicate content groups and 2 cross-split hashes. Historical quality gate remains FAIL because official train/test contain exact duplicate content.

## Files reviewed

- `promts/food101_elt_prompt_pack/prompts/00_GUARDRAILS.md`
- `artifacts/reports/elt/baseline.md`
- Phase 2.5 reports/checkpoint under `data/reports/elt/v2/phase_2_5_verification_retry/`
- This phase's prior analysis: `data/reports/elt/v2/phase_2_6_analysis/phase_2_6_report.md`
- `docs/food101_elt_architecture.md`
- `configs/food101_elt.yaml`, `configs/data.yaml`, `configs/classes_poc.yaml`
- `src/food_assistant/ml/food101_verifier.py`, `datasets.py`, `transforms.py`, `cli.py`; `scripts/verify_food101.py`
- `src/food_assistant/data_pipeline/atomic.py`, `load.py`, `config.py`
- `tests/unit/test_food101_verifier.py`, `test_food101_data_safety.py`, `test_food101_elt_config.py`; POC preprocessing/ML tests

## Files changed

- Added `configs/food101_elt_full_v2.yaml` and `configs/classes_full.yaml` (class order derived from raw official `meta/classes.txt`).
- Added `src/food_assistant/data_pipeline/splits.py` and `scripts/build_clean_food101_split.py`.
- Extended `src/food_assistant/data_pipeline/config.py` with validated split policy fields; old POC config retains backward-compatible defaults.
- Extended `src/food_assistant/ml/datasets.py` to resolve relative metadata roots while retaining absolute-root v1 behavior.
- Added/extended `tests/unit/test_food101_clean_split.py` and `tests/unit/test_food101_elt_config.py`.
- Published `data/splits/v2/` (manifest, train/test CSVs, labels, metadata and internal quality report).
- Wrote external report `data/reports/elt/v2/phase_2_6/clean_split_v2_report.json`.
- Updated this Phase 2.6 report.

## Full config and implementation

Full config: `configs/food101_elt_full_v2.yaml`.

- Scope `full`; classes follow all official `classes.txt` entries in stable order.
- Split version `v2`; output `data/splits/v2`; report root `data/reports/elt/v2/phase_2_6`.
- Downloads disabled; preserve official test; exclude every train ID whose SHA-256 occurs in test; no validation; reject cross-partition duplicates.
- Build uses a sibling staging directory and resource-scoped lock, validates before same-filesystem rename, refuses overwrite, validates matching existing output before SKIP, and reports fingerprint/config conflicts.
- Manifest is CSV-compatible with the existing `ManifestDataset` reader and uses relative image paths. Schema discriminator is `2.0`; v1 remains readable with its legacy absolute root.
- Split fingerprint binds config, ordered image IDs/partitions/class IDs/checksums, and exclusion provenance. Raw/checkpoint signatures are checked before build and again before publish.
- Phase 2.5's complete schema-2 checkpoint envelope, per-image valid/decode/SHA evidence, current filesystem signatures, official metadata SHA values, inventory fingerprint, source marker, and source fingerprint were validated. This reuses the approved verified inventory evidence and detects changes without re-decoding every image.

## Duplicate analysis

All eight Phase 2.5 duplicate groups were mapped against official `meta/train.txt` / `meta/test.txt`. Hashes and resolutions were also checked on the corresponding files.

| SHA-256 | Class(es) | Train image(s) | Test image(s) | Resolution |
|---|---|---|---|---|
| `ec9b9f1437dde19e478d277d3519678d11f64435fbcf6f62ed965d048c9de9fe` | churros | `images/churros/2547960.jpg` | `images/churros/2617186.jpg` | 512×512 |
| `6579d6a6d246c0ab172f9aa125ef8e30c5a80aa84d4a50b8552252e4c1ab7670` | dumplings | `images/dumplings/167000.jpg`; `images/dumplings/180290.jpg` | — | 512×384 |
| `a36278a3acab370553ca36ab48ee3d95bd5960907d75f2b96c49668ebf997d9c` | hamburger | `images/hamburger/3578279.jpg`; `images/hamburger/3783650.jpg` | — | 332×512 |
| `24a7003393e446af26422ce6b4e08224d02ba92f54a86a42ec65ea9044f4e0b7` | omelette / pancakes | `images/omelette/2619753.jpg`; `images/pancakes/2619752.jpg` | — | 512×512 |
| `f40544646c0e07fcc3e6539bd03af71949dabb4b79515bfe207ca66e5f1c6f54` | paella | `images/paella/2199941.jpg` | `images/paella/2199939.jpg` | 512×512 |
| `398a960aa99898499a7565c7e277cda61b3c18cd7460448068242cf9a2a9035a` | pho | `images/pho/4787.jpg`; `images/pho/51112.jpg` | — | 384×512 |
| `d7ad31a0c3bcc83f20b42057d0575c6c1b956e9869c25095d32c1cdd5fe12ed0` | pho | `images/pho/51122.jpg`; `images/pho/651427.jpg` | — | 512×384 |
| `a9da5a645cb2c9f28cbe90ede5b09ed31ce27180dd5b42920dad46d8c988077b` | prime_rib / steak | `images/prime_rib/76271.jpg`; `images/steak/97656.jpg` | — | 512×512 |

The two cross-split groups were verified exactly: Churros train `2547960.jpg` / test `2617186.jpg`; Paella train `2199941.jpg` / test `2199939.jpg`. Each has one train exclusion. Six duplicate groups are train-only and remain in v2. The two cross-class groups (`omelette`/`pancakes`, `prime_rib`/`steak`) are reported as `label_inconsistency_risk`; no labels/images were modified.

## Split v2

- Output: `data/splits/v2/`
- Manifest/labels schema: `2.0`, dataset version `v2`; 101 class labels.
- Train: **75,748**; test: **25,250**; validation: **0**.
- Official test IDs preserved: **25,250 / 25,250**.
- Excluded train references: `churros/2547960` matched to `churros/2617186`, and `paella/2199941` matched to `paella/2199939`; reason `exact_sha256_match_in_official_test` and hash recorded per exclusion.
- Inventory SHA-256: `b91c327114b45c4fe2e626404a86c449891e6e258c832e67a7f2350aeb59adb9`.
- Source fingerprint: `f5cfdb7480c2650ed94c8624652ac595e4a7f038d660f4886869836632c56ed1`.
- Config hash: `2f0579ad6b85d09a14d3585386fbe81132389c33f07e5397bfdce9c73b878b95`.
- Split fingerprint: `319780d638052e9bfddbf28aa9e3df9bda9d9d78224a220ef7603b3d02bb150e`.
- Class counts: all 750 train / 250 test except churros and paella at 749 train / 250 test.
- Clean manifest contains six intra-train duplicate hash groups and zero cross-partition duplicate hashes.

## Quality gates

- **Raw Integrity: PASS.** Phase 2.5 full audit/checkpoint reported all 101 classes and 101,000 decodable verified images with no missing/unlisted files. Current signatures and official metadata/source/inventory fingerprints match that evidence.
- **Official Split Leakage: FAIL.** Two exact hashes still cross the untouched official train/test partitions. This status was retained and is explicit in split report/metadata.
- **Clean Split v2: PASS.** Manifest audit passed; no duplicate IDs, missing image references, invalid mapping, or cross-partition hash duplicates; full test ID set and exclusions match policy; metadata/report fingerprints agree.

## Tests and commands

- **RUN/PASS:** `.venv-311\Scripts\python.exe -m pytest tests/unit/test_food101_clean_split.py tests/unit/test_food101_elt_config.py tests/unit/test_food101_verifier.py tests/unit/test_food101_data_safety.py tests/unit/test_poc_preprocessing.py tests/unit/test_ml_pipeline.py -q` — **63 passed**.
- **RUN/PASS:** Ruff check over the changed Python modules, CLI and tests — all checks passed.
- **RUN/PASS:** `scripts/build_clean_food101_split.py` read-only preflight — raw/source/inventory fingerprints matched and calculated 75,748 train / 25,250 test.
- **RUN/PASS:** Atomic publish through `build_clean_split(..., publish=True)` — state `PASS`.
- **RUN/PASS:** Post-publish `ManifestDataset(data/splits/v2, "train")` validation and explicit check of manifest counts, 101 labels, all official test IDs, exclusions, external/internal report fingerprint, and recomputed split fingerprint.
- Synthetic coverage includes full/POC config parsing, unsafe policy/path/version rejection, multiple train duplicates against test, preserved test IDs, train-only duplicates, cross-class risk, path/hash/label/manifest checks, legacy absolute-root v1 reader compatibility, raw/checkpoint mismatch, conflict/skip, publish interruption, concurrent processes, failure without complete metadata, and side-effect-free preflight/config load.

## Safety check

- Raw image tree, `download_metadata.json`, and official metadata were never written; source/inventory/signature checks matched the pre-publish Phase 2.5 evidence.
- `data/splits/v1`, POC config, and model artifacts were not changed. Safety diff check was empty for `data/raw`, `data/splits/v1`, `artifacts/models`, and `artifacts/production`.
- No download, training, image deletion, or official split modification was performed.

## Acceptance and next phase

**ACCEPTANCE: PASS** for Phase 2.6 implementation and clean split v2. The official split leakage gate remains FAIL and must not be interpreted as clean; v2 resolves leakage only in the new manifest. POC workflows are not switched to v2.

**RISKS/OPEN ITEMS:** The two cross-class duplicate groups remain unchanged and may have label inconsistency. Six intra-train exact duplicate groups remain in v2 by policy. The raw integrity evidence is the complete Phase 2.5 checkpoint plus verified current signatures, rather than a new full decode pass.

**NEXT PHASE:** Phase 3 — Downloader Retry/Resume.

## Verification note

The first post-publish fingerprint comparison exposed a type-normalization bug in the checker: CSV `class_id` values are strings while build-time values are integers. The checker now normalizes class IDs before digest calculation. No manifest bytes were changed for this fix. The subsequent full `ManifestDataset` reader audit, test-ID comparison, split-fingerprint comparison, and final 63-test regression run all passed.
