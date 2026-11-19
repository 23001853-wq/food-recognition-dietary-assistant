# Food-101 filesystem ELT architecture (Phase 1)

## Baseline and scope

Phase 0 is reported as `PASS` in `artifacts/reports/elt/baseline.md`. It observed a full raw tree (101 classes, 101,000 JPEG paths) but did not freshly revalidate that evidence; the earlier integrity report is historical and `INCOMPLETE`, with two exact duplicate-content groups crossing the official split. A `.part` archive was present at baseline and a complete archive was absent in that observation. The current tree contains `data/raw/food-101/{images,meta}`, metadata/license files and a pre-existing `data/splits/v1`; this phase did not inspect every image or refresh those reports. POC v1 has 10,000 images across 10 classes and remains immutable. Full Food-101 (101,000 / 101 classes) and the POC subset are separate scopes and must never share a success claim.

This phase defines contracts and a side-effect-free config loader. It does not publish a runner, downloader, migrate files, or change manifest v1. The default download policy is disabled. Any future download requires both an explicit config policy and confirmation in `configs/dataset_approval.json`.

## Observed and target flow

```text
Observed:
download_food101.py / download_food101_parallel.py
  -> raw tree and download marker (extract directly into output today)
  -> verify_food101.py / food101_verifier.audit_food101 (full audit + reports)
  -> ml.cli.create_splits -> datasets.prepare_food101 -> data/splits/v1
  -> ml.cli.verify_dataset / ManifestDataset -> transforms.build_transform -> callers' DataLoader

Target (runner belongs to Phase 5):
config validation (read-only)
  -> Extract: approved source archive + origin/byte evidence (optional; disabled by default)
  -> Load: safe staged extraction -> structural and full integrity gate -> atomic first publish to raw
  -> Transform: source verification -> quality gate -> immutable manifest vN + labels/config snapshot
  -> final audit/report and state record
```

Extract may be skipped only when a verified archive or verified raw source matches the requested fingerprint and provenance. Load may be skipped only when raw verification proves the same archive fingerprint, expected contract, and complete marker; existence/count alone is insufficient. Transform may be skipped only when an immutable manifest has matching schema, config hash, input fingerprint, output hashes, and a passing audit. No downstream stage may run after `FAIL`, `INCOMPLETE`, or `BLOCKED` upstream quality. An `INCOMPLETE` verifier report is never promoted to `PASS` merely by resuming.

## Stage contracts

| Stage | Input | Output | Schema | Fingerprint | Preconditions | Success proof | Failure states | Skip/resume rule | Side effects | Rollback |
|---|---|---|---|---|---|---|---|---|---|---|
| Config | YAML, approval and class config paths | Validated in-memory config | `food101_elt.yaml` schema 1 | Canonical normalized config SHA-256 | Types, ranges, supported version, workspace path containment, no output overlap with raw or v1 | Parsing + validation only; no filesystem mutation | `FAIL` invalid/missing; `BLOCKED` unsafe path | Always re-read; no cache | None | None |
| Extract (E) | Approved URL or existing archive; approval record | Complete archive plus origin metadata | Archive metadata schema v1: origin, retrieval time, byte length, SHA-256, official checksum result, validator version | Archive SHA-256 + HTTP validators; partial identity also includes expected total/range | Download policy `approved_only`, approval explicitly confirmed, resource lock, valid resume evidence | Exact total length, HTTP status/range, expected offset, validator consistency, digest; official digest verified when supplied | `BLOCKED` approval/policy; `INCOMPLETE` interrupted transfer; `FAIL` bad range/digest; `FAIL` lock conflict | Reuse only full digest-verified archive; `.part` is never trusted solely by length | Archive and provenance in archive location | Retain previous partial; quarantine/remove only pipeline-owned invalid temp under explicit recovery; never overwrite raw |
| Load (L) | Verified archive OR existing raw tree plus source metadata | Raw Food-101 tree and verification marker | Official ETH layout; marker schema v1 binds archive/source fingerprint, expected inventory, audit version, status | Source archive SHA-256 + canonical metadata + sorted relative paths and image-content SHA-256 inventory | Safe extraction; staging on same filesystem; scoped raw writer lock; raw absent for publish, or exact already verified same source | Metadata/path safety checks; full image decode/hash; expected 101 classes and 101,000 images for full or declared fixture contract; marker `PASS`; atomic no-replace directory publish | `BLOCKED` raw exists unverified/different; `FAIL` traversal/links/special/duplicate paths/limits/integrity; `INCOMPLETE` interrupted stage | Existing raw can skip only with valid marker and current fingerprint/integrity evidence; incomplete cache fails closed and is rebuilt in a new staging dir | New staging, report, raw publish only when destination absent | Delete only unpublished owned staging on explicit recovery; published raw is immutable, never rollback by replacing it |
| Quality gate | Raw source, expected contract, verifier outputs | Gate report | `PASS|FAIL|INCOMPLETE` with criterion details | Input inventory and verifier version/config hash | Full decode/checksum for a PASS; duplicate crossing splits is FAIL by default; no unknown criteria | All required criteria PASS; warnings retained; full vs POC expected counts are explicit | `FAIL` integrity/duplicates; `INCOMPLETE` skipped checks, stale evidence or untrusted checkpoint | Reuse only if fingerprint and verifier/schema versions match; invalid checkpoint fails closed | New versioned report/checkpoint only | Keep older reports; no mutation of raw/v1 |
| Transform (T) | Passing raw audit, class-selection config, split policy | New immutable manifest directory, labels, metadata, config snapshot, quality report | New manifest schema/version; CSV field semantics may reuse `datasets.FIELDS`; relative image paths and explicit `root_override` | Raw inventory fingerprint + selected classes + seed/fraction + transform contract + config hash; output hashes per artifact | Destination version absent; classes unique and known; seed and val fraction valid; official test entries unchanged; duplicate crossing splits resolved/rejected | Reopen outputs with `audit_manifest`, verify hashes/schema/class mapping/split counts, confirm output hashes; atomic directory publish only if destination absent | `FAIL` leakage/schema/hash; `BLOCKED` gate failure or occupied destination; `INCOMPLETE` interrupted build | Reuse matching immutable output only after complete proof; never edit a published version | Versioned new manifest and report only | Remove only unpublished staging; preserve v1 and all prior versions |

Full dataset contract means official Food-101: 101 classes, 75,750 original train entries, 25,250 official test entries, 101,000 images. POC means the frozen 10-class/10,000-image v1 for compatibility, or a newly named POC version produced from the approved class list; it does not imply all 101 classes are locally verified. Validation is sampled deterministically only from original train; official test metadata is never repartitioned. Any exact duplicate crossing splits fails the gate unless a separately reviewed, versioned policy explicitly resolves it; v1 remains untouched.

## Sources of truth

- **Raw:** configured `raw_root`, plus a versioned verification marker binding source identity, exact inventory, content hashes, and full integrity result. Folder presence or a count is not proof.
- **Archive:** configured archive plus separate provenance metadata. A local SHA-256 proves identity consistency only; it is not an official checksum unless compared to an independently supplied official value.
- **Manifest:** each `manifest_root/<version>` is immutable after atomic publication. Its metadata binds schema, relative paths, config snapshot/hash, source fingerprint, class mapping, split policy, and artifact hashes. `data/splits/v1` remains a readable legacy version and is never overwritten.
- **Config:** the exact validated YAML bytes and referenced class config are snapshotted and hashed per run. Approval is read from its configured file immediately before a future network action.
- **Quality:** versioned reports/checkpoints are evidence, not authority if their fingerprint, schema or verifier version is stale or malformed.

## Run state and transition rules

State records are append-only, versioned JSON in a new run-state directory. Each run has `run_id` (UUID), schema version, config hash, input fingerprint, stage, state, UTC start/end timestamps, output hashes, `error_category`, concise error detail and optional `skip_reason`. Valid stage transitions are `PENDING -> RUNNING -> {PASS, FAIL, INCOMPLETE, BLOCKED, SKIPPED}`. A resumed attempt creates a new attempt record linked to the prior run; terminal `INCOMPLETE` never transitions in-place to `PASS`. `SKIPPED` requires proof of matching validated output and records that proof as `skip_reason`; otherwise it is `BLOCKED`. A dependent stage starts only after upstream `PASS` (or a proof-backed `SKIPPED` whose recorded output remains valid). Categories include `CONFIG`, `APPROVAL`, `NETWORK`, `HTTP_RANGE`, `CHECKSUM`, `ARCHIVE_SECURITY`, `LOCK`, `IO`, `SCHEMA`, `QUALITY`, and `INTERNAL`.

## Configuration and validation

`configs/food101_elt.yaml` defines raw/staging/archive/manifest/report paths; approval and class-selection files; full vs POC scope; new split version; deterministic seed and validation fraction; disabled-by-default download policy, retries/backoff; quality policy; lock timeout; and test-only fixture switch/root. `load_pipeline_config()` validates YAML shape and values, disallows unsupported/frozen `v1`, enforces workspace containment and refuses staging/archive/output paths inside raw or v1. Config loading has no side effects. Future runner validation must also validate class-file contents, approval schema, path symlinks/reparse points, and same-filesystem staging before any write.

## Module boundaries

Keep future implementation under `src/food_assistant/data_pipeline/`: `config.py` owns parsing/validation; `extract.py` owns source/archive transfer and provenance; `load.py` owns staging, secure extraction and raw publication; `quality.py` adapts the existing `food101_verifier`; `transform.py` orchestrates the existing `prepare_food101`, `audit_manifest`, and `ManifestDataset` contracts without copying their logic; `state.py` owns run records and transitions; `atomic.py` owns no-replace atomic file/directory publication; `pipeline.py` composes stages. `utils/process_lock.py` currently protects model training outputs, not shared dataset/archive resources. Do not reuse that training lock; add resource-scoped locks with cross-process semantics and contention tests in Phase 2. `ml/transforms.py` remains the image transform authority.

## Portability and recovery

New manifest metadata stores paths relative to the configured dataset root, with a controlled root override at load time; legacy v1 absolute `root` remains readable through its existing override. Never rewrite v1 to make it portable. Preserve the original POC CLI and `ManifestDataset` behavior as the migration fallback. Existing raw directories or partial archives that lack matching, trusted provenance are treated as `BLOCKED` for automatic reuse. A new pipeline may inspect them read-only and produce a report; it may not clean, replace, or extract over them. All temporary writes use uniquely named sibling staging paths on the same filesystem. Dry-run validates and plans without lock creation, staging, report writes, or run-state writes. Rerunning an already proven stage returns proof-backed `SKIPPED` and leaves validated outputs byte-identical.

## Phase 1 files and limitations

Phase 1 added this architecture, the default YAML config, and a read-only config loader with regression tests. It did not create a runner, download, extract, audit, migrate, or mutate raw/v1/model artifacts.

## Phase 2 implementation status

Phase 2 adds sibling-temp atomic JSON/text/CSV writers, resource-scoped OS locks, archive extraction into a unique sibling staging directory, validation before no-replace directory rename, and first-publish-only manifest generation. The legacy download scripts now refuse an existing raw tree, lock archive/chunk writers, validate the extracted full Food-101 image inventory and decodes before embedding a `complete: true` marker, and use the staged publisher. The verifier checkpoint uses schema 2, an expected-path input hash, per-file signatures, a completion flag, and a SHA-256 envelope; malformed or mismatched evidence is discarded. The verifier's report files are atomic. The legacy manifest verification CLI writes its report outside v1. No runner or retry/resume redesign was introduced.

On Windows, publication uses `Path.rename` from a same-volume staging sibling while holding the destination's resource lock. The code first rejects an existing target. Windows rename fails if another process has created that target; participating writers serialize on the same persistent lock file. It does not call `os.replace` on a non-empty directory. Invalid extraction removes only its uniquely named staging directory; raw and v1 are never cleanup targets.
