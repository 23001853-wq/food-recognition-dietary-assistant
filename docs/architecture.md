# PLAN v2 — Food Recognition & Portion-Aware Dietary Assistant

Updated 2026-10-31. This supersedes the original architecture at the user's explicit request.
The full original plan remains in [PLAN_v1_historical.md](PLAN_v1_historical.md) for research detail.
Status evidence lives in [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md).

## Scope and architecture

Solo developer. Streamlit → HTTP → FastAPI → application services → PostgreSQL.
The backend calls ImageService, CropService, InferenceService/ModelRegistry,
NutritionService, MealService and DashboardService. UI never imports SQL/training.
FastAPI is now part of MVP. React, login, multi-user authorization, detectors and
segmentation remain future work. Local demo binds ports to loopback.

Offline: approved dataset → versioned manifests/labels → HOG/SVM, ResNet18,
EfficientNet-B0 → validation → frozen candidates → independent test report → release.
Training never runs during an upload. Model selection uses validation macro-F1,
with measured latency as tie breaker; final test results are reported separately.

## Contracts

- `PredictionResult`: schema_version=1.0, prediction_id UUID, model identity and
  dataset_version, is_mock, predictions with persisted class_id/class_key, scores and latency.
- Mock is opt-in, deterministic and explicit. APP_ENV=production rejects ALLOW_MOCK_AI=true.
- One image/crop → one primary class; multiple dishes require separate manual crops.
- Missing real checkpoint produces MODEL_NOT_READY, never an automatic mock fallback.
- Model artifacts have checksums, exact label maps, preprocessing and dataset identity.
  Only trusted local artifacts are loaded; pickle/joblib are never accepted from uploads.
- Smoke-only training is never production eligible. Synthetic data is only a test fixture.
- `client_request_id` makes meal creation idempotent; conflicting bodies return 409.
- PostgreSQL mutation transactions complete before a successful response.
- Date filters use the configured local calendar. Database timestamps are UTC-aware.

## Persistence

Preserve original names: users, food_classes, nutrition_foods, food_nutrition_mappings,
model_versions, prediction_history, meal_logs, meal_items. They are the equivalent
structures requested as nutrition_facts/portion_profiles/model_registry:
nutrition_foods holds nutrients per 100g plus nullable S/M/L mass profiles;
model_versions is the release audit catalog, with the filesystem pointer as runtime truth.
Food class UUIDs are independent from dataset class indices.

Meal items snapshot kcal/macros, source URL, source/version and mapping assumptions.
Updating reference data does not rewrite history. Metadata-only meal patches preserve
snapshots; explicit item edits recalculate using current references after UI confirmation.
Deleting the last item is implemented as deleting its entire meal transactionally.

Prediction response metadata is stored to support retrieval and provenance. Raw images
are not stored by default. The original SAVE_PREDICTION_HISTORY switch is retired in v2:
the API persistently stores metadata, while SAVE_UPLOADED_IMAGES controls image retention.

## Dataset governance

The user confirmed Food-101 academic approval on 2026-10-31. Official ETH source and
citation are recorded in configs/dataset_approval.json. Use is local academic research;
do not redistribute source images. Inspect the archive README/terms after download.
No formal dataset counts are claimed until extraction and audit have completed.

Food-101 original test is immutable. Validation is stratified from original train only.
Class order is frozen in labels.json. Manifests include checksums and leakage checks;
duplicates crossing splits block preparation rather than silently changing the benchmark.
Custom Vietnamese datasets require rights metadata and capture-group-disjoint splits.
The proposed Vietnamese classes are configuration, not evidence of collected images.

## Nutrition

`kcal = kcal_per_100g × grams / 100`; same scaling for protein/carbohydrate/fat.
Null stays null; score never scales nutrition. Values are sourced references and
selected/entered masses, not measurements from image pixels.
Current curated example: USDA SR Legacy FDC 170317, regular-crust cooked cheese pizza.
S/M/L 100/200/300g is an explicit demo convention. Other recipes are not equivalent.
User-entered kcal is marked manual and never relabeled as a sourced estimate.

## Acceptance and remaining work

Mock end-to-end acceptance requires live PostgreSQL, API, UI and CRUD/dashboard tests.
Formal AI acceptance additionally requires the complete audited dataset, pretrained
fine-tuning, independent test evaluation on common labels/samples, error analysis and
a release selected from real evidence. See the implementation/test/evaluation reports.
No metrics, images or trained-model status may be invented.
