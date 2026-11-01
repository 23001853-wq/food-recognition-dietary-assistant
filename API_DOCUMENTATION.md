# REST API v1

Run backend and open http://127.0.0.1:8000/docs or /openapi.json.
The API is a local single-user demo; no login or multi-user authorization is implied.

| Method | Path | Behavior |
|---|---|---|
| GET | /health | 200 with DB connection, otherwise 503 degraded |
| GET | /api/v1/foods | Supported food catalog with UUIDs |
| GET | /api/v1/foods/{id} | Food detail |
| POST | /api/v1/predictions | Multipart file, model_name, top_k, optional JSON crop_box |
| GET | /api/v1/predictions/{id} | Stored standardized response |
| GET | /api/v1/nutrition/{food_id} | portion, optional grams/manual_kcal; source, status and nullable nutrients |
| POST | /api/v1/meals | Validated meal, one or more items, client_request_id UUID |
| GET | /api/v1/meals | start/end dates, optional meal_type |
| GET/PATCH/DELETE | /api/v1/meals/{id} | Read/edit/delete confirmed meal |
| GET | /api/v1/dashboard/summary | Totals, missing_count, daily, meal-type/frequent-food counts |
| GET | /api/v1/dashboard/daily | Daily data only |
| GET | /api/v1/models | Mock availability and local releases |

Prediction JSON Schema and explicit mock example are in contracts/. HTTP upload allows
JPEG/PNG only; MIME and actual decoded content must agree. Byte/pixel limits and EXIF
orientation are enforced. crop_box is normalized [x1,y1,x2,y2], minimum 8×8 pixels.
No endpoint accepts arbitrary file paths or uploaded model binaries.

Meal example (replace food UUID or use custom_food_name):

```json
{"consumed_on":"2026-10-31","meal_type":"lunch","client_request_id":"024b7644-c196-4640-b9b3-9cf6f3009ee1","items":[{"custom_food_name":"Món tự nhập","portion_size":"custom","grams":"150","manual_kcal":null}]}
```

Meal food UUID is not an artifact class_id. A new request ID is needed for a new meal.
Identical retries return the same ID; reuse with a changed body returns 409.
PATCH metadata preserves snapshots; PATCH items explicitly recalculates snapshots.
Delete the whole meal when removing its last item. API delete expects user confirmation
to happen in the UI. Unknown nutrition remains null and increments missing_count.

Error format: `{ "error": { "code": "MODEL_NOT_READY", "message": "..." } }`.
Pydantic request errors use standard FastAPI 422 detail. Database details/credentials are
never returned. Date ranges are inclusive with a maximum 366-day difference.

Nutrition response extends existing fields with `status` (available/unavailable/manual),
`source_type` (reference/user_input/unavailable), `source_name`, `source_record_id`,
`source_version`, `reference_name`, `portion_grams`, `mass_source`, `portion_note`,
`match_note`, and `unavailable_reason`. `manual_kcal` previews user energy without
changing reference data; macros remain null in this mode. Decimal fields serialize
as JSON strings. Default API portion remains custom for backward compatibility;
the recognition UI selects medium by default.

Meal items expose derived `source_type` and retain immutable nutrition snapshots.
New manual energy uses `nutrition_source_snapshot=user_input`; old records are unchanged.
Dashboard adds `known_count`, `reference_count`, `user_input_count`, and daily
`known_count`. Existing aggregate `estimated_kcal` is the sum of known values; always
interpret with counts. A sum of zero with only unknown items does not establish zero
energy; the UI displays unavailable for that case. See [NUTRITION_GUIDE.md](NUTRITION_GUIDE.md).

Dashboard summary now also includes `nutrients` for `estimated_kcal`, `protein_g`,
`carbohydrate_g`, `fat_g`. Each has `{total, known_count, missing_count}`; total is null
when no item has a value, and zero only when known values sum to zero. Each daily row
has the same `nutrients` structure. Aggregates read persisted snapshots, with independent
coverage per nutrient; manual kcal never creates inferred macros. Existing aggregate
fields/serialization remain compatible. Restart the updated backend to expose these fields.
Raw images are not retained unless SAVE_UPLOADED_IMAGES=true; response metadata is
persisted for retrieval/provenance. Treat metadata as private local demo data.
