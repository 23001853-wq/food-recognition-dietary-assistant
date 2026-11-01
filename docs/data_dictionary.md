# Current data dictionary

Generated from SQLAlchemy ORM; see DATABASE_GUIDE.md for design rules.

## food_classes

| Column | Type | Nullable | Keys |
|---|---|---|---|
| dataset_key | VARCHAR(150) | False |  |
| display_name_en | VARCHAR(200) | False |  |
| display_name_vi | VARCHAR(200) | True |  |
| label_version | VARCHAR(100) | False |  |
| is_active | BOOLEAN | False |  |
| id | UUID | False | PK |
| created_at | DATETIME | False |  |

## model_versions

| Column | Type | Nullable | Keys |
|---|---|---|---|
| model_name | VARCHAR(80) | False |  |
| run_id | VARCHAR(100) | False |  |
| dataset_version | VARCHAR(100) | False |  |
| num_classes | INTEGER | False |  |
| artifact_dir | TEXT | False |  |
| metrics_json | JSONB | True |  |
| is_production | BOOLEAN | False |  |
| id | UUID | False | PK |
| created_at | DATETIME | False |  |

## nutrition_foods

| Column | Type | Nullable | Keys |
|---|---|---|---|
| name | VARCHAR(200) | False |  |
| kcal_per_100g | NUMERIC(10, 2) | True |  |
| protein_per_100g | NUMERIC(10, 2) | True |  |
| carbohydrate_per_100g | NUMERIC(10, 2) | True |  |
| fat_per_100g | NUMERIC(10, 2) | True |  |
| source_name | VARCHAR(200) | False |  |
| source_record_id | VARCHAR(120) | True |  |
| provenance_url | TEXT | True |  |
| source_version | VARCHAR(100) | False |  |
| unit | VARCHAR(20) | False |  |
| retrieved_at | DATETIME | True |  |
| small_grams | NUMERIC(10, 2) | True |  |
| medium_grams | NUMERIC(10, 2) | True |  |
| large_grams | NUMERIC(10, 2) | True |  |
| mapping_note | TEXT | True |  |
| reviewed | BOOLEAN | False |  |
| updated_at | DATETIME | False |  |
| id | UUID | False | PK |
| created_at | DATETIME | False |  |

## users

| Column | Type | Nullable | Keys |
|---|---|---|---|
| display_name | VARCHAR(120) | False |  |
| email | VARCHAR(255) | True |  |
| id | UUID | False | PK |
| created_at | DATETIME | False |  |

## food_nutrition_mappings

| Column | Type | Nullable | Keys |
|---|---|---|---|
| food_class_id | UUID | False | FK → food_classes.id |
| nutrition_food_id | UUID | False | FK → nutrition_foods.id |
| is_default | BOOLEAN | False |  |
| confidence_note | TEXT | True |  |
| id | UUID | False | PK |
| created_at | DATETIME | False |  |

## meal_logs

| Column | Type | Nullable | Keys |
|---|---|---|---|
| user_id | UUID | False | FK → users.id |
| client_request_id | UUID | False |  |
| request_hash | VARCHAR(64) | False |  |
| consumed_on | DATE | False |  |
| meal_type | VARCHAR(20) | False |  |
| notes | TEXT | True |  |
| updated_at | DATETIME | False |  |
| id | UUID | False | PK |
| created_at | DATETIME | False |  |

## prediction_history

| Column | Type | Nullable | Keys |
|---|---|---|---|
| user_id | UUID | False | FK → users.id |
| model_version_id | UUID | True | FK → model_versions.id |
| image_path | TEXT | True |  |
| image_sha256 | VARCHAR(64) | True |  |
| crop_box_json | JSONB | True |  |
| response_json | JSONB | False |  |
| is_mock | BOOLEAN | False |  |
| id | UUID | False | PK |
| created_at | DATETIME | False |  |

## meal_items

| Column | Type | Nullable | Keys |
|---|---|---|---|
| meal_log_id | UUID | False | FK → meal_logs.id |
| confirmed_food_class_id | UUID | True | FK → food_classes.id |
| custom_food_name | VARCHAR(200) | True |  |
| prediction_id | UUID | True | FK → prediction_history.id |
| portion_size | VARCHAR(20) | False |  |
| estimated_grams | NUMERIC(10, 2) | True |  |
| kcal_per_100g_snapshot | NUMERIC(10, 2) | True |  |
| estimated_kcal | NUMERIC(10, 2) | True |  |
| protein_g | NUMERIC(10, 2) | True |  |
| carbohydrate_g | NUMERIC(10, 2) | True |  |
| fat_g | NUMERIC(10, 2) | True |  |
| nutrition_source_snapshot | TEXT | True |  |
| nutrition_snapshot | JSONB | True |  |
| calculation_method | VARCHAR(30) | False |  |
| is_user_corrected | BOOLEAN | False |  |
| notes | TEXT | True |  |
| updated_at | DATETIME | False |  |
| id | UUID | False | PK |
| created_at | DATETIME | False |  |
