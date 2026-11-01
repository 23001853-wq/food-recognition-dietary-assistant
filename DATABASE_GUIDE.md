# PostgreSQL and migrations

PostgreSQL is required; SQLite is not substituted. Current schema revision:
`cba28587a601`. Tables preserve the original PLAN vocabulary:

| Requested concept | Implemented structure |
|---|---|
| users | users |
| food_classes | food_classes (UUID, dataset_key, label_version) |
| nutrition_facts | nutrition_foods with nullable kcal and macros per 100g |
| portion_profiles | nullable small/medium/large_grams on nutrition_foods |
| model_registry | model_versions; runtime release pointer on filesystem |
| prediction_history | prediction_history, is_mock and full versioned response |
| meal_logs / meal_items | meal_logs / meal_items with snapshot provenance |

UUID primary/foreign keys, numeric constraints, UTC timestamps, default-profile partial
unique index and single-production partial unique index are included. IDs in model
labels are contiguous integers independent of database UUIDs. One request transaction
commits before success; errors roll back. Meal creation advisory locks serialize retries.

```powershell
docker compose up -d postgres
.venv/Scripts/python.exe -m alembic upgrade head
.venv/Scripts/python.exe scripts/seed_database.py
.venv/Scripts/python.exe -m alembic check
```

Seed is idempotent: demo user, ten POC food names, reviewed nutrition CSV/mapping.
No dataset image bytes or model weights are stored in DB. Never run downgrade/reset
against a database containing user data. Back up before schema changes.

Test DATABASE_URL must be separate and its name must end in `_test`. Tests create
that database if absent, apply migrations, and isolate writes using outer rollback and
savepoints. They do not drop/recreate the application database.

Windows native alternative: install PostgreSQL, create a role and database using your
chosen credentials, place a psycopg URL in .env, and run the same migration/seed commands.
This path is supported by code but has not been verified with a native Windows server.
