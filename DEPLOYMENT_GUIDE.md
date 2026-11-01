# Deployment — Windows 11 / Docker

Use the repository root containing pyproject.toml. Python 3.11 required.

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set POSTGRES_PASSWORD and the matching URL password in .env. Never overwrite an
existing .env or database as a setup shortcut. Use URL-safe credentials or percent-encode
special URL characters. For local mock development ALLOW_MOCK_AI=true; production
requires false and an actual validated release.

```powershell
docker compose up -d postgres
.venv/Scripts/python.exe -m alembic upgrade head
.venv/Scripts/python.exe scripts/seed_database.py
.venv/Scripts/python.exe -m uvicorn food_assistant.api.main:app --host 127.0.0.1 --port 8000
# Separate terminal:
.venv/Scripts/python.exe -m streamlit run src/food_assistant/app.py
```

UI: http://127.0.0.1:8501 — API docs: http://127.0.0.1:8000/docs.
If Windows resolves localhost slowly, use 127.0.0.1 in DATABASE_URL.

All-container option:

```powershell
docker compose up -d --build
docker compose ps
docker compose logs --tail 50 backend
docker compose stop
```

`stop` preserves the named DB volume. Do not use `down -v` with real data.
Do not run host servers and containers on the same ports simultaneously.
The default image supports mock/demo and sourced nutrition. It excludes raw datasets,
secrets, venv, uploads and model weights from build context. Artifacts/storage mount at
runtime. To run real inference inside Docker, install the compatible ML extra in a
separate ML-enabled image; default lightweight image does not include PyTorch.

PostgreSQL, backend and frontend have health checks, dependency ordering and loopback
host port bindings. Backend applies migrations and idempotent seed on startup.
Training remains outside Docker by default. No public deployment/authentication is supplied.

```powershell
.venv/Scripts/python.exe scripts/smoke_test.py --base-url http://127.0.0.1:8000 --model mock
```

This creates and deletes only its own uniquely identified smoke meal, leaving a
prediction-history record as provenance. Existing user meal data is untouched.
