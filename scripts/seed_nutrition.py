"""Append reviewed nutrition only; no migrations, model changes or meal rewrites."""

import json

from food_assistant.config import Settings
from food_assistant.persistence.database import make_engine, session_factory
from food_assistant.persistence.seed import seed_nutrition

if __name__ == "__main__":
    config = Settings()
    if config.database_url is None:
        raise SystemExit("DATABASE_URL required")
    engine = make_engine(config.database_url.get_secret_value())
    try:
        with session_factory(engine).begin() as session:
            summary = seed_nutrition(session)
        print(json.dumps(summary, indent=2))
    finally:
        engine.dispose()
