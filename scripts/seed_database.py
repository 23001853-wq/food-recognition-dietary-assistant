"""Seed demo user and food catalog; idempotent and no fabricated nutrition."""

from food_assistant.config import Settings
from food_assistant.persistence.database import make_engine, session_factory
from food_assistant.persistence.seed import seed

if __name__ == "__main__":
    config = Settings()
    if config.database_url is None:
        raise SystemExit("DATABASE_URL required")
    engine = make_engine(config.database_url.get_secret_value())
    with session_factory(engine).begin() as session:
        summary = seed(session)
    engine.dispose()
    print("Seed complete (idempotent); reviewed nutrition import:", summary)
