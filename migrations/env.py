"""Migration entry point. Business schema is scheduled for Phase 4."""

from alembic import context
from sqlalchemy import create_engine

from food_assistant.config import Settings
from food_assistant.persistence import orm_models  # noqa: F401
from food_assistant.persistence.database import Base

target_metadata = Base.metadata


def run_migrations() -> None:
    if not target_metadata.tables:
        raise RuntimeError("Business schema is not implemented yet; see PLAN.md section 12.")
    settings = Settings()
    if settings.database_url is None:
        raise RuntimeError("Set DATABASE_URL in .env before running migrations.")
    url = settings.database_url.get_secret_value()
    if context.is_offline_mode():
        context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
        with context.begin_transaction():
            context.run_migrations()
    else:
        engine = create_engine(url)
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()
        engine.dispose()


run_migrations()
