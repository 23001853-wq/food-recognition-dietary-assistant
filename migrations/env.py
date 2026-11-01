"""Migration entry point. Business schema is scheduled for Phase 4."""

from alembic import context

from food_assistant.config import Settings
from food_assistant.persistence import orm_models  # noqa: F401
from food_assistant.persistence.database import Base, make_engine

target_metadata = Base.metadata


def run_migrations() -> None:
    supplied_connection = context.config.attributes.get("connection")
    if supplied_connection is not None:
        context.configure(connection=supplied_connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
        return
    settings = Settings()
    if settings.database_url is None:
        raise RuntimeError("Set DATABASE_URL in .env before running migrations.")
    url = settings.database_url.get_secret_value()
    if context.is_offline_mode():
        context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
        with context.begin_transaction():
            context.run_migrations()
    else:
        engine = make_engine(url)
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()
        engine.dispose()


run_migrations()
