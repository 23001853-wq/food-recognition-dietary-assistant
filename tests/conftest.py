import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from food_assistant.api.main import create_app
from food_assistant.config import Settings
from food_assistant.persistence.seed import seed


@pytest.fixture(scope="session")
def pg_engine():
    url = os.environ.get("TEST_DATABASE_URL") or dotenv_values(".env").get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not configured; PostgreSQL tests not run")
    parsed = make_url(url)
    if not parsed.database.endswith("_test"):
        raise RuntimeError("Refusing to test against a database not ending in _test")
    admin = create_engine(
        parsed.set(database="postgres"),
        isolation_level="AUTOCOMMIT",
        connect_args={"connect_timeout": 5},
    )
    with admin.connect() as connection:
        if not connection.scalar(
            text("SELECT 1 FROM pg_database WHERE datname=:name"), {"name": parsed.database}
        ):
            connection.exec_driver_sql(
                'CREATE DATABASE "' + parsed.database.replace('"', '""') + '"'
            )
    admin.dispose()
    engine = create_engine(url, connect_args={"connect_timeout": 5})
    config = Config(str(Path("alembic.ini").resolve()))
    with engine.connect() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    yield engine
    engine.dispose()


@pytest.fixture
def client(pg_engine):
    from fastapi.testclient import TestClient

    with pg_engine.connect() as connection:
        transaction = connection.begin()
        factory = sessionmaker(
            bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
        )
        with factory.begin() as session:
            seed(session)
        settings = Settings(_env_file=None, allow_mock_ai=True, default_model="mock")
        app = create_app(settings, engine=pg_engine)
        app.state.session_factory = factory
        with TestClient(app) as test_client:
            yield test_client
        transaction.rollback()


@pytest.fixture
def image_bytes():
    from io import BytesIO

    from PIL import Image

    data = BytesIO()
    Image.new("RGB", (64, 64), "red").save(data, "PNG")
    return data.getvalue()
