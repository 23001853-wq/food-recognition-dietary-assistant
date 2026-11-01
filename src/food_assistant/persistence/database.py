"""PostgreSQL engines; one SQLAlchemy Session per request/transaction."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def make_engine(url: str):
    if not url.startswith("postgresql+psycopg://"):
        raise ValueError("PostgreSQL with psycopg is required")
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})


def session_factory(engine):
    return sessionmaker(engine, expire_on_commit=False)
