"""Declarative base. Engine and transaction management will be added with repositories."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
