# Migrations

Business schema revision cba28587a601 is implemented and has run against PostgreSQL. Use alembic upgrade head and alembic check. No automatic downgrade or database reset. Tests migrate only the configured test database. See DATABASE_GUIDE.md for constraints, naming equivalence and snapshot policy.
