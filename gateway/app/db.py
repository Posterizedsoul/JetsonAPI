"""Postgres pool and a migration runner.

Migrations are numbered .sql files applied in filename order and recorded in
schema_migrations. No Alembic: the schema is small, the files are readable,
and "apply every file I haven't seen" is the whole feature.
"""

from pathlib import Path

from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

from app import config

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

pool = ConnectionPool(config.DATABASE_URL, min_size=1, max_size=8, open=False)


def migrate() -> list[str]:
    """Apply pending migrations. Returns the names applied."""
    applied: list[str] = []
    with pool.connection() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " name TEXT PRIMARY KEY,"
            " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        done = {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in done:
                continue
            # Each migration commits as one transaction; a failure leaves it
            # unrecorded so the next boot retries it.
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,)
            )
            conn.commit()
            applied.append(path.name)
    return applied


def query(sql: str, params: tuple = ()) -> list[dict]:
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def execute(sql: str, params: tuple = ()) -> None:
    with pool.connection() as conn:
        conn.execute(sql, params)
