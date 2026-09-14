"""
Database setup: engine, session factory, and the per-request session dependency.

SQLite is used deliberately. It is a single file, needs no server, and makes the
project runnable by anyone who clones the repo. Because everything goes through
SQLAlchemy's ORM, switching to PostgreSQL later means changing one line
(DATABASE_URL) rather than rewriting queries.
"""

from collections.abc import Generator

import logging

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)

# check_same_thread=False: FastAPI serves requests from a thread pool, and
# SQLite otherwise refuses connections created on a different thread.
connect_args = (
    {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)

# Pool sizing, and why it is not the default.
#
# FastAPI runs a sync dependency (`get_db`) and the endpoint body as two
# separate threadpool jobs. Under a burst of requests, every request's
# `get_db` can run first -- each one checking out a pool connection -- while
# the endpoint bodies that would *release* them wait for a thread. With the
# default pool (5 + 10 overflow) the 16th request then blocks inside `get_db`
# holding one of the two threads, the bodies never run, and after 30 s the
# pool times out with a 500. A priority inversion, seen under the load test.
#
# The connection count must therefore never be the scarce resource. uvicorn's
# --limit-concurrency (32) already bounds in-flight requests, so the overflow
# is set above that; a SQLite connection is a file handle and a small page
# cache, not something worth rationing. pool_timeout is short so that if
# something *else* ever exhausts it, the failure is fast and visible.
_pool_kwargs = {"pool_size": 5, "max_overflow": 40, "pool_timeout": 5}

engine = create_engine(
    settings.DATABASE_URL, connect_args=connect_args, echo=False, **_pool_kwargs
)

if settings.DATABASE_URL.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record) -> None:
        """Tune SQLite for one process with several threads.

        WAL (write-ahead log): readers no longer block on the writer and vice
        versa, so a chat request can read chunks while the worker thread is
        committing a section. The default rollback journal serialises them.

        busy_timeout: how long a writer waits for the single write lock before
        giving up with "database is locked". The default is five seconds; with
        a background worker committing every few seconds, a request that
        happens to collide should wait, not fail.

        Both are per-connection settings, hence the connect hook. Neither
        applies to Postgres, which handles all of this itself.
        """
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=15000")
        cursor.execute("PRAGMA synchronous=NORMAL")  # safe under WAL, faster commits
        cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Parent class for all ORM models."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: opens a session, guarantees it is closed.

    Endpoints commit explicitly. Keeping commits out of this helper makes
    transaction boundaries visible in the code that owns the business logic.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables if they don't exist. Called once on startup.

    A real production app would use Alembic migrations; for a single-developer
    project with a stable schema, create_all is the honest, simple choice.
    """
    from app import models  # noqa: F401  -- registers models on Base.metadata

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    """Forward-only, additive schema repair for an existing database.

    `create_all` creates tables that do not exist and never touches ones that
    do, so a column added to a model is silently absent from any database
    created before it -- and the first query that names it fails. This walks
    every mapped table, compares columns against the live schema, and issues
    `ALTER TABLE ... ADD COLUMN` for each one missing.

    It is the smallest thing that keeps a long-lived dev database (and a
    Postgres deployment, if one is configured) working across additive model
    changes. It does not rename, drop, or change types; that is what Alembic
    is for, and the README says so. New columns are added nullable or with
    the model's default, which is all the columns added so far have needed.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                ddl = f"ALTER TABLE {table.name} ADD COLUMN {column.name} "                       f"{column.type.compile(dialect=engine.dialect)}"
                conn.execute(text(ddl))
                logger.warning("Schema repair: added %s.%s", table.name, column.name)
