"""
SQLAlchemy async database engine and session management.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config.settings import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.postgres_dsn,
    echo=settings.app_debug,
    pool_size=settings.postgres_pool_size,
    max_overflow=settings.postgres_max_overflow,
    pool_timeout=settings.postgres_pool_timeout,  # fail fast instead of 30s default hang
    pool_pre_ping=True,       # healthcheck before using a connection
    pool_recycle=3600,        # recycle connections after 1h
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


async def init_db() -> None:
    """Create all tables on startup (non-destructive)."""
    from app.database import models  # noqa: F401 – registers models
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        # create_all only CREATEs missing tables — it never ALTERs an existing
        # one. agent_logs predates the quality-signal columns, so add them
        # idempotently here. `ADD COLUMN IF NOT EXISTS` is a no-op when present,
        # so this is safe to run on every startup (and avoids needing Alembic).
        agent_log_columns = [
            ("turn_id", "VARCHAR(64)"),
            ("endpoint", "VARCHAR(32)"),
            ("intent", "VARCHAR(32)"),
            ("needs_lookup", "DOUBLE PRECISION"),
            ("needs_reasoning", "DOUBLE PRECISION"),
            ("needs_empathy", "DOUBLE PRECISION"),
            ("max_dense_score", "DOUBLE PRECISION"),
            ("faithfulness_score", "DOUBLE PRECISION"),
        ]
        for col, col_type in agent_log_columns:
            await conn.execute(
                text(f"ALTER TABLE agent_logs ADD COLUMN IF NOT EXISTS {col} {col_type}")
            )
        # Index turn_id for the async eval UPDATE lookup, and intent for analytics.
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_agent_logs_turn_id ON agent_logs (turn_id)")
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_agent_logs_intent ON agent_logs (intent)")
        )


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager yielding a database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency injector for database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
