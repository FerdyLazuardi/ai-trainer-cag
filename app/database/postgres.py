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
            ("retrieved_context", "JSONB"),
            # Cache observability columns (Jun 2026 — written by
            # _log_cache_event in app/utils/cache.py). Populated for
            # `endpoint='cache_lookup'` rows only; NULL for chat turns.
            # cache_score: 1.0 exact, qdrant cosine for semantic, NULL miss.
            # cache_namespace: 'rag' (Ava) / 'rag_user_<id>' / 'portfolio'.
            # query_hash: sha256(query.strip().lower())[:16], same scheme as
            #   the Redis cache key (cache.py:_cache_key) so dashboard
            #   joins to live cache state don't re-hash user text.
            ("cache_score", "DOUBLE PRECISION"),
            ("cache_namespace", "VARCHAR(64)"),
            ("query_hash", "VARCHAR(64)"),
            # Semantic-gate trace columns (Jun 2026 — written by chat.py
            # via _quality_log_fields for every turn that ran the gate).
            # gate_decision: "HIT" | "MISS" | "SKIP" — what the gate did.
            # gate_intent: the centroid that won (HIT only); None otherwise.
            # gate_best_cosine / gate_second_cosine / gate_margin: raw
            # numbers behind the decision, indexed for histogram queries
            # in the Streamlit dashboard. Migrated 2026-06-17 after a
            # production error showed SQLAlchemy INSERT raised
            # UndefinedColumnError on a fresh schema.
            ("gate_decision", "VARCHAR(8)"),
            ("gate_intent", "VARCHAR(32)"),
            ("gate_best_cosine", "DOUBLE PRECISION"),
            ("gate_second_cosine", "DOUBLE PRECISION"),
            ("gate_margin", "DOUBLE PRECISION"),
        ]
        for col, col_type in agent_log_columns:
            await conn.execute(
                text(f"ALTER TABLE agent_logs ADD COLUMN IF NOT EXISTS {col} {col_type}")
            )
        # Index turn_id for the async eval UPDATE lookup, intent for analytics,
        # and query_hash so the Streamlit cache-event drilldown can look up
        # "all cache events for query X" without a full table scan.
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_agent_logs_turn_id ON agent_logs (turn_id)")
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_agent_logs_intent ON agent_logs (intent)")
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_agent_logs_query_hash ON agent_logs (query_hash)")
        )
        # gate_decision + gate_margin are the dashboard's primary filters
        # (HIT vs MISS counts, margin histogram for threshold tuning).
        # gate_intent supports per-intent break-downs.
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_agent_logs_gate_decision ON agent_logs (gate_decision)")
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_agent_logs_gate_margin ON agent_logs (gate_margin)")
        )
        # Composite (endpoint, created_at DESC) for admin dashboard queries
        # that filter by endpoint (cache_lookup exclusion, askfer exclusion)
        # AND sort/limit by created_at. Single-column created_at index
        # already exists; this one covers the combined predicate without a
        # sort step.
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_agent_logs_endpoint_created_at "
                 "ON agent_logs (endpoint, created_at DESC)")
        )

        # documents.source: the ingest dedup path (moodle_sync / portfolio_sync)
        # looks documents up by `source` on every synced file. create_all won't
        # add an index to a pre-existing table, so add it idempotently here.
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_documents_source ON documents (source)")
        )
        # Stale-doc cleanup filters documents by JSON metadata keys
        # (metadata->>'course_id' in moodle_sync, metadata->>'doc_type' in
        # portfolio_sync). B-tree expression indexes on the extracted text turn
        # those seq-scans into index lookups. `->>` works on the json column as
        # is, so no JSONB migration / table rewrite is needed for these
        # equality filters (GIN would only help containment/key-existence).
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_documents_meta_course_id "
                 "ON documents ((metadata->>'course_id'))")
        )
        await conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_documents_meta_doc_type "
                 "ON documents ((metadata->>'doc_type'))")
        )

        # user_profiles predates the onboarding-tour flag. create_all won't ALTER
        # an existing table, so add the column idempotently here (NULL = tour
        # never seen). Drives the DB-backed first-run tour gate.
        await conn.execute(
            text("ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS "
                 "onboarding_completed_at TIMESTAMPTZ")
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
