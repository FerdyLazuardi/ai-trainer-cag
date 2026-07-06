"""
SQLAlchemy ORM models for document management and agent observability.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.postgres import Base


class Document(Base):
    """Represents an ingested source document."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    source: Mapped[str] = mapped_column(String(1024), nullable=False, default="", index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    ingestion_state: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )  # pending | processing | completed | failed
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} source={self.source!r} state={self.ingestion_state}>"


class Chunk(Base):
    """Represents a single text chunk derived from a Document."""

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk id={self.id} doc={self.document_id} idx={self.chunk_index}>"


class AgentLog(Base):
    """Stores structured observability logs for each RAG pipeline invocation."""

    __tablename__ = "agent_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # Correlation key set by the chat handler at log time. The async eval task
    # uses it to UPDATE this row with the faithfulness score once the judge runs.
    turn_id: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    endpoint: Mapped[str] = mapped_column(String(32), nullable=True)
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[str] = mapped_column(Text, nullable=True)
    answer: Mapped[str] = mapped_column(Text, nullable=True)
    chunks_retrieved: Mapped[int] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=True)
    llm_tokens_used: Mapped[int] = mapped_column(Integer, nullable=True)
    cache_hit: Mapped[bool] = mapped_column(nullable=False, default=False)
    # Cache observability columns (Jun 2026 — wired by _log_cache_event in
    # app/utils/cache.py). Populated for `endpoint='cache_lookup'` rows only.
    # cache_score: 1.0 for Redis exact hit, Qdrant cosine for semantic hit,
    #   None for miss. The Streamlit dashboard bins p50/p95 across this.
    # cache_namespace: 'rag' (Ava), 'rag_user_<id>' (user-scoped),
    #   'portfolio' (Askfer). Lets ops see which persona drives hit-rate.
    # query_hash: sha256(query.strip().lower())[:16] — same scheme as the
    #   Redis cache key (cache.py:_cache_key), so rows can be joined to
    #   the live cache state without re-hashing the user text.
    cache_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    cache_namespace: Mapped[str | None] = mapped_column(String(64), nullable=True)
    query_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    # ── Quality signals (durable) ──
    intent: Mapped[str] = mapped_column(String(32), nullable=True, index=True)
    needs_lookup: Mapped[float] = mapped_column(Float, nullable=True)
    needs_reasoning: Mapped[float] = mapped_column(Float, nullable=True)
    needs_empathy: Mapped[float] = mapped_column(Float, nullable=True)
    max_dense_score: Mapped[float] = mapped_column(Float, nullable=True)
    # ── Semantic-gate trace (Jun 2026) ──
    # gate_decision: "HIT" (gate committed an intent), "MISS" (gate ran but
    #   stayed silent — caller fell through to KNOWLEDGE), or "SKIP" (gate
    #   wasn't run because regex Tier-1 already returned an intent).
    # gate_intent: the centroid that won (HIT only); None otherwise.
    # gate_best_cosine / gate_second_cosine / gate_margin: the raw numbers
    #   behind the decision, indexed together for histogram queries in the
    #   Streamlit dashboard. Stored as the centroid-relative signal so a
    #   future re-tune of centroids doesn't lose the trace.
    gate_decision: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    gate_intent: Mapped[str | None] = mapped_column(String(32), nullable=True)
    gate_best_cosine: Mapped[float | None] = mapped_column(Float, nullable=True)
    gate_second_cosine: Mapped[float | None] = mapped_column(Float, nullable=True)
    gate_margin: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    # Filled asynchronously by the LLM-as-judge eval task (sampled).
    faithfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    # OR cache observability (Jun 2026)
    or_prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    or_cached_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    or_completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    or_provider: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    or_generation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    or_cache_savings_usd: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    or_duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    or_cost: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieved_context: Mapped[list[dict]] = mapped_column(JSON, nullable=True)
    error: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    def __repr__(self) -> str:
        return f"<AgentLog id={self.id} query={self.query[:40]!r} latency={self.latency_ms}ms>"


class KnowledgeBasePack(Base):
    """Stores deterministic CAG knowledge-base packs."""

    __tablename__ = "kb_packs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    kb_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, index=True
    )

    def __repr__(self) -> str:
        return f"<KnowledgeBasePack id={self.id} source={self.source!r} active={self.is_active}>"


class UserProfile(Base):
    """Stores persistent user preferences derived from conversation history."""

    __tablename__ = "user_profiles"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    role: Mapped[str] = mapped_column(String(128), nullable=True)
    preferred_tone: Mapped[str] = mapped_column(String(64), nullable=True)
    formatting_pref: Mapped[str] = mapped_column(String(64), nullable=True)
    custom_instructions: Mapped[str] = mapped_column(Text, nullable=True)
    # First-run onboarding tour: set once when the user finishes (or skips) the
    # tour. NULL = never seen it. DB-backed (not localStorage) so the tour shows
    # exactly once per Moodle user, even across browsers/devices.
    onboarding_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<UserProfile user_id={self.user_id} role={self.role!r}>"




