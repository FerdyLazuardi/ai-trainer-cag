"""
SQLAlchemy ORM models for document management and agent observability.
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
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
    source: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
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
    qdrant_point_id: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
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
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=True, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[str] = mapped_column(Text, nullable=True)
    answer: Mapped[str] = mapped_column(Text, nullable=True)
    chunks_retrieved: Mapped[int] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=True)
    llm_tokens_used: Mapped[int] = mapped_column(Integer, nullable=True)
    cache_hit: Mapped[bool] = mapped_column(nullable=False, default=False)
    error: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    def __repr__(self) -> str:
        return f"<AgentLog id={self.id} query={self.query[:40]!r} latency={self.latency_ms}ms>"


class UserMemory(Base):
    """Persists user learning context across chat sessions."""

    __tablename__ = "user_memory"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    summary: Mapped[str] = mapped_column(Text, nullable=True)
    topics: Mapped[list] = mapped_column(JSON, nullable=True, default=list)
    last_active: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<UserMemory user_id={self.user_id} last_active={self.last_active}>"
