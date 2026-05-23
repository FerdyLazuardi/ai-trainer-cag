"""
Pydantic v2 request and response models for the API layer.
"""
from pydantic import BaseModel, Field


# ─── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    conversation_id: str | None = Field(
        default=None,
        description="Optional session ID for multi-turn conversation memory",
    )
    course_id: int | None = Field(
        default=None,
        description="Optional course ID to scope the semantic cache and retrieval",
    )

    model_config = {"json_schema_extra": {"example": {"query": "What is LangGraph?", "course_id": 4}}}


class SourceReference(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    title: str
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    conversation_id: str | None = None
    resolved_query: str | None = None
    cached: bool = False
    latency_ms: float | None = None


# ─── Ingestion ────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw document text to ingest")
    title: str = Field(default="", description="Document title")
    source: str = Field(default="", description="Source URL or file path")
    metadata: dict = Field(default_factory=dict, description="Arbitrary key-value metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "LangGraph is a library for building stateful, multi-actor LLM applications.",
                "title": "LangGraph Overview",
                "source": "https://langchain.com/langgraph",
                "metadata": {"category": "tech", "language": "en"},
            }
        }
    }


class IngestResponse(BaseModel):
    document_id: str
    chunks_count: int
    total_tokens: int
    message: str = "Document ingested successfully"


# ─── Askfer (Portfolio Chat) ──────────────────────────────────────────────────

class AskferRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Visitor question")

    model_config = {
        "json_schema_extra": {
            "example": {"query": "What did you build in the Agent Network project?"}
        }
    }


class AskferSyncRequest(BaseModel):
    force_reingest: bool = Field(
        default=False,
        description="Re-process all portfolio docs even if content_hash matches.",
    )
