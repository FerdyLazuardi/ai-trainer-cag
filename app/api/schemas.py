"""
Pydantic v2 request and response models for the API layer.
"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    conversation_id: str | None = Field(
        default=None,
        description="Optional session ID for multi-turn conversation memory",
    )
    course_id: int | None = Field(
        default=None,
        description="Optional course ID to scope the exact answer cache",
    )
    coaching_mode: bool = Field(
        default=False,
        description=(
            "When true, Ava enters Socratic coaching mode: for diagnostic / "
            "reasoning questions ('kenapa', 'gimana caranya') she opens with ONE "
            "grounded guiding question to make the user think, keeps each turn "
            "light, then confirms + teaches in full at the wrap-up. Pure factual "
            "lookups are still answered directly. User opts in via a UI toggle; "
            "default false = normal direct answers (Ava still acts as an L&D "
            "trainer by default, just without the Socratic guiding questions)."
        ),
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
    ban_remaining_seconds: int | None = None
