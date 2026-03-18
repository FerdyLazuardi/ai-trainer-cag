"""
Shared schema for retrieval results across all retriever types.
"""
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """Unified result schema returned by any retriever."""
    chunk_id: str
    text: str
    score: float                     # normalized relevance score [0, 1]
    document_id: str = ""
    source: str = ""
    title: str = ""
    chunk_index: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def to_source_dict(self) -> dict:
        """Return a clean source reference for API responses."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
        }
