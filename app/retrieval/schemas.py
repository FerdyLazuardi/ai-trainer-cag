"""
Shared schema for retrieval results across all retriever types.
"""
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """Unified result schema returned by any retriever."""
    chunk_id: str
    text: str
    score: float                     # fused relevance score — relative-score fusion of dense + sparse BM25
    hybrid_score: float = 0.0        # same fused score (kept for downstream/metric compatibility)
    dense_score: float = 0.0         # raw dense cosine [0, 1] — absolute signal used for the NOT-FOUND gate
    sparse_score: float = 0.0        # raw BM25 score — 0.0 = no lexical match in KB; >0 = exact term hit
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
            "hybrid_score": round(self.hybrid_score, 4),
            "dense_score": round(self.dense_score, 4),
            "sparse_score": round(self.sparse_score, 4),
        }


@dataclass
class HybridSearchResult:
    """Return envelope for `hybrid_search`.

    Carries the final fused chunks PLUS pool-level retrieval signals that the
    NOT-FOUND gate needs but that the truncated `chunks` list cannot provide.

    C4 — `pool_max_dense` / `pool_max_sparse` are the MAX raw scores over the
    full fetch_k candidate pool (per modality), computed BEFORE the top-k slice.
    The gate must see these, not the per-chunk maxes of the returned top-k: a
    chunk with the highest raw dense cosine can rank below the top-k by *fused*
    score (fusion blends in normalized sparse) and get sliced off, which would
    make the gate read an artificially low max and emit a false NOT-FOUND.

    C5 — `dense_available` is False when the dense embedding could not be
    produced (embedding provider down) and retrieval degraded to sparse-only
    BM25. The gate uses this to avoid treating a missing dense signal as a
    semantic miss (it would gate purely on sparse in that window).
    """
    chunks: list[RetrievedChunk] = field(default_factory=list)
    pool_max_dense: float = 0.0
    pool_max_sparse: float = 0.0
    dense_available: bool = True
    sparse_available: bool = True
