"""
Score-based reranker.
Normalizes and re-ranks chunks by their RRF score, applying diversity filtering
to avoid returning near-duplicate chunks.
"""
from loguru import logger

from app.config.settings import get_settings
from app.retrieval.schemas import RetrievedChunk

settings = get_settings()

_SIMILARITY_THRESHOLD = 0.85  # Jaccard similarity above this = near-duplicate


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Approximate deduplication via token-level Jaccard similarity."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def rerank(
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
    deduplicate: bool = True,
) -> list[RetrievedChunk]:
    """
    Rerank a list of retrieved chunks.

    Steps:
        1. Sort by descending score (already done by RRF, but re-sort for safety).
        2. Remove near-duplicate chunks (Jaccard > threshold).
        3. Return top-K.

    Args:
        chunks: Retrieved chunks from hybrid search.
        top_k: Max results to return.
        deduplicate: Whether to filter near-duplicate chunks.

    Returns:
        Reranked list of RetrievedChunk.
    """
    k = top_k or settings.reranked_top_k

    # Sort descending by score
    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

    if not deduplicate:
        return sorted_chunks[:k]

    # Greedy deduplication: keep chunk if not too similar to any already-kept chunk
    kept: list[RetrievedChunk] = []
    for candidate in sorted_chunks:
        is_duplicate = any(
            _jaccard_similarity(candidate.text, kept_chunk.text) > _SIMILARITY_THRESHOLD
            for kept_chunk in kept
        )
        if not is_duplicate:
            kept.append(candidate)
        if len(kept) >= k:
            break

    logger.debug(
        "Reranking complete",
        input_chunks=len(chunks),
        output_chunks=len(kept),
        top_k=k,
    )
    return kept
