"""
Semantic reranker using Cohere and Jaccard deduplication.
"""
import cohere
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


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
    deduplicate: bool = True,
) -> list[RetrievedChunk]:
    """
    Rerank a list of retrieved chunks using Cohere's semantic reranker.

    Steps:
        1. Sort by descending score (already done by RRF, but re-sort for safety).
        2. Remove near-duplicate chunks (Jaccard > threshold).
        3. If Cohere API key is set, semantically rerank the deduplicated chunks.
        4. Return top-K.

    Args:
        query: The user's query used for semantic reranking.
        chunks: Retrieved chunks from hybrid search.
        top_k: Max results to return.
        deduplicate: Whether to filter near-duplicate chunks.

    Returns:
        Reranked list of RetrievedChunk.
    """
    k = top_k or settings.reranked_top_k

    if not chunks:
        return []

    # Sort descending by score
    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

    # Greedy deduplication: keep chunk if not too similar to any already-kept chunk
    if deduplicate:
        kept: list[RetrievedChunk] = []
        for candidate in sorted_chunks:
            is_duplicate = any(
                _jaccard_similarity(candidate.text, kept_chunk.text) > _SIMILARITY_THRESHOLD
                for kept_chunk in kept
            )
            if not is_duplicate:
                kept.append(candidate)
        sorted_chunks = kept

    if not settings.cohere_api_key:
        logger.debug(
            "Cohere API key not set, skipping semantic reranking",
            input_chunks=len(chunks),
            output_chunks=len(sorted_chunks[:k]),
            top_k=k,
        )
        return sorted_chunks[:k]

    try:
        co = cohere.AsyncClient(settings.cohere_api_key)
        texts = [c.text for c in sorted_chunks]
        
        response = await co.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=texts,
            top_n=k,
        )
        
        reranked_chunks = []
        for result in response.results:
            chunk = sorted_chunks[result.index]
            # Replace the original retrieval score with Cohere's relevance score
            chunk.score = result.relevance_score
            reranked_chunks.append(chunk)
            
        logger.debug(
            "Cohere Reranking complete",
            input_chunks=len(chunks),
            output_chunks=len(reranked_chunks),
            top_k=k,
        )
        return reranked_chunks
    except Exception as e:
        logger.warning(f"Cohere reranking failed, falling back to original ordering: {e}")
        return sorted_chunks[:k]
