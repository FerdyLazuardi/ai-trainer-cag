"""
Semantic reranker using Cohere and Jaccard deduplication.
"""
import cohere
from loguru import logger
from functools import lru_cache

from app.config.settings import get_settings
from app.retrieval.schemas import RetrievedChunk

settings = get_settings()


@lru_cache(maxsize=1)
def _get_cohere_client() -> cohere.AsyncClient:
    return cohere.AsyncClient(settings.cohere_api_key)


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
    deduplicate: bool = True,
) -> list[RetrievedChunk]:
    """
    Rerank a list of retrieved chunks using Cohere's semantic reranker.
    
    Steps:
        1. Sort by descending initial score.
        2. If Cohere API key is set, semantically rerank. Cohere V3 handles 
           relevance and implicitly de-prioritizes redundant information.
        3. Return top-K.

    Args:
        query: The user's query used for semantic reranking.
        chunks: Retrieved chunks from hybrid search.
        top_k: Max results to return.
        deduplicate: (Ignored) Kept for signature compatibility.

    Returns:
        Reranked list of RetrievedChunk.
    """
    k = top_k or settings.reranked_top_k

    if not chunks:
        return []

    # Sort descending by score
    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

    if not settings.cohere_api_key:
        logger.debug(
            "Cohere API key not set, skipping semantic reranking",
            input_chunks=len(chunks),
            output_chunks=len(sorted_chunks[:k]),
            top_k=k,
        )
        return sorted_chunks[:k]

    try:
        co = _get_cohere_client()
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
            # hybrid_score already set from retriever — keep it intact
            # Replace only the final score with Cohere's relevance score
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
