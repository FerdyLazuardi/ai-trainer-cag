"""
Qdrant dense vector retriever.
Performs approximate nearest-neighbor search using COSINE similarity.
"""
from loguru import logger
from qdrant_client.models import Filter

from app.config.settings import get_settings
from app.database.qdrant_client import get_qdrant_client
from app.ingestion.embedder import embed_query
from app.retrieval.schemas import RetrievedChunk

settings = get_settings()


async def vector_search(
    query: str,
    top_k: int | None = None,
    score_threshold: float = 0.0,
    filters: Filter | None = None,
) -> list[RetrievedChunk]:
    """
    Perform dense vector search in Qdrant.

    Args:
        query: User query to embed and search.
        top_k: Number of results to retrieve.
        score_threshold: Minimum similarity score.
        filters: Optional Qdrant Filter for metadata filtering.

    Returns:
        List of RetrievedChunk sorted by descending similarity score.
    """
    k = top_k or settings.retrieval_top_k
    qdrant = get_qdrant_client()

    query_vector = await embed_query(query)

    results = await qdrant.client.query_points(
        collection_name=qdrant.collection,
        query=query_vector,
        using="dense",
        limit=k,
        score_threshold=score_threshold,
        query_filter=filters,
        with_payload=True,
        with_vectors=False,
    )
    
    # query_points returns a QueryResponse which has a .points attribute
    points = results.points

    chunks: list[RetrievedChunk] = []
    for hit in points:
        payload = hit.payload or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=str(hit.id),
                text=payload.get("text", ""),
                score=float(hit.score),
                document_id=payload.get("document_id", ""),
                source=payload.get("source", ""),
                title=payload.get("title", ""),
                chunk_index=payload.get("chunk_index", 0),
                token_count=payload.get("token_count", 0),
                metadata={k: v for k, v in payload.items() if k not in
                          ("text", "document_id", "source", "title", "chunk_index", "token_count")},
            )
        )

    logger.debug("Vector search complete", query=query[:60], results=len(chunks))
    return chunks
