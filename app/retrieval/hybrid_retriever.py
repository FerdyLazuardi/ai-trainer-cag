"""
Hybrid retriever using LlamaIndex with Qdrant Vector Store.
Merges dense vector results and Sparse BM25 results natively via LlamaIndex configuration.
"""
from loguru import logger

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore

from app.config.settings import get_settings
from app.config.embedding_config import ensure_llamaindex_configured
from app.database.qdrant_client import get_qdrant_client
from app.retrieval.schemas import RetrievedChunk

settings = get_settings()

_index_cache: dict[str, VectorStoreIndex] = {}

def _get_cached_index(collection_name: str) -> VectorStoreIndex:
    """Singleton VectorStoreIndex per collection to avoid regenerating on every call."""
    if collection_name not in _index_cache:
        qdrant = get_qdrant_client()
        vector_store = qdrant.get_vector_store(collection_name, enable_hybrid=True)
        _index_cache[collection_name] = VectorStoreIndex.from_vector_store(vector_store)
    return _index_cache[collection_name]


async def hybrid_search(
    query: str,
    top_k: int | None = None,
    collection: str | None = None,      # Optional: override collection
) -> list[RetrievedChunk]:
    """
    Perform native hybrid (dense + sparse) retrieval utilizing LlamaIndex via Qdrant.

    Args:
        query: User query.
        top_k: Number of final results after fusion.
        collection: Qdrant collection name. Defaults to settings.qdrant_kb_collection.

    Returns:
        Fused list of RetrievedChunk natively via LlamaIndex.
    """
    ensure_llamaindex_configured()
    k = top_k or settings.retrieval_top_k

    # Use the specified collection or the Knowledge_Base default
    collection_name = collection or settings.qdrant_kb_collection

    index = _get_cached_index(collection_name)

    # Use Retriever
    retriever = index.as_retriever(
        similarity_top_k=k,
    )

    # 4. Async Retrieve
    nodes = await retriever.aretrieve(query)

    # Metadata fields surfaced from frontmatter in KB documents
    FRONTMATTER_FIELDS = {"department", "topic", "course_id", "course_name"}

    chunks: list[RetrievedChunk] = []
    for node_score in nodes:
        node = node_score.node
        payload = node.metadata or {}
        # Build rich metadata dict: include frontmatter + any other non-standard fields
        extra_meta = {k: v for k, v in payload.items() if k not in
                      ("document_id", "source", "title", "chunk_index", "token_count")}
        raw_score = float(node_score.score) if node_score.score else 0.0
        chunks.append(
            RetrievedChunk(
                chunk_id=node.node_id,
                text=node.text,
                score=raw_score,
                hybrid_score=raw_score,   # preserve native LlamaIndex hybrid score (dense+BM25)
                document_id=payload.get("document_id", ""),
                source=payload.get("source", ""),
                title=payload.get("title", ""),
                chunk_index=int(payload.get("chunk_index", 0) or 0),
                token_count=int(payload.get("token_count", 0) or 0),
                metadata=extra_meta,
            )
        )

    logger.debug(
        "LlamaIndex Hybrid retrieval complete",
        results=len(chunks),
        query=query[:60],
        collection=collection_name,
    )
    return chunks
