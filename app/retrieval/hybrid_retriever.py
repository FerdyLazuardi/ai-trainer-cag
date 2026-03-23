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


async def hybrid_search(
    query: str,
    top_k: int | None = None,
    bm25_weight: float | None = None,  # Kept for signature compatibility
    vector_weight: float | None = None, # Kept for signature compatibility
    collection: str | None = None,      # Optional: override collection
) -> list[RetrievedChunk]:
    """
    Perform native hybrid (dense + sparse) retrieval utilizing LlamaIndex via Qdrant.

    Args:
        query: User query.
        top_k: Number of final results after fusion.
        bm25_weight: Ignored
        vector_weight: Ignored
        collection: Qdrant collection name. Defaults to settings.qdrant_kb_collection.

    Returns:
        Fused list of RetrievedChunk natively via LlamaIndex.
    """
    ensure_llamaindex_configured()
    k = top_k or settings.retrieval_top_k
    qdrant = get_qdrant_client()

    # Use the specified collection or the Knowledge_Base default
    collection_name = collection or settings.qdrant_kb_collection

    # 1. Initialize LlamaIndex Qdrant Vector Store
    vector_store = QdrantVectorStore(
        aclient=qdrant.client,
        collection_name=collection_name,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        dense_vector_name="text-dense",
        sparse_vector_name="text-sparse",
    )

    # 2. Setup VectorStoreIndex
    index = VectorStoreIndex.from_vector_store(vector_store)

    # 3. Use Retriever
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
        chunks.append(
            RetrievedChunk(
                chunk_id=node.node_id,
                text=node.text,
                score=float(node_score.score) if node_score.score else 0.0,
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
