"""
Embedding generator.
Uses langchain-openai to connect to the configured embedding URL (e.g. OpenRouter/OpenAI).
"""
import asyncio
from functools import lru_cache

from fastembed import SparseTextEmbedding
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from app.config.settings import get_settings

settings = get_settings()

@lru_cache(maxsize=1)
def _get_sparse_model() -> SparseTextEmbedding:
    """Load and cache the FastEmbed sparse model (BM25)."""
    logger.info("Initializing fastembed sparse client", model="Qdrant/bm25")
    return SparseTextEmbedding("Qdrant/bm25", threads=1)



@lru_cache(maxsize=1)
def _get_model() -> OpenAIEmbeddings:
    """Load and cache the embeddings client (called once)."""
    logger.info("Initializing embedding client", model=settings.embedding_model)

    kwargs = {
        "model": settings.embedding_model,
        "openai_api_key": settings.openrouter_api_key,
    }
    
    if settings.openrouter_embedding_url:
        base = settings.openrouter_embedding_url
        if base.endswith("/embeddings"):
            base = base[:-11]
        kwargs["openai_api_base"] = base
        
    # the text-embedding-3 models support dimension shrinking natively via the API
    if "text-embedding-3" in settings.embedding_model and settings.embedding_dim:
        kwargs["dimensions"] = settings.embedding_dim

    return OpenAIEmbeddings(**kwargs)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts asynchronously.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (list of float).
    """
    if not texts:
        return []

    model = _get_model()
    embeddings = await model.aembed_documents(texts)
    
    logger.debug("Embeddings generated", count=len(texts), dim=len(embeddings[0]) if embeddings else 0)
    return embeddings


async def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    Args:
        query: The query to embed.

    Returns:
        Embedding vector.
    """
    model = _get_model()
    embedding = await model.aembed_query(query)
    return embedding


async def embed_sparse_texts(texts: list[str]) -> list[dict]:
    """
    Generate sparse embeddings for a list of texts asynchronously using FastEmbed.
    Runs in a threadpool to prevent blocking the async event loop.
    """
    if not texts:
        return []

    model = _get_sparse_model()
    loop = asyncio.get_running_loop()
    
    def _run():
        return list(model.embed(texts, parallel=0))
        
    embeddings = await loop.run_in_executor(None, _run)
    # Convert FastEmbed SparseEmbedding objects to dictionaries Qdrant can use natively
    return [{"indices": e.indices.tolist(), "values": e.values.tolist()} for e in embeddings]


async def embed_sparse_query(query: str) -> dict:
    """
    Embed a single sparse query string using FastEmbed.
    """
    embeddings = await embed_sparse_texts([query])
    return embeddings[0]
