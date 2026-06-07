"""
Consolidated LlamaIndex embedding & splitter configuration.

Call `ensure_llamaindex_configured()` once before any LlamaIndex operation.
This replaces the duplicate init functions that were in hybrid_retriever.py,
ingestion/pipeline.py, and ingestion/moodle_sync.py.
"""
from typing import Any

from llama_index.core import Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from loguru import logger

from app.config.settings import get_settings

_initialized = False
_last_chunk_config = (0, 0)


def ensure_llamaindex_configured(
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> None:
    """
    Initialize LlamaIndex global Settings (embedding model + text splitter).

    Safe to call multiple times — only the first call takes effect.
    If chunk_size/chunk_overlap differ from the current config, the splitter
    is re-initialized (embedding model stays the same).
    """
    global _initialized
    settings = get_settings()

    if not _initialized:
        # ── Embedding model (one-time) ──────────────────────────────────
        logger.info("Initializing LlamaIndex embedding model", model=settings.embedding_model)
        
        # Determine the correct API key for embeddings
        emb_api_key = settings.openrouter_embedding_key or settings.openrouter_api_key
        if not emb_api_key:
            raise ValueError("Neither OPENROUTER_EMBEDDING_KEY nor OPENROUTER_API_KEY is set in environment variables")
        
        kwargs: dict[str, Any] = {
            "model_name": settings.embedding_model,
            "api_key": emb_api_key,
        }
        
        # OpenRouter uses OpenAI-compatible API
        if settings.openrouter_embedding_url:
            base = settings.openrouter_embedding_url
            if base.endswith("/embeddings"):
                base = base[:-11]
            kwargs["api_base"] = base
            logger.info("Using custom embedding URL", url=base)
        else:
            # Default to OpenRouter base URL for embeddings
            kwargs["api_base"] = settings.openrouter_base_url
            logger.info("Using OpenRouter base URL for embeddings", url=settings.openrouter_base_url)

        # For OpenAI text-embedding-3 models, we can specify dimensions
        if "text-embedding-3" in settings.embedding_model and settings.embedding_dim:
            kwargs["dimensions"] = settings.embedding_dim
            logger.info("Setting embedding dimensions", dimensions=settings.embedding_dim)

        try:
            Settings.embed_model = OpenAILikeEmbedding(**kwargs)
            logger.info("LlamaIndex embedding model initialized successfully")
            _initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    # ── Splitter (re-configurable per ingestion call) ───────────────────
    global _last_chunk_config
    if _last_chunk_config != (chunk_size, chunk_overlap):
        Settings.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        logger.debug("Text splitter configured", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        _last_chunk_config = (chunk_size, chunk_overlap)
