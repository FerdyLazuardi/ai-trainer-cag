"""
Consolidated LlamaIndex embedding & splitter configuration.

Call `ensure_llamaindex_configured()` once before any LlamaIndex operation.
This replaces the duplicate init functions that were in hybrid_retriever.py,
ingestion/pipeline.py, and ingestion/moodle_sync.py.
"""
from llama_index.core import Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from loguru import logger

from app.config.settings import get_settings

_initialized = False


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
        
        # Validate API key
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment variables")
        
        kwargs = {
            "model": settings.embedding_model,
            "api_key": settings.openrouter_api_key,
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
            Settings.embed_model = OpenAIEmbedding(**kwargs)
            logger.info("LlamaIndex embedding model initialized successfully")
            _initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    # ── Splitter (re-configurable per ingestion call) ───────────────────
    Settings.text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.debug("Text splitter configured", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
