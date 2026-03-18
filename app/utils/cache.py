"""
Redis query cache decorator.
Caches RAG pipeline results by query hash to avoid repeated LLM calls.
"""
import hashlib
import json
from functools import wraps
from typing import Any, Callable

from loguru import logger

from app.config.settings import get_settings
from app.database.redis_client import get_redis_client

settings = get_settings()

_PREFIX = "rag:cache:"


def _cache_key(query: str) -> str:
    """Generate a deterministic Redis key from the query string."""
    query_hash = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
    return f"{_PREFIX}{query_hash}"


async def get_cached_response(query: str) -> dict | None:
    """
    Retrieve a cached RAG response for the given query.

    Returns:
        Cached dict with 'answer' and 'sources', or None on miss.
    """
    redis = get_redis_client()
    key = _cache_key(query)

    try:
        raw = await redis.get(key)
        if raw:
            data = json.loads(raw)
            logger.info("Cache HIT", query=query[:60], key=key)
            return data
        logger.debug("Cache MISS", query=query[:60])
        return None
    except Exception as exc:
        logger.warning("Redis cache GET failed", error=str(exc))
        return None


async def set_cached_response(
    query: str,
    answer: str,
    sources: list[dict],
    ttl: int | None = None,
) -> None:
    """
    Store a RAG response in Redis cache.

    Args:
        query: The user query (used as cache key input).
        answer: The generated answer.
        sources: List of source references.
        ttl: Optional TTL override in seconds.
    """
    redis = get_redis_client()
    key = _cache_key(query)
    ttl_ = ttl or settings.cache_query_ttl_seconds

    try:
        data = json.dumps({"answer": answer, "sources": sources})
        await redis.set(key, data, ex=ttl_)
        logger.debug("Cache SET", query=query[:60], ttl=ttl_)
    except Exception as exc:
        logger.warning("Redis cache SET failed", error=str(exc))
