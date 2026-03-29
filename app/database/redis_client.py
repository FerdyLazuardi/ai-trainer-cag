"""
Redis async connection pool and client singleton.
Uses redis.asyncio with connection pooling for performance.
"""
from functools import lru_cache

import redis.asyncio as aioredis

from app.config.settings import get_settings

settings = get_settings()

_pool: aioredis.ConnectionPool | None = None


def _create_pool() -> aioredis.ConnectionPool:
    return aioredis.ConnectionPool.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
    )


@lru_cache(maxsize=1)
def get_redis_client() -> aioredis.Redis:
    """Return the singleton Redis client (cached)."""
    global _pool
    _pool = _create_pool()
    return aioredis.Redis(connection_pool=_pool)
