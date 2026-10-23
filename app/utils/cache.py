"""
Query cache — Redis exact-match only.

Redis exact-match has no risk of cross-user/cross-tone leaks: the key is sha256 of the exact (canonical)
query string, so only a byte-identical re-ask hits, and a grounded turn is
deterministic (temp 0) so the re-served answer is the one that query would have
produced anyway.
"""
import hashlib
import json
from typing import Any

from loguru import logger

from app.config.settings import get_settings
from app.database.redis_client import get_redis_client
from app.utils.logger_batch import batch_logger

settings = get_settings()

_PREFIX = "rag:cache:"


def _query_hash(query: str) -> str:
    """sha256[:16] of the canonicalized query (`strip().lower()`)."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def _cache_key(query: str, course_id: int | None = None, namespace: str = "rag") -> str:
    """Generate a deterministic Redis key from the query string and course_id."""
    cid_str = str(course_id) if course_id and course_id > 0 else "global"
    return f"{namespace}:cache:{cid_str}:{_query_hash(query)}"


async def _log_cache_event(
    *,
    hit: bool,
    course_id: int | None,
    query: str,
    namespace: str,
    score: float | None = None,
) -> None:
    """Persist a cache_lookup HIT event to agent_logs via BatchLogger."""
    try:
        cid = course_id if course_id and course_id > 0 else None
        query_hash = _query_hash(query or "")
        await batch_logger.add_log({
            "endpoint": "cache_lookup",
            "query": query or "",
            "course_id": cid,
            "cache_hit": bool(hit),
            "cache_score": round(float(score), 4) if score is not None else None,
            "cache_namespace": namespace,
            "query_hash": query_hash,
        })
    except Exception as e:
        logger.debug("Failed to log cache event to agent_logs", error=str(e))


async def get_cached_response(
    query: str,
    course_id: int | None = None,
    cache_namespace: str = "rag",
) -> dict | None:
    """Retrieve a cached response via Redis exact string match."""
    if course_id == 0:
        course_id = None

    redis = get_redis_client()
    key = _cache_key(query, course_id, namespace=cache_namespace)
    try:
        raw = await redis.get(key)
        if raw:
            data = json.loads(raw)
            logger.info("Redis Exact Cache HIT", query=query[:60], course_id=course_id, namespace=cache_namespace)
            await _log_cache_event(
                hit=True, course_id=course_id, score=1.0,
                query=query, namespace=cache_namespace,
            )
            return data
    except Exception as exc:
        logger.warning("Redis cache GET failed", error=str(exc))

    logger.debug("Cache MISS", query=query[:60], course_id=course_id, namespace=cache_namespace)
    return None


async def set_cached_response(
    query: str,
    answer: str,
    sources: list[dict],
    course_id: int | None = None,
    ttl: int | None = None,
    cache_namespace: str = "rag",
) -> None:
    """Store a response in Redis with a TTL."""
    if len(answer) < 100:
        logger.debug("Skipping cache write: response too short", length=len(answer))
        return

    lower_answer = answer.lower()
    error_phrases = [
        "maaf saya tidak", "terjadi kesalahan",
        "maaf, aku tidak menemukan", "maaf, saya tidak menemukan",
        "belum nemu", "belum menemukan", "belum ketemu",
        "sejauh pengetahuanku belum ada", "kurang tahu pasti",
        "ga nemu", "gak nemu", "nggak nemu",
        "coba pastikan lagi pertanyaannya", "coba cari dengan kata kunci lain",
    ]
    if any(phrase in lower_answer for phrase in error_phrases):
        logger.debug("Skipping cache write: response contains error/not-found phrase")
        return

    if course_id == 0:
        course_id = None

    redis = get_redis_client()
    key = _cache_key(query, course_id, namespace=cache_namespace)
    ttl_ = ttl or settings.cache_query_ttl_seconds

    payload: dict[str, Any] = {"answer": answer, "sources": sources, "namespace": cache_namespace}
    if course_id:
        payload["course_id"] = int(course_id)

    try:
        await redis.set(key, json.dumps(payload), ex=ttl_)
    except Exception as exc:
        logger.warning("Redis cache SET failed", error=str(exc))


async def flush_cache_by_namespace(namespace: str) -> None:
    """Delete Redis cache keys for a specific namespace."""
    if not namespace:
        return

    redis = get_redis_client()
    try:
        cursor = 0
        prefix = f"{namespace}:cache:"
        while True:
            cursor, keys = await redis.scan(cursor, match=f"{prefix}*", count=5000)
            if keys:
                await redis.unlink(*keys)
            if cursor == 0:
                break
        logger.info("Redis namespace cache flushed", namespace=namespace)
    except Exception as exc:
        logger.warning("Redis namespace cache flush failed", error=str(exc))


async def flush_cache_by_course(course_id: int) -> None:
    """Delete Redis cache keys for a specific course_id — global + all user namespaces."""
    redis = get_redis_client()
    patterns = [
        f"{_PREFIX}{course_id}:*",
        f"rag_user_*:cache:{course_id}:*",
        f"{_PREFIX}global:*",
        "rag_user_*:cache:global:*",
    ]
    for match in patterns:
        try:
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=match, count=5000)
                if keys:
                    await redis.unlink(*keys)
                if cursor == 0:
                    break
            logger.info("Redis course cache flushed", course_id=course_id, pattern=match)
        except Exception as exc:
            logger.warning("Redis course cache flush failed", error=str(exc), pattern=match)


async def flush_cache() -> None:
    """Clear the entire Ava Redis exact cache, including user-scoped keys."""
    redis = get_redis_client()
    try:
        for match in (f"{_PREFIX}*", "rag_user_*:cache:*"):
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=match, count=5000)
                if keys:
                    await redis.unlink(*keys)
                if cursor == 0:
                    break
        logger.info("Redis cache flushed successfully")
    except Exception as exc:
        logger.warning("Redis cache flush failed", error=str(exc))
