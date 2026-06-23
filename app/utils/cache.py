"""
Query cache — Redis exact-match only.

Previously a two-layer cache (Redis exact + Qdrant semantic-similarity). The
semantic layer was REMOVED: a cosine match (>=0.88) could serve an answer cached
for a DIFFERENT user's question — a cross-intent / cross-tone leak we observed in
practice ("emng topiknya apa aja" hit a product-list answer at 0.8838). Because
answers can be personalized (tone, user history, role), serving one user's
answer to another via a fuzzy match is a correctness AND privacy hazard.

Redis exact-match has no such risk: the key is sha256 of the exact (canonical)
query string, so only a byte-identical re-ask hits, and a grounded turn is
deterministic (temp 0) so the re-served answer is the one that query would have
produced anyway.

Tradeoff: paraphrases ("apa itu modal" vs "modal itu apa") are now cache misses
— each costs one extra LLM call, never a wrong/foreign answer.
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
    """sha256[:16] of the canonicalized query.

    Canonical form = `strip().lower()`. Shared by `_cache_key` and
    `_log_cache_event` so admin can join agent_logs `query_hash` rows to live
    cache state without re-hashing user text. Keep this canonicalization stable:
    changing it invalidates the live join between agent_logs and the cache.
    """
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def _cache_key(query: str, course_id: int | None = None, namespace: str = "rag") -> str:
    """Generate a deterministic Redis key from the query string and course_id.

    `namespace` lets parallel personas (ava vs askfer) share the cache infra
    without polluting each other. Default 'rag' preserves existing Ava keys
    byte-identically.
    """
    cid_str = str(course_id) if course_id and course_id > 0 else 'global'
    return f"{namespace}:cache:{cid_str}:{_query_hash(query)}"


async def _log_cache_event(
    *,
    hit: bool,
    course_id: int | None,
    query: str,
    namespace: str,
    score: float | None = None,
) -> None:
    """Persist a cache_lookup event to agent_logs via BatchLogger.

    Cache observability flows through the same agent_logs chokepoint as
    turn-level logs. PII in `query` is auto-redacted by
    BatchLogger._redact_entry. Failures here are swallowed at debug level so a
    Postgres/Redis blip never blocks the cache hot path.

    `score` is 1.0 on a Redis exact hit, None on a miss (there is no longer a
    semantic-similarity score to record).
    """
    try:
        cid = course_id if course_id and course_id > 0 else None
        # Same scheme as _cache_key via the shared _query_hash() helper, so admin
        # can join agent_logs rows to live cache state without re-hashing text.
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
    """Retrieve a cached RAG response via Redis exact string match.

    `cache_namespace` isolates parallel personas (default 'rag' = Ava). Only a
    byte-identical (canonicalized) re-ask of the same query hits.
    """
    # Normalize 0 to None
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
    await _log_cache_event(
        hit=False, course_id=course_id,
        query=query, namespace=cache_namespace,
    )
    return None


async def set_cached_response(
    query: str,
    answer: str,
    sources: list[dict],
    course_id: int | None = None,
    ttl: int | None = None,
    cache_namespace: str = "rag",
) -> None:
    """Store a RAG response in Redis with a TTL.

    `cache_namespace` isolates parallel personas. Default 'rag' preserves the
    Ava key/payload format byte-identically.
    """
    # Cheap rejections first — never cache short/error responses.
    if len(answer) < 100:
        logger.debug("Skipping cache write: response too short", length=len(answer))
        return

    lower_answer = answer.lower()
    # Never cache a NOT-FOUND / error reply. If one gets cached, every future
    # re-ask is served the stale "belum nemu" even after retrieval is fixed —
    # exactly the topic-panel bug (the generator says "Aku belum nemu info
    # soal X", which the old short list ("maaf, aku tidak menemukan") missed,
    # so the broken answer stuck in cache). Match the phrasings the generator
    # actually emits, not just the formal "maaf" forms.
    error_phrases = [
        "maaf saya tidak", "terjadi kesalahan",
        "maaf, aku tidak menemukan", "maaf, saya tidak menemukan",
        "belum nemu", "belum menemukan", "belum ketemu",
        "sejauh pengetahuanku belum ada", "kurang tahu pasti",
        "sejauh pengetahuanku belum ada", "ga nemu", "gak nemu", "nggak nemu",
        "coba pastikan lagi pertanyaannya", "coba cari dengan kata kunci lain",
    ]
    if any(phrase in lower_answer for phrase in error_phrases):
        logger.debug("Skipping cache write: response contains error/not-found phrase")
        return

    # Normalize 0 to None
    if course_id == 0:
        course_id = None

    redis = get_redis_client()
    key = _cache_key(query, course_id, namespace=cache_namespace)
    ttl_ = ttl or settings.cache_query_ttl_seconds

    payload: dict[str, Any] = {"answer": answer, "sources": sources, "namespace": cache_namespace}
    if course_id:  # handles both None and 0
        payload["course_id"] = int(course_id)

    try:
        await redis.set(key, json.dumps(payload), ex=ttl_)
    except Exception as exc:
        logger.warning("Redis cache SET failed", error=str(exc))


async def flush_cache_by_namespace(namespace: str) -> None:
    """Delete Redis cache keys for a specific namespace (e.g. 'askfer').

    Used by the profile.md auto-refresh watcher and any lightweight refresh path
    that needs to invalidate one persona's cache without touching Ava's.
    """
    if not namespace:
        return

    redis = get_redis_client()
    try:
        # 5000 is the sweet spot for redis 8.x with I/O threading + batched
        # prefetch; ~3-4x faster on large key sets than the old 1000.
        cursor = 0
        prefix = f"{namespace}:cache:"
        while True:
            cursor, keys = await redis.scan(cursor, match=f"{prefix}*", count=5000)
            if keys:
                # UNLINK is non-blocking; background reclamation.
                await redis.unlink(*keys)
            if cursor == 0:
                break
        logger.info("Redis namespace cache flushed", namespace=namespace)
    except Exception as exc:
        logger.warning("Redis namespace cache flush failed", error=str(exc))


async def flush_cache_by_course(course_id: int) -> None:
    """Delete Redis cache keys for a specific course_id — global + all user namespaces.

    The exact layer is keyed `{namespace}:cache:{cid}:{hash}`. Sweeping only the
    global `rag:cache:{cid}:*` prefix would leave every PER-USER entry
    (`rag_user_{uid}:cache:{cid}:*`, written by chat.py's private_ns path) alive
    after a Moodle re-ingest → those users would keep getting the STALE
    pre-ingest answer. So we sweep BOTH the global and all user namespaces.
    """
    redis = get_redis_client()
    # `rag:cache:{cid}:*` (global) and `rag_user_*:cache:{cid}:*` (per-user) do
    # not overlap: after "rag" the global key has ":" and the user key has "_".
    patterns = [
        f"{_PREFIX}{course_id}:*",          # rag:cache:{cid}:*
        f"rag_user_*:cache:{course_id}:*",  # rag_user_{uid}:cache:{cid}:*
        f"{_PREFIX}None:*",                 # rag:cache:None:* (global queries)
        f"rag_user_*:cache:None:*",         # rag_user_{uid}:cache:None:*
    ]
    for match in patterns:
        try:
            cursor = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=match, count=5000)
                if keys:
                    # UNLINK is non-blocking; background reclamation.
                    await redis.unlink(*keys)
                if cursor == 0:
                    break
            logger.info("Redis course cache flushed", course_id=course_id, pattern=match)
        except Exception as exc:
            logger.warning("Redis course cache flush failed", error=str(exc), pattern=match)


async def flush_cache() -> None:
    """Clear the entire Redis exact cache (all `rag:cache:*` keys)."""
    redis = get_redis_client()
    try:
        # 5000: see flush_cache_by_namespace.
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor, match=f"{_PREFIX}*", count=5000)
            if keys:
                # UNLINK is non-blocking; background reclamation.
                await redis.unlink(*keys)
            if cursor == 0:
                break
        logger.info("Redis cache flushed successfully")
    except Exception as exc:
        logger.warning("Redis cache flush failed", error=str(exc))
