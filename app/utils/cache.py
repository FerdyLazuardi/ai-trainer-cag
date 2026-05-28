"""
Hybrid query cache decorator.
Caches RAG pipeline results first using Redis (exact match), 
then using Qdrant (semantic similarity match) to catch variations of the same query.
"""
import hashlib
import json
import uuid

from loguru import logger
from qdrant_client import models as qdrant_models
from llama_index.core import Settings

from app.config.settings import get_settings
from app.database.redis_client import get_redis_client
from app.database.qdrant_client import get_qdrant_client
from app.config.embedding_config import ensure_llamaindex_configured
from app.observability import get_tracer, is_observability_enabled

settings = get_settings()

_PREFIX = "rag:cache:"
# Cosine threshold for paraphrase match. text-embedding-3-small puts close
# Bahasa Indonesia paraphrases at ~0.83-0.88 (e.g. "apa prinsip CP" vs
# "sebutkan prinsip CP"). 0.88 was too strict — semantic hits never fired
# (verified in Phoenix: 1/14 cache hits, all from Redis exact-match).
# 0.82 keeps obvious paraphrases together while rejecting topic-drift.
_SEMANTIC_THRESHOLD = 0.82
_semantic_collection_ready = False


def _cache_key(query: str, course_id: int | None = None, namespace: str = "rag") -> str:
    """Generate a deterministic Redis key from the query string and course_id.

    `namespace` lets parallel personas (a-pedi vs askfer) share the cache infra
    without polluting each other. Default 'rag' preserves existing A-Pedi keys
    byte-identically.
    """
    query_hash = hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]
    cid_str = str(course_id) if course_id and course_id > 0 else 'global'
    return f"{namespace}:cache:{cid_str}:{query_hash}"


async def _ensure_semantic_collection() -> None:
    """Ensure the semantic_cache collection exists in Qdrant with course_id + namespace indexes."""
    global _semantic_collection_ready
    if _semantic_collection_ready:
        return

    qdrant = get_qdrant_client()
    collections = await qdrant.client.get_collections()
    existing = {c.name for c in collections.collections}
    if "semantic_cache" not in existing:
        await qdrant.client.create_collection(
            collection_name="semantic_cache",
            vectors_config=qdrant_models.VectorParams(
                size=settings.embedding_dim,
                distance=qdrant_models.Distance.COSINE,
            )
        )
        logger.info("semantic_cache collection created")

    # Always ensure indexes exist (idempotent — Qdrant ignores if already present)
    for field, schema in [
        ("course_id", qdrant_models.PayloadSchemaType.INTEGER),
        ("namespace", qdrant_models.PayloadSchemaType.KEYWORD),
    ]:
        try:
            await qdrant.client.create_payload_index(
                collection_name="semantic_cache",
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass  # Index already exists

    _semantic_collection_ready = True


def _log_cache_event(hit: bool, course_id: int | None, score: float | None = None):
    """Emit a cache_lookup span to Phoenix with hit/miss + similarity attributes."""
    if not is_observability_enabled():
        return
    try:
        from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

        tracer = get_tracer("cache")
        cid = course_id if course_id and course_id > 0 else None
        with tracer.start_as_current_span("cache_lookup") as span:
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.RETRIEVER.value,
            )
            span.set_attribute("cache.hit", 1 if hit else 0)
            span.set_attribute("cache.tag", "cache_hit" if hit else "cache_miss")
            if cid is not None:
                span.set_attribute("course_id", cid)
            if score is not None:
                span.set_attribute("similarity_score", round(float(score), 4))
    except Exception as e:
        logger.warning("Failed to emit cache_lookup span", error=str(e))


async def get_cached_response(
    query: str,
    course_id: int | None = None,
    query_embedding: list[float] | None = None,
    cache_namespace: str = "rag",
) -> dict | None:
    """
    Retrieve a cached RAG response for the given query.
    1. Checks Redis for an exact string match.
    2. Checks Qdrant for a semantic match (e.g. "what is X" vs "explain X").

    `cache_namespace` isolates parallel personas (default 'rag' = A-Pedi).
    Semantic matches are filtered by the same namespace, so Askfer queries
    cannot pull A-Pedi answers and vice-versa.

    If `query_embedding` is supplied, the embedding API call for the semantic
    lookup is skipped — caller can reuse the same vector elsewhere.
    """
    # Normalize 0 to None
    if course_id == 0:
        course_id = None

    # 1. Exact match (Redis)
    redis = get_redis_client()
    key = _cache_key(query, course_id, namespace=cache_namespace)
    try:
        raw = await redis.get(key)
        if raw:
            data = json.loads(raw)
            logger.info("Redis Exact Cache HIT", query=query[:60], course_id=course_id, namespace=cache_namespace)
            _log_cache_event(hit=True, course_id=course_id, score=1.0)
            return data
    except Exception as exc:
        logger.warning("Redis cache GET failed", error=str(exc))

    # 2. Semantic match (Qdrant)
    try:
        await _ensure_semantic_collection()

        if query_embedding is None:
            ensure_llamaindex_configured()
            query_embedding = await Settings.embed_model.aget_query_embedding(query)

        qdrant = get_qdrant_client()

        must_clauses = [
            qdrant_models.FieldCondition(
                key="namespace",
                match=qdrant_models.MatchValue(value=cache_namespace),
            )
        ]
        if course_id is not None and course_id > 0:
            must_clauses.append(
                qdrant_models.FieldCondition(
                    key="course_id",
                    match=qdrant_models.MatchValue(value=course_id),
                )
            )
        query_filter = qdrant_models.Filter(must=must_clauses)

        search_result = await qdrant.client.query_points(
            collection_name="semantic_cache",
            query=query_embedding,
            limit=1,
            with_payload=True,
            score_threshold=_SEMANTIC_THRESHOLD,
            query_filter=query_filter
        )

        if search_result.points:
            best_hit = search_result.points[0]
            logger.info("Qdrant Semantic Cache HIT", query=query[:60], score=round(best_hit.score, 4), course_id=course_id, namespace=cache_namespace)
            _log_cache_event(hit=True, course_id=course_id, score=best_hit.score)
            return best_hit.payload

    except Exception as exc:
        logger.warning("Semantic cache GET failed", error=str(exc))

    logger.debug("Cache MISS", query=query[:60], course_id=course_id, namespace=cache_namespace)
    _log_cache_event(hit=False, course_id=course_id)
    return None


async def set_cached_response(
    query: str,
    answer: str,
    sources: list[dict],
    course_id: int | None = None,
    ttl: int | None = None,
    query_embedding: list[float] | None = None,
    cache_namespace: str = "rag",
) -> None:
    """
    Store a RAG response in both Redis (with TTL) and Qdrant caches.

    `cache_namespace` isolates parallel personas. Default 'rag' preserves the
    A-Pedi key/payload format byte-identically.

    If `query_embedding` is supplied, the embedding API call is skipped.
    """
    # Cheap rejections first — avoid embedding work for short/error responses.
    if len(answer) < 100:
        logger.debug("Skipping cache write: response too short", length=len(answer))
        return

    lower_answer = answer.lower()
    error_phrases = ["maaf saya tidak", "terjadi kesalahan", "maaf, aku tidak menemukan", "maaf, saya tidak menemukan"]
    if any(phrase in lower_answer for phrase in error_phrases):
        logger.debug("Skipping cache write: response contains error phrase")
        return

    # Normalize 0 to None
    if course_id == 0:
        course_id = None

    # 1. Set Exact match (Redis)
    redis = get_redis_client()
    key = _cache_key(query, course_id, namespace=cache_namespace)
    ttl_ = ttl or settings.cache_query_ttl_seconds

    # Base payload
    payload = {"answer": answer, "sources": sources, "namespace": cache_namespace}
    if course_id:  # This correctly handles both None and 0
        payload["course_id"] = int(course_id)

    try:
        await redis.set(key, json.dumps(payload), ex=ttl_)
    except Exception as exc:
        logger.warning("Redis cache SET failed", error=str(exc))

    # 2. Set Semantic match (Qdrant)
    try:
        await _ensure_semantic_collection()

        if query_embedding is None:
            ensure_llamaindex_configured()
            query_embedding = await Settings.embed_model.aget_query_embedding(query)

        qdrant = get_qdrant_client()

        point_id = str(uuid.uuid4())
        await qdrant.client.upsert(
            collection_name="semantic_cache",
            wait=True,
            points=[
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=query_embedding,
                    payload=payload
                )
            ]
        )
        logger.debug(f"Semantic Cache SET for query='{query[:60]}' course_id={course_id} namespace={cache_namespace}")
    except Exception as exc:
        logger.warning("Semantic cache SET failed", error=str(exc))


async def flush_cache_by_namespace(namespace: str) -> None:
    """
    Delete Qdrant points and Redis keys for a specific cache namespace
    (e.g. 'askfer'). Used by the profile.md auto-refresh watcher and any
    lightweight refresh path that needs to invalidate one persona's cache
    without touching A-Pedi's cache.
    """
    if not namespace:
        return

    # 1. Clear Redis cache keys for this namespace
    redis = get_redis_client()
    try:
        cursor = 0
        prefix = f"{namespace}:cache:"
        while True:
            cursor, keys = await redis.scan(cursor, match=f"{prefix}*", count=100)
            if keys:
                await redis.delete(*keys)
            if cursor == 0:
                break
        logger.info("Redis namespace cache flushed", namespace=namespace)
    except Exception as exc:
        logger.warning("Redis namespace cache flush failed", error=str(exc))

    # 2. Clear Qdrant semantic cache for this namespace
    qdrant = get_qdrant_client()
    try:
        collections = await qdrant.client.get_collections()
        if "semantic_cache" in {c.name for c in collections.collections}:
            await qdrant.client.delete(
                collection_name="semantic_cache",
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="namespace",
                                match=qdrant_models.MatchValue(value=namespace)
                            )
                        ]
                    )
                )
            )
            logger.info("Semantic namespace cache flushed", namespace=namespace)
    except Exception as exc:
        logger.warning("Semantic namespace cache flush failed", error=str(exc))


async def flush_cache_by_course(course_id: int) -> None:
    """
    Delete Qdrant points and Redis keys for a specific course_id.
    """
    # 1. Clear Redis cache keys for this course
    redis = get_redis_client()
    try:
        cursor = 0
        prefix = f"{_PREFIX}{course_id}:"
        while True:
            cursor, keys = await redis.scan(cursor, match=f"{prefix}*", count=100)
            if keys:
                await redis.delete(*keys)
            if cursor == 0:
                break
        logger.info("Redis course cache flushed", course_id=course_id)
    except Exception as exc:
        logger.warning("Redis course cache flush failed", error=str(exc))
        
    # 2. Clear Qdrant semantic cache for this course
    qdrant = get_qdrant_client()
    try:
        collections = await qdrant.client.get_collections()
        if "semantic_cache" in {c.name for c in collections.collections}:
            await qdrant.client.delete(
                collection_name="semantic_cache",
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="course_id",
                                match=qdrant_models.MatchValue(value=course_id)
                            )
                        ]
                    )
                )
            )
            logger.info("Semantic course cache flushed", course_id=course_id)
    except Exception as exc:
        logger.warning("Semantic course cache flush failed", error=str(exc))


async def flush_cache() -> None:
    """
    Clear both Redis exact cache and Qdrant semantic cache completely.
    """
    global _semantic_collection_ready
    # 1. Clear Redis cache keys starting with the prefix
    redis = get_redis_client()
    try:
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor, match=f"{_PREFIX}*", count=100)
            if keys:
                await redis.delete(*keys)
            if cursor == 0:
                break
        logger.info("Redis cache flushed successfully")
    except Exception as exc:
        logger.warning("Redis cache flush failed", error=str(exc))
        
    # 2. Clear Qdrant semantic cache
    qdrant = get_qdrant_client()
    try:
        collections = await qdrant.client.get_collections()
        if "semantic_cache" in {c.name for c in collections.collections}:
            await qdrant.client.delete_collection("semantic_cache")
        _semantic_collection_ready = False  # Reset so next call recreates
        await _ensure_semantic_collection()
        logger.info("Semantic cache flushed successfully")
    except Exception as exc:
        logger.warning("Semantic cache flush failed", error=str(exc))
