"""
Hybrid query cache decorator.
Caches RAG pipeline results first using Redis (exact match), 
then using Qdrant (semantic similarity match) to catch variations of the same query.
"""
import asyncio
import hashlib
import json
import time
import uuid
from typing import Any

from loguru import logger
from qdrant_client import models as qdrant_models
from llama_index.core import Settings

from app.config.settings import get_settings
from app.database.redis_client import get_redis_client
from app.database.qdrant_client import get_qdrant_client
from app.config.embedding_config import ensure_llamaindex_configured
from app.utils.logger_batch import batch_logger

settings = get_settings()

_PREFIX = "rag:cache:"
# Cosine threshold for paraphrase match on the Qdrant semantic cache.
#
# CALIBRATED (pre-launch, bge-m3 on real LMS-domain queries): the old 0.60
# was far too loose. A live diagnostic on the dev cache showed different-intent
# queries that merely SHARE AN ENTITY collide hard — e.g. "produk amartha apa
# saja" scored 0.74-0.78 against a cached "apa itu amartha" company-overview
# answer and was wrongly served it (the reported bug). A 30-query / 6-intent
# sweep put the worst cross-intent collision at ~0.84 and showed false-positives
# only vanish at >=0.85, while legitimate paraphrases stay well above it
# (re-asks of the same intent cluster 0.85-0.91). The earlier comment claiming
# bge-m3 paraphrases sit in a flat 0.55-0.75 band was WRONG — measured live,
# genuine paraphrases land 0.88-0.91.
#
# 0.88 is chosen DELIBERATELY ABOVE the 0.85 dev-measured false-positive floor:
# the dev cache is only a handful of points and the real corpus is 13k users
# with far more varied phrasing, so the production score distribution will be
# WIDER in both directions. Until we have real traffic, bias hard toward
# precision — a wrong cached answer is user-visible to 13k people, a miss just
# costs one extra LLM call (and the Redis exact layer still catches verbatim
# repeats). POST-LAUNCH: recalibrate from agent_logs (cache_hit / cache_score /
# query_hash are already logged, and the query text is now stored in the cache
# payload) by histogramming real cache_score values, then lower toward the true
# precision/recall knee. Do NOT trust this number from dev data alone.
#
# H8 (DECOUPLED from D3): this is a SEMANTIC-SIMILARITY threshold for cache
# paraphrase matching — it is NOT the faithfulness/hallucination judge
# threshold τ=0.75 (D3). Different axis entirely: τ scores whether a generated
# answer is grounded in retrieved context; this scores whether two QUERIES are
# close enough to reuse a cached answer. The two numbers are unrelated; tune
# this on its own paraphrase test-set, NOT against τ.
_SEMANTIC_THRESHOLD = 0.88
_semantic_collection_ready = False
_semantic_collection_lock: asyncio.Lock | None = None


def _get_semantic_lock() -> asyncio.Lock:
    """Lazy-init the collection creation lock (must be inside a running loop)."""
    global _semantic_collection_lock
    if _semantic_collection_lock is None:
        _semantic_collection_lock = asyncio.Lock()
    return _semantic_collection_lock

# ── Semantic-cache eviction ──────────────────────────────────────────────────
# Redis entries expire via native TTL, but Qdrant has no TTL — without eviction
# `semantic_cache` grows unbounded (a real OOM risk on the shared 8GB box).
# Each point is stamped with `created_at` (epoch). Expired points are (a) never
# served — filtered out at read time — and (b) physically pruned by a lazy,
# time-gated sweep so we don't pay a delete-by-filter on every write.
_last_cache_prune_ts: float = 0.0
# Minimum gap between prune sweeps. The sweep itself is one delete-by-filter,
# so once per 10 min is plenty to keep the collection bounded.
_CACHE_PRUNE_INTERVAL_SECONDS = 600
# Hard cap on semantic_cache size. At 600 DAU each generating ~5 cached
# responses/session, the cache would otherwise grow ~3k points/day and OOM
# Qdrant within weeks (each point is 1024-d float32 = 4KB just for the
# vector, plus payload). When count > MAX, prune the oldest 10% in one
# sweep so we never hit a cliff.
_SEMANTIC_CACHE_MAX_POINTS = 50_000
_SEMANTIC_CACHE_PRUNE_BATCH = 5_000


def _query_hash(query: str) -> str:
    """sha256[:16] of the canonicalized query.

    Canonical form = `strip().lower()` — same scheme used by the Redis cache
    key. Shared by `_cache_key` and `_log_cache_event` so admin can join
    agent_logs `query_hash` rows to live cache state without re-hashing user
    text. Keep this canonicalization stable: changing it invalidates the
    live join between agent_logs and the cache.
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


async def _ensure_semantic_collection() -> None:
    """Ensure the semantic_cache collection exists in Qdrant with course_id + namespace indexes.

    Guarded by asyncio.Lock so concurrent cold-start requests don't all
    race to create_collection + create_payload_index simultaneously.
    """
    global _semantic_collection_ready
    if _semantic_collection_ready:
        return

    lock = _get_semantic_lock()
    async with lock:
        # Re-check after acquiring lock — another coroutine may have finished
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
                    on_disk=True,
                ),
                on_disk_payload=True,
            )
            logger.info("semantic_cache collection created")

        # Always ensure indexes exist (idempotent — Qdrant ignores if already present)
        for field, schema in [
            ("course_id", qdrant_models.PayloadSchemaType.INTEGER),
            ("namespace", qdrant_models.PayloadSchemaType.KEYWORD),
            ("created_at", qdrant_models.PayloadSchemaType.FLOAT),
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
    BatchLogger._redact_entry (query is in _PII_COLUMNS). Failures here
    are swallowed at debug level so a Postgres/Redis blip never blocks the
    cache hot path.

    Args:
        hit: True for Redis exact or Qdrant semantic match, False for miss.
        course_id: LMS course id, or None/0 for non-course-scoped queries.
        query: The user query text (PII-redacted by BatchLogger).
        namespace: Cache namespace ('rag', 'rag_user_<id>', 'portfolio', etc).
        score: Cosine similarity (Qdrant semantic hit) or 1.0 (Redis exact),
            or None (miss).
    """
    try:
        cid = course_id if course_id and course_id > 0 else None
        # Same scheme as _cache_key via the shared _query_hash() helper, so
        # admin can join agent_logs rows to live cache state without
        # re-hashing user text. (Earlier the two sites diverged — log path
        # hashed (query or "").encode("utf-8") with no strip/lower, which
        # made every case-mismatched query un-joinable. Fixed in f59fe3d
        # follow-up: both call _query_hash.)
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
    query_embedding: list[float] | None = None,
    cache_namespace: str = "rag",
) -> dict | None:
    """
    Retrieve a cached RAG response for the given query.
    1. Checks Redis for an exact string match.
    2. Checks Qdrant for a semantic match (e.g. "what is X" vs "explain X").

    `cache_namespace` isolates parallel personas (default 'rag' = Ava).
    Semantic matches are filtered by the same namespace, so Askfer queries
    cannot pull Ava answers and vice-versa.

    If `query_embedding` is supplied, the embedding API call for the semantic
    lookup is skipped — caller can reuse the same vector elsewhere.
    """
    # Normalize 0 to None
    if course_id == 0:
        course_id = None

    # 1. Exact match (Redis)
    import time as _time_cache
    _t_redis = _time_cache.perf_counter()
    redis = get_redis_client()
    key = _cache_key(query, course_id, namespace=cache_namespace)
    try:
        raw = await redis.get(key)
        logger.debug(f"[CACHE TIMING] redis.get: {_time_cache.perf_counter()-_t_redis:.3f}s")
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

    # 2. Semantic match (Qdrant) — wrapped in 3s timeout so a slow/hung Qdrant
    # never blocks the hot path for 10+ seconds on a cache miss.
    _t_ensure = _time_cache.perf_counter()
    import asyncio as _asyncio
    try:
        await _ensure_semantic_collection()
        logger.debug(f"[CACHE TIMING] _ensure_semantic_collection: {_time_cache.perf_counter()-_t_ensure:.3f}s")

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
        min_created = time.time() - settings.cache_query_ttl_seconds
        must_clauses.append(
            qdrant_models.FieldCondition(
                key="created_at",
                range=qdrant_models.Range(gte=min_created),
            )
        )
        query_filter = qdrant_models.Filter(must=must_clauses)

        try:
            search_result = await _asyncio.wait_for(
                qdrant.client.query_points(
                    collection_name="semantic_cache",
                    query=query_embedding,
                    limit=1,
                    with_payload=True,
                    score_threshold=_SEMANTIC_THRESHOLD,
                    query_filter=query_filter
                ),
                timeout=3.0,
            )
        except _asyncio.TimeoutError:
            logger.debug("Semantic cache query timed out (>3s) — treating as miss")
            search_result = None

        if search_result is not None and search_result.points:
            best_hit = search_result.points[0]
            logger.info("Qdrant Semantic Cache HIT", query=query[:60], score=round(best_hit.score, 4), course_id=course_id, namespace=cache_namespace)
            await _log_cache_event(
                hit=True, course_id=course_id, score=best_hit.score,
                query=query, namespace=cache_namespace,
            )
            return best_hit.payload

    except Exception as exc:
        logger.warning(f"Semantic cache GET failed: {type(exc).__name__}: {exc}")

    logger.debug("Cache MISS", query=query[:60], course_id=course_id, namespace=cache_namespace)
    await _log_cache_event(
        hit=False, course_id=course_id,
        query=query, namespace=cache_namespace,
    )
    return None


async def get_cached_response_multi_ns(
    query: str,
    namespaces: list[str],
    course_id: int | None = None,
    query_embedding: list[float] | None = None,
) -> dict | None:
    """Check multiple cache namespaces in a single round-trip per backend.

    Optimisation for the common pattern in chat.py that checks both a
    private (user-scoped) and global namespace.  Instead of 2× Redis GET +
    2× Qdrant semantic query, this does:
      1. Redis MGET (1 round-trip, both keys)
      2. Qdrant query_points with a 'should' filter (OR across namespaces)

    Saves ~50-200ms per cache-miss turn at 600 DAU scale.
    """
    if course_id == 0:
        course_id = None
    if not namespaces:
        return None

    # 1. Redis MGET — all keys in one round-trip
    redis = get_redis_client()
    keys = [_cache_key(query, course_id, namespace=ns) for ns in namespaces]
    try:
        results = await redis.mget(*keys)
        for i, raw in enumerate(results):
            if raw:
                data = json.loads(raw)
                logger.info(
                    "Redis Exact Cache HIT (multi-ns)",
                    query=query[:60],
                    course_id=course_id,
                    namespace=namespaces[i],
                )
                await _log_cache_event(
                    hit=True, course_id=course_id, score=1.0,
                    query=query, namespace=namespaces[i],
                )
                return data
    except Exception as exc:
        logger.warning("Redis multi-ns cache MGET failed", error=str(exc))

    # 2. Qdrant semantic — single query with OR filter on namespaces
    import asyncio as _asyncio
    try:
        await _ensure_semantic_collection()

        if query_embedding is None:
            ensure_llamaindex_configured()
            query_embedding = await Settings.embed_model.aget_query_embedding(query)

        qdrant = get_qdrant_client()

        # OR across namespaces ("should" = any-of)
        ns_clauses = [
            qdrant_models.FieldCondition(
                key="namespace",
                match=qdrant_models.MatchValue(value=ns),
            )
            for ns in namespaces
        ]

        must_clauses = []
        if course_id is not None and course_id > 0:
            must_clauses.append(
                qdrant_models.FieldCondition(
                    key="course_id",
                    match=qdrant_models.MatchValue(value=course_id),
                )
            )
        min_created = time.time() - settings.cache_query_ttl_seconds
        must_clauses.append(
            qdrant_models.FieldCondition(
                key="created_at",
                range=qdrant_models.Range(gte=min_created),
            )
        )

        query_filter = qdrant_models.Filter(
            must=must_clauses,
            should=ns_clauses,
            min_should=qdrant_models.MinShould(min_count=1),  # type: ignore[call-arg]  # qdrant-client stub version gap
        )

        try:
            search_result = await _asyncio.wait_for(
                qdrant.client.query_points(
                    collection_name="semantic_cache",
                    query=query_embedding,
                    limit=1,
                    with_payload=True,
                    score_threshold=_SEMANTIC_THRESHOLD,
                    query_filter=query_filter,
                ),
                timeout=3.0,
            )
        except _asyncio.TimeoutError:
            logger.debug("Semantic cache multi-ns query timed out (>3s) — treating as miss")
            search_result = None

        if search_result is not None and search_result.points:
            best_hit = search_result.points[0]
            hit_ns = (best_hit.payload or {}).get("namespace", "unknown")
            logger.info(
                "Qdrant Semantic Cache HIT (multi-ns)",
                query=query[:60],
                score=round(best_hit.score, 4),
                course_id=course_id,
                namespace=hit_ns,
            )
            await _log_cache_event(
                hit=True, course_id=course_id, score=best_hit.score,
                query=query, namespace=hit_ns,
            )
            return best_hit.payload

    except Exception as exc:
        logger.warning(f"Semantic cache multi-ns GET failed: {type(exc).__name__}: {exc}")

    logger.debug("Cache MISS (multi-ns)", query=query[:60], course_id=course_id, namespaces=namespaces)
    # For multi-ns miss, log under the first namespace as a representative
    # bucket (the full set is in the log message above). cache_namespace
    # column is single-valued for indexing/grouping convenience.
    await _log_cache_event(
        hit=False, course_id=course_id,
        query=query, namespace=namespaces[0] if namespaces else "multi",
    )
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
    Ava key/payload format byte-identically.

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
    payload: dict[str, Any] = {"answer": answer, "sources": sources, "namespace": cache_namespace}
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

        # Stamp with epoch so the read-time TTL filter + lazy prune can age it
        # out. Redis has native TTL; Qdrant does not, so we carry it explicitly.
        # Also stamp the originating query text (capped) so a semantic HIT can be
        # audited post-launch: "query X was served the answer cached for query Y
        # — was that reuse correct?". This is the data the threshold recalibration
        # needs and was previously impossible (only the answer was stored, so the
        # cached point's source query had to be guessed). Not used on the read
        # path; pure observability.
        qdrant_payload = {**payload, "created_at": time.time(), "query": query[:500]}
        point_id = str(uuid.uuid4())
        await qdrant.client.upsert(
            collection_name="semantic_cache",
            wait=True,
            points=[
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=query_embedding,
                    payload=qdrant_payload
                )
            ]
        )
        logger.debug(f"Semantic Cache SET for query='{query[:60]}' course_id={course_id} namespace={cache_namespace}")

        # Lazy, time-gated eviction: at most once per _CACHE_PRUNE_INTERVAL,
        # delete all points older than the cache TTL. Cheap insurance against
        # unbounded growth without a delete-by-filter on every write.
        await _maybe_prune_semantic_cache(qdrant)
    except Exception as exc:
        logger.warning(f"Semantic cache SET failed: {type(exc).__name__}: {exc}")


async def get_or_compute_cached_response(
    query: str,
    compute_fn,
    *,
    course_id: int | None = None,
    cache_namespace: str = "rag",
    lock_ttl_seconds: int = 30,
    wait_total_seconds: float = 8.0,
    poll_interval_seconds: float = 0.4,
) -> tuple[dict, bool]:
    """Get a cached response OR compute one — with single-flight semantics.

    On a cold cache (or after a flush), N concurrent users asking the
    same question would otherwise each fire their own LLM call. With
    600 DAU, p99 of the first request after a cache flush is the
    cost of N parallel calls, not 1.

    Pattern:
      1. Try cache (Redis exact, then Qdrant semantic).
      2. If hit → return (cached_dict, True).
      3. If miss → try to acquire a Redis SETNX lock keyed on the
         same cache key. If we get the lock, run `compute_fn`,
         store the result, release the lock, return (result, False).
      4. If another caller holds the lock, poll Redis exact cache
         for the result they will publish, up to wait_total_seconds.
         If the result appears, return it. If the wait times out,
         fall through to compute_fn anyway (defense in depth — a
         stuck lock must not hang the request indefinitely).

    Returns a tuple (dict, from_cache: bool) so callers can log /
    meter cache-hit vs cache-compute differently. The dict format
    matches `get_cached_response()`: {"answer": ..., "sources":
    [...], "namespace": ...} and is suitable for return as the
    `cached` field of /api/v1/chat.

    `compute_fn` is an async callable that takes no arguments and
    returns the same dict shape. It will be called AT MOST ONCE
    per (key, time window) across all concurrent callers.
    """
    # 1. Cache lookup (fast path)
    cached = await get_cached_response(
        query, course_id=course_id, cache_namespace=cache_namespace,
    )
    if cached is not None:
        return cached, True

    # 2. Miss: try single-flight lock.
    redis = get_redis_client()
    key = _cache_key(query, course_id, namespace=cache_namespace)
    lock_key = f"{key}:lock"
    # SET NX EX: atomic acquire-or-fail. value is a random nonce so
    # release only fires on the lock we own (not someone else's).
    nonce = uuid.uuid4().hex
    got_lock = False
    try:
        got_lock = await redis.set(lock_key, nonce, nx=True, ex=lock_ttl_seconds)
    except Exception as exc:
        logger.warning("singleflight SETNX failed (falling through to compute): {}", exc)

    if got_lock:
        try:
            result = await compute_fn()
            # Populate the cache so concurrent pollers (and the next
            # non-stampede caller) find it. Without this write-back the
            # single-flight lock is a no-op: 1 holder + 99 pollers all
            # fall through to their own compute_fn, defeating the
            # stampede mitigation and thundering-herd-ing the LLM.
            # set_cached_response has its own error guards — failures
            # are logged inside, not raised — so a Qdrant/Redis blip on
            # the write path can never take down a successful LLM call.
            try:
                await set_cached_response(
                    query=query,
                    answer=result["answer"],
                    sources=result["sources"],
                    course_id=course_id,
                    cache_namespace=cache_namespace,
                )
            except Exception as exc:
                logger.warning(
                    "singleflight holder: cache write failed (non-fatal)",
                    error=str(exc),
                )
            return result, False
        finally:
            # Release only if WE still own the lock (Lua: get+del
            # only if value matches). Without the Lua check, a slow
            # compute past lock_ttl could let another caller
            # delete our (now-stale) lock value.
            try:
                await _release_lock_if_owner(redis, lock_key, nonce)
            except Exception as exc:
                logger.warning("singleflight lock release failed (non-fatal): {}", exc)

    # 3. Another caller is computing. Poll Redis exact cache for
    # the result they'll publish.
    waited = 0.0
    while waited < wait_total_seconds:
        await asyncio.sleep(poll_interval_seconds)
        waited += poll_interval_seconds
        try:
            raw = await redis.get(key)
            if raw:
                return json.loads(raw), True
        except Exception as exc:
            logger.warning("singleflight poll failed (continuing): {}", exc)

    # 4. Timeout: fall through to compute anyway. The lock holder
    # may have crashed; we cannot let their absence starve us.
    logger.warning(
        "singleflight wait timed out, computing anyway",
        waited_s=round(waited, 2),
        lock_key=lock_key,
    )
    result = await compute_fn()
    return result, False


_LUA_RELEASE_IF_OWNER = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


# Lazy-init Script object for SHA-cached EVALSHA on the single-flight lock
# release path. Same SHA-caching win as the _APPEND_HISTORY_LUA script in
# memory.py — first call uploads the Lua body, subsequent calls send only
# the SHA + KEYS + ARGS via EVALSHA.
_release_lock_script = None


def _get_release_lock_script(redis):
    global _release_lock_script
    if _release_lock_script is None:
        _release_lock_script = redis.register_script(_LUA_RELEASE_IF_OWNER)
    return _release_lock_script


async def _release_lock_if_owner(redis, key: str, nonce: str) -> None:
    """Release a single-flight lock only if WE still own it.

    Implemented as a small Lua script so the GET+DEL is atomic.
    A slow compute_fn() can let the lock TTL expire, in which case
    another caller may now hold the lock — deleting it would
    let a THIRD caller compute redundantly.
    """
    # The redis-py async client exposes script loading via .eval().
    # Fall back to a non-atomic delete if the script fails to load
    # (e.g. older redis-server without eval).
    try:
        # SHA-cached EVALSHA via registered Script object — see the lazy
        # _release_lock_script singleton above. Same EVALSHA-with-NOSCRIPT-
        # fallback pattern as memory.py; no per-call script transfer.
        script = _get_release_lock_script(redis)
        await script(keys=[key], args=[nonce])
    except Exception:
        try:
            current = await redis.get(key)
            if current == nonce:
                await redis.delete(key)
        except Exception:
            pass


async def _maybe_prune_semantic_cache(qdrant) -> None:
    """Delete expired semantic_cache points, throttled to one sweep per interval.

    Two eviction triggers, both throttled to once per interval so we
    don't pay the cost on every write:

      1. TTL sweep: delete every point older than the configured
         cache_query_ttl_seconds. Catches long-tail traffic that
         writes a few points/day forever (otherwise a niche persona
         could keep points alive for months).

      2. Size cap: if total count > _SEMANTIC_CACHE_MAX_POINTS,
         delete the oldest _SEMANTIC_CACHE_PRUNE_BATCH points. This
         is the actual OOM prevention — at 600 DAU each generating
         ~5 cached responses/session the cache would otherwise grow
         ~3k points/day. Keeping the ceiling at 50k bounds RAM at
         ~200MB (50k × 4KB vectors) on the 8GB host, leaving the
         rest for the other 4 Qdrant collections + Postgres + Redis.

    Fire-and-forget: any failure is logged and swallowed so a prune
    hiccup never breaks the cache-write path.
    """
    global _last_cache_prune_ts
    now = time.time()
    if now - _last_cache_prune_ts < _CACHE_PRUNE_INTERVAL_SECONDS:
        return
    _last_cache_prune_ts = now  # claim the slot before awaiting so concurrent writers don't double-sweep
    try:
        # 1. TTL sweep: physically delete all expired points.
        cutoff = now - settings.cache_query_ttl_seconds
        await qdrant.client.delete(
            collection_name="semantic_cache",
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="created_at",
                            range=qdrant_models.Range(lt=cutoff),
                        )
                    ]
                )
            ),
            wait=False,
        )
        logger.debug("semantic_cache TTL sweep issued", cutoff_epoch=round(cutoff))

        # 2. Size cap: if the collection is over the hard ceiling,
        # drop the oldest N points by created_at. count() is cheap
        # (returns the Qdrant-side cached count, no scan).
        info = await qdrant.client.get_collection("semantic_cache")
        count = info.points_count or 0
        if count > _SEMANTIC_CACHE_MAX_POINTS:
            # Fetch the oldest batch by ascending created_at, then
            # delete by id list. Two round-trips but bounded.
            scroll = await qdrant.client.scroll(
                collection_name="semantic_cache",
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="created_at",
                            range=qdrant_models.Range(
                                lt=now - 60  # ignore writes from the last 60s
                            ),
                        )
                    ]
                ),
                order_by=qdrant_models.OrderBy(
                    key="created_at",
                    direction=qdrant_models.Direction.ASC,
                ),
                limit=_SEMANTIC_CACHE_PRUNE_BATCH,
                with_payload=False,
                with_vectors=False,
            )
            old_ids = [str(p.id) for p in (scroll[0] or [])]
            if old_ids:
                await qdrant.client.delete(
                    collection_name="semantic_cache",
                    points_selector=qdrant_models.PointIdsListSelector(  # type: ignore[attr-defined]  # qdrant stub version gap
                        points=old_ids,
                    ),
                    wait=False,
                )
                logger.info(
                    "semantic_cache size-cap prune",
                    count_before=count,
                    pruned=len(old_ids),
                    cap=_SEMANTIC_CACHE_MAX_POINTS,
                )
    except Exception as exc:
        logger.warning("semantic_cache prune failed (non-fatal)", error=str(exc))


async def flush_cache_by_namespace(namespace: str) -> None:
    """
    Delete Qdrant points and Redis keys for a specific cache namespace
    (e.g. 'askfer'). Used by the profile.md auto-refresh watcher and any
    lightweight refresh path that needs to invalidate one persona's cache
    without touching Ava's cache.
    """
    if not namespace:
        return

    # 1. Clear Redis cache keys for this namespace
    redis = get_redis_client()
    try:
        # 5000 is the new sweet spot for 8.x with I/O threading + batched
        # prefetch (8.8); ~3-4× faster on large key sets than the old 1000.
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

    H2 (correctness): the Redis EXACT layer is keyed `{namespace}:cache:{cid}:{hash}`
    and is checked FIRST (short-circuits before the semantic layer). Sweeping only
    the global `rag:cache:{cid}:*` prefix left every PER-USER entry
    (`rag_user_{uid}:cache:{cid}:*`, written by chat.py's private_ns path) alive
    after a Moodle re-ingest → those users kept getting the STALE pre-ingest
    answer from the exact layer even though the semantic layer (course_id filter,
    namespace-agnostic) had already been purged. We now sweep BOTH the global and
    all user namespaces for this course. (Cross-dep C1.)
    """
    # 1. Clear Redis cache keys for this course — global + all user namespaces.
    redis = get_redis_client()
    # `rag:cache:{cid}:*` (global) and `rag_user_*:cache:{cid}:*` (per-user) do
    # not overlap: after "rag" the global key has ":" and the user key has "_".
    patterns = [
        f"{_PREFIX}{course_id}:*",          # rag:cache:{cid}:*
        f"rag_user_*:cache:{course_id}:*",  # rag_user_{uid}:cache:{cid}:*
    ]
    for match in patterns:
        try:
            # 5000 is the new sweet spot for 8.x with I/O threading + batched
            # prefetch (8.8); ~3-4× faster on large key sets than the old 1000.
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
        # 5000 is the new sweet spot for 8.x with I/O threading + batched
        # prefetch (8.8); ~3-4× faster on large key sets than the old 1000.
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
