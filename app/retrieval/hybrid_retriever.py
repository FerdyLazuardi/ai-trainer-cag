"""
Hybrid retriever — native dense + sparse (BM25) fusion against Qdrant.

We query Qdrant directly (dense KNN + sparse BM25 KNN in one batched
round-trip) instead of going through LlamaIndex's `as_retriever`. Two reasons:

  1. `index.as_retriever(similarity_top_k=k)` never sets the query mode to
     HYBRID, so LlamaIndex silently runs the dense-only branch — the sparse
     BM25 vectors stored at ingestion were never actually queried.
  2. LlamaIndex's `relative_score_fusion` discards the raw dense cosine score
     after min-max normalization. We need that raw cosine (an absolute [0,1]
     signal) for the NOT-FOUND gate in the graph, which the per-query
     normalized fusion score cannot provide.

Embeddings stay on LlamaIndex: the query is embedded with the SAME model used
at ingestion (`Settings.embed_model`, baai/bge-m3 @1024) and the
sparse side uses the SAME fastembed BM25 encoder the vector store used to write
documents (`Qdrant/bm25`). This keeps query/document encodings aligned.
"""
import asyncio
from functools import lru_cache
import hashlib
import time
from typing import Any

import httpx
from loguru import logger
from qdrant_client import models as rest
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llama_index.core import Settings
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.vector_stores.qdrant.utils import fastembed_sparse_encoder

from app.config.settings import get_settings
from app.config.embedding_config import ensure_llamaindex_configured
from app.database.qdrant_client import get_qdrant_client
from app.retrieval.schemas import RetrievedChunk, HybridSearchResult

settings = get_settings()


async def _call_openrouter_rerank(
    query: str,
    chunks: list[RetrievedChunk],
    settings: Any,
) -> list[tuple[int, float]]:
    """
    Call OpenRouter's Rerank API to reorder the given chunks.
    Returns a list of tuples: (original_index, relevance_score) sorted by relevance.
    """
    documents = [c.text for c in chunks]
    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": documents,
        "top_n": len(chunks),
    }
    
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/FerdyLazuardi/peped-BE",
        "X-OpenRouter-Title": "Peped BE RAG System",
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            settings.rerank_base_url,
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        result = response.json()
        
        results = result.get("results", [])
        parsed_results = []
        for r in results:
            idx = r.get("index")
            score = r.get("relevance_score", 0.0)
            if idx is not None:
                parsed_results.append((int(idx), float(score)))
        return parsed_results

# Vector names match QdrantManager._create_*_collection (dense + sparse).
_DENSE_VECTOR_NAME = "text-dense"
_SPARSE_VECTOR_NAME = "text-sparse"
# Must match `fastembed_sparse_model` passed to QdrantVectorStore at ingestion.
_SPARSE_MODEL = "Qdrant/bm25"


_sparse_semaphore: asyncio.Semaphore | None = None

def _get_sparse_semaphore() -> asyncio.Semaphore:
    # Module-global lazy-init (NOT lru_cache): a Semaphore must bind to the
    # running loop on first use, so we create it on first call rather than at
    # import. Concurrency is tunable via SPARSE_ENCODE_CONCURRENCY — default 1
    # because each encode already spans both vCPUs via OMP_NUM_THREADS=2; see
    # the field comment in settings.py before raising it.
    global _sparse_semaphore
    if _sparse_semaphore is None:
        from app.config.settings import get_settings
        _sparse_semaphore = asyncio.Semaphore(get_settings().sparse_encode_concurrency)
    return _sparse_semaphore


@lru_cache(maxsize=1)
def _get_sparse_encoder():
    """Load the fastembed BM25 sparse query encoder once (process lifetime)."""
    return fastembed_sparse_encoder(_SPARSE_MODEL)



def _minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max scale a {id: score} map into [0, 1] within this result set.

    Mirrors LlamaIndex relative_score_fusion: each modality is scaled against
    its OWN min/max, so the unbounded BM25 scale and the cosine scale become
    comparable before weighting. An all-equal (or single-item) set maps to 1.0
    to avoid division by zero.
    """
    if not scores:
        return {}
    values = scores.values()
    hi, lo = max(values), min(values)
    if hi == lo:
        return {node_id: 1.0 for node_id in scores}
    span = hi - lo
    return {node_id: (val - lo) / span for node_id, val in scores.items()}


async def _embed_query_resilient(query: str) -> list[float] | None:
    """Embed the query with bounded retry + timeout (C5).

    The embedding provider is a single remote SPOF. We wrap the call in a
    tenacity retry (exponential backoff) and a per-attempt timeout. On final
    failure we return None — the caller degrades to sparse-only BM25 rather
    than raising, so retrieval stays available during an embedding outage.

    Embedding cache: short-lived Redis exact-match keyed by sha256(query).
    qwen3-8B / bge-m3 embeddings for the same text are byte-deterministic,
    so reusing one for the same query is safe. 24h TTL covers daily-mood
    bursts; the cache grows only as KB traffic grows and is auto-bounded
    by volatile-lru + TTL. Hit rates expected ~2x on FAQ phrasings, much
    higher on repeated profanity/injection probes (which all collapse to
    the same SHA). ponytail: skip if cache hit latency > 50ms (>3σ vs the
    0.5–2.5s fresh embed cost) — disable on Redis blip.
    """
    from app.database.redis_client import get_redis_client

    _t0 = time.monotonic()
    cache_key = f"rag:embed:{hashlib.sha256(query.encode()).hexdigest()[:32]}"
    cache_ttl = 24 * 3600  # 24h
    try:
        redis = get_redis_client()
        cached = await redis.get(cache_key)
        if cached:
            logger.debug(
                f"[TIMING] embed cache HIT: {(time.monotonic()-_t0)*1000:.1f}ms"
            )
            import json
            return json.loads(cached)
    except Exception as exc:
        logger.debug(f"embed cache read skipped: {exc}")

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(settings.embedding_max_attempts),
            wait=wait_exponential(
                multiplier=settings.embedding_backoff_base_seconds,
                max=settings.embedding_backoff_max_seconds,
            ),
            retry=retry_if_exception_type((Exception,)),
            reraise=True,
        ):
            with attempt:
                vec = await asyncio.wait_for(
                    Settings.embed_model.aget_query_embedding(query),
                    timeout=settings.embedding_timeout_seconds,
                )
    except Exception as exc:
        logger.error(
            "Query embedding failed after retries — degrading to sparse-only",
            error=str(exc)[:200],
            error_type=type(exc).__name__,
            query=query[:60],
        )
        return None

    logger.debug(
        f"[TIMING] embed fresh: {(time.monotonic()-_t0)*1000:.1f}ms"
    )
    # Best-effort write — never raises upward.
    try:
        redis = get_redis_client()
        import json
        await redis.set(cache_key, json.dumps(vec), ex=cache_ttl)
    except Exception as exc:
        logger.debug(f"embed cache write skipped: {exc}")
    return vec


async def hybrid_search(
    query: str,
    top_k: int | None = None,
    collection: str | None = None,
    fetch_k: int | None = None,
    query_embedding: list[float] | None = None,
) -> HybridSearchResult:
    """
    Perform native hybrid (dense + sparse BM25) retrieval against Qdrant.

    Ranking uses relative-score fusion: each modality is min-max normalized,
    then combined as `alpha * dense + (1 - alpha) * sparse` where
    `alpha = settings.vector_weight`. The raw dense cosine score is preserved
    on each chunk as `dense_score` for the downstream NOT-FOUND gate.

    `fetch_k` controls the candidate pool pulled per modality before fusion;
    `top_k` is the final cut returned after fusion. A wider pool than the final
    cut gives fusion more to work with (this replaces the old retrieve-wide-
    then-rerank-narrow pattern), so `fetch_k` defaults to the larger of `top_k`
    and `settings.retrieval_top_k`.

    Args:
        query: User query (already rewritten by the pre-processor).
        top_k: Number of fused results to return. Defaults to
            settings.retrieval_top_k.
        collection: Qdrant collection name. Defaults to
            settings.qdrant_kb_collection.
        fetch_k: Candidates per modality before fusion. Defaults to
            max(top_k, settings.retrieval_top_k).
        query_embedding: Optional precomputed dense embedding for THIS exact
            query string (H5). When supplied we skip re-embedding — the caller
            (chat route) already embedded the same text for the cache/LTM
            lookup. MUST be the embedding of `query`; callers guard this with a
            text-equality check. Pass None to embed here.

    Returns:
        HybridSearchResult: fused `chunks` (sorted by descending fused score)
        plus pool-level signals — `pool_max_dense`/`pool_max_sparse` (max raw
        score over the FULL fetch_k pool, pre-slice, for the NOT-FOUND gate)
        and `dense_available`/`sparse_available` (False when that modality could
        not run, e.g. embedding outage → dense-only degrade).
    """
    ensure_llamaindex_configured()
    k = top_k or settings.retrieval_top_k
    pool = fetch_k or max(k, settings.retrieval_top_k)
    collection_name = collection or settings.qdrant_kb_collection
    alpha = settings.vector_weight  # weight on dense; (1 - alpha) on sparse

    qdrant = get_qdrant_client()

    # 1. Encode the query: dense (same embed model as ingestion) + sparse BM25.
    # Run both encodings concurrently — they are independent I/O + CPU calls.
    # Dense: async OpenAI API call (resilient — retry/timeout, may return None).
    # Sparse: CPU-bound fastembed BM25 offloaded to a thread so the loop stays
    # free. C5: we use return_exceptions semantics via separate awaits so a
    # dense-embed failure does NOT abort the (independent) sparse arm — we then
    # run sparse-only instead of failing the whole retrieval.
    encoder = _get_sparse_encoder()

    async def _encode_sparse():
        # Semaphore bounds concurrent fastembed (onnxruntime) encodes so it
        # doesn't saturate all CPU cores and starve the asyncio event loop.
        # Bound is SPARSE_ENCODE_CONCURRENCY (default 1; see settings.py field).
        async with _get_sparse_semaphore():
            return await asyncio.to_thread(encoder, [query])

    # H5: reuse a precomputed embedding for the SAME query text instead of
    # re-embedding. The caller guarantees (by text equality) it matches `query`.
    async def _dense():
        if query_embedding is not None:
            return query_embedding
        return await _embed_query_resilient(query)

    dense_vec: list[float] | BaseException | None = None
    sparse_encoded: Any = None
    dense_vec, sparse_encoded = await asyncio.gather(
        _dense(),
        _encode_sparse(),
        return_exceptions=True,
    )

    # Sparse is local CPU (fastembed) — an exception here is unexpected; treat
    # as sparse unavailable rather than crashing the request.
    sparse_available = not isinstance(sparse_encoded, BaseException)
    dense_available = (not isinstance(dense_vec, BaseException)) and dense_vec is not None

    if not dense_available and not sparse_available:
        logger.error(
            "Both dense and sparse encodings failed — returning empty result",
            query=query[:60],
        )
        return HybridSearchResult(
            chunks=[], pool_max_dense=0.0, pool_max_sparse=0.0,
            dense_available=False, sparse_available=False,
        )

    if not dense_available:
        logger.warning("Dense embedding unavailable — running sparse-only (BM25) retrieval")
    if not sparse_available:
        logger.warning("Sparse encoding unavailable — running dense-only retrieval")

    sparse_idx = sparse_val = None
    if sparse_available:
        sparse_idx, sparse_val = sparse_encoded  # type: ignore[misc]

    # 2. One batched round-trip: dense KNN + sparse BM25 KNN over a `pool`-sized
    #    candidate set per modality. Only include the arms that are available.
    requests: list[rest.QueryRequest] = []
    _dense_req_idx = _sparse_req_idx = None
    if dense_available:
        assert isinstance(dense_vec, list)  # narrowed by dense_available check above
        _dense_req_idx = len(requests)
        requests.append(
            rest.QueryRequest(
                query=dense_vec,
                using=_DENSE_VECTOR_NAME,
                limit=pool,
                with_payload=True,
                # Cap the HNSW search beam at 64. Qdrant's default is
                # 128, which is fine for high-recall offline jobs but
                # ~2x the CPU on a hot path. 64 keeps p95 recall close
                # to 128 for top_k<=20 and cuts the per-search CPU
                # cost noticeably.
                params={"hnsw_ef": 64},
            )
        )
    if sparse_available:
        assert sparse_idx is not None and sparse_val is not None  # narrowed by sparse_available
        _sparse_req_idx = len(requests)
        requests.append(
            rest.QueryRequest(
                query=rest.SparseVector(indices=sparse_idx[0], values=sparse_val[0]),
                using=_SPARSE_VECTOR_NAME,
                limit=pool,
                with_payload=True,
                # No hnsw_ef for sparse — sparse uses inverted index, not HNSW.
            )
        )

    responses = await qdrant.client.query_batch_points(
        collection_name=collection_name,
        requests=requests,
    )
    dense_points = responses[_dense_req_idx].points if _dense_req_idx is not None else []
    sparse_points = responses[_sparse_req_idx].points if _sparse_req_idx is not None else []

    # 3. Raw scores per modality + first-seen payload per node.
    dense_raw: dict[str, float] = {str(p.id): float(p.score) for p in dense_points}
    sparse_raw: dict[str, float] = {str(p.id): float(p.score) for p in sparse_points}
    payloads: dict[str, dict] = {}
    for p in dense_points:
        payloads.setdefault(str(p.id), p.payload or {})
    for p in sparse_points:
        payloads.setdefault(str(p.id), p.payload or {})

    # C4: pool-level maxes over the FULL fetch_k candidate set, computed BEFORE
    # the top-k slice below. The NOT-FOUND gate must read these — a chunk with
    # the highest raw dense cosine can rank below the top-k by FUSED score (the
    # fused rank blends in normalized sparse) and get sliced off at step 5, so
    # the per-chunk max of the returned top-k can read artificially low and emit
    # a false NOT-FOUND. These are the true retrieval signals for the pool.
    pool_max_dense = max(dense_raw.values()) if dense_raw else 0.0
    pool_max_sparse = max(sparse_raw.values()) if sparse_raw else 0.0

    if not payloads:
        logger.debug(
            "Hybrid retrieval returned no candidates",
            query=query[:60],
            collection=collection_name,
        )
        return HybridSearchResult(
            chunks=[],
            pool_max_dense=0.0,
            pool_max_sparse=0.0,
            dense_available=dense_available,
            sparse_available=sparse_available,
        )

    # 4. Normalize each modality independently, then weighted-fuse. A node
    #    missing from one modality contributes 0 for that side.
    dense_norm = _minmax_normalize(dense_raw)
    sparse_norm = _minmax_normalize(sparse_raw)
    fused: dict[str, float] = {
        node_id: alpha * dense_norm.get(node_id, 0.0)
        + (1 - alpha) * sparse_norm.get(node_id, 0.0)
        for node_id in payloads
    }

    # 5. Rank by fused score; keep intermediate rerank pool if enabled, else keep top-k.
    intermediate_k = settings.rerank_top_k if settings.rerank_enabled else k
    ranked_ids = sorted(fused, key=lambda node_id: fused[node_id], reverse=True)[:intermediate_k]

    # 6. Materialize intermediate RetrievedChunk list. `node.metadata` carries the clean
    #    frontmatter dict (course_name, doc_type, etc.); `metadata` on the chunk
    #    excludes the promoted top-level fields, matching the prior contract.
    candidate_chunks: list[RetrievedChunk] = []
    _node_meta: dict[str, dict] = {}  # node_id → extracted metadata dict
    for node_id in ranked_ids:
        payload = payloads.get(node_id) or {}
        try:
            node = metadata_dict_to_node(payload)
            meta = node.metadata or {}
            text = node.text or ""  # type: ignore[attr-defined]  # TextNode at runtime
        except Exception:
            meta = payload
            text = payload.get("text", "")
        _node_meta[node_id] = meta
        extra_meta = {
            mk: mv
            for mk, mv in meta.items()
            if mk not in ("document_id", "source", "title", "chunk_index", "token_count")
        }
        candidate_chunks.append(
            RetrievedChunk(
                chunk_id=str(node_id),
                text=text,
                score=round(fused[node_id], 6),
                hybrid_score=round(fused[node_id], 6),
                dense_score=round(dense_raw.get(node_id, 0.0), 6),
                sparse_score=round(sparse_raw.get(node_id, 0.0), 6),
                document_id=meta.get("document_id", ""),
                source=meta.get("source", ""),
                title=meta.get("title", ""),
                chunk_index=int(meta.get("chunk_index", 0) or 0),
                token_count=int(meta.get("token_count", 0) or 0),
                metadata=extra_meta,
            )
        )

    # 6.0 Rerank candidates if enabled.
    if settings.rerank_enabled and len(candidate_chunks) > 0:
        try:
            rerank_results = await _call_openrouter_rerank(query, candidate_chunks, settings)
            ranked_chunks = []
            seen_indices = set()
            for orig_idx, score in rerank_results:
                if 0 <= orig_idx < len(candidate_chunks):
                    chunk = candidate_chunks[orig_idx]
                    chunk.score = round(score, 6)
                    ranked_chunks.append(chunk)
                    seen_indices.add(orig_idx)
            # Append any candidates not returned (fallback safety)
            for orig_idx, chunk in enumerate(candidate_chunks):
                if orig_idx not in seen_indices:
                    chunk.score = 0.0
                    ranked_chunks.append(chunk)
            candidate_chunks = ranked_chunks
            logger.info(
                f"Reranking successful via {settings.rerank_model}. "
                f"Top chunk score: {candidate_chunks[0].score if candidate_chunks else 0.0}"
            )
        except Exception as exc:
            logger.warning(
                f"Reranking failed: {exc}. Falling back to original Relative Score Fusion order."
            )
            # Fallback: candidate_chunks remains sorted by fused score.

    # Keep top-k final results
    chunks = candidate_chunks[:k]
    ranked_ids = [c.chunk_id for c in chunks]

    # 6.1 Hierarchical expansion: H3 ↔ H2 parent/children.
    #     Check each top-k chunk for parent/child metadata.
    #     If H3 → fetch its H2 parent + sibling H3s. If H2 → fetch H3 children.
    #     Dedup by chunk_id so the same chunk isn't included twice.
    #     Two-pass: first collect parent/child IDs, then after fetching,
    #     look up siblings from the fetched parent payloads.
    _already_ids = set(ranked_ids)  # IDs already materialized
    _hierarchy_fetch_ids: set[str] = set()
    _parents_needing_siblings: list[str] = []  # parent IDs whose siblings we need to find

    for node_id in ranked_ids:
        meta = _node_meta.get(node_id, {})
        # H3 chunk: has parent_chunk_id → fetch parent + siblings
        parent_id = meta.get("parent_chunk_id")
        if parent_id and parent_id not in _already_ids:
            _hierarchy_fetch_ids.add(parent_id)
            _parents_needing_siblings.append(parent_id)
        # H2 chunk: has child_chunk_ids → fetch all children
        for child_id in (meta.get("child_chunk_ids") or []):
            if child_id not in _already_ids:
                _hierarchy_fetch_ids.add(child_id)

    # Fetch expansion chunks not already in the payload pool (pass 1).
    _hierarchy_count = 0
    if _hierarchy_fetch_ids:
        try:
            _missing = [nid for nid in _hierarchy_fetch_ids if nid not in payloads]
            if _missing:
                _fetched = await qdrant.client.retrieve(
                    collection_name=collection_name,
                    ids=_missing,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in _fetched:
                    pid = str(point.id)
                    if pid not in payloads:
                        payloads[pid] = point.payload or {}
                        fused[pid] = 0.0
                        _hierarchy_count += 1

            # Pass 2: look up sibling H3s from fetched parent payloads.
            for parent_id in _parents_needing_siblings:
                if parent_id in _hierarchy_fetch_ids:
                    # Parent was fetched — read its child_chunk_ids from payload
                    try:
                        p_node = metadata_dict_to_node(payloads.get(parent_id) or {})
                        p_meta = p_node.metadata or {}
                    except Exception:
                        p_meta = payloads.get(parent_id) or {}
                    for sibling_id in (p_meta.get("child_chunk_ids") or []):
                        if sibling_id not in _already_ids and sibling_id not in _hierarchy_fetch_ids:
                            _hierarchy_fetch_ids.add(sibling_id)

            # Fetch any remaining sibling chunks (pass 2 fetch).
            _remaining = [nid for nid in _hierarchy_fetch_ids if nid not in payloads]
            if _remaining:
                _fetched2 = await qdrant.client.retrieve(
                    collection_name=collection_name,
                    ids=_remaining,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in _fetched2:
                    pid = str(point.id)
                    if pid not in payloads:
                        payloads[pid] = point.payload or {}
                        fused[pid] = 0.0
                        _hierarchy_count += 1

            # Materialize expansion chunks and append to result list.
            # ponytail: sibling/parent expansion noise filter — drop any
            # expansion chunk that scored ZERO on BOTH modalities in the
            # initial top-pool (i.e. neither dense nor sparse saw any
            # lexical/semantic overlap with the query). Such chunks are
            # topically distant from the user's question and only pad the
            # context window. Top-k chunks always have a fused score > 0,
            # so the original top-k keeps priority above the filter.
            _already_chunk_ids = {c.chunk_id for c in chunks}
            # Skip if either modality didn't run — no raw scores available.
            _can_filter_noise = dense_available and sparse_available
            
            # Sibling/parent noise thresholds to prevent weak candidates
            # from padding the context when fetch_k is large.
            DENSE_SIBLING_FLOOR = 0.60
            SPARSE_SIBLING_FLOOR = 5.0
            
            for exp_id in _hierarchy_fetch_ids:
                if exp_id in _already_chunk_ids:
                    continue  # already in result
                if _can_filter_noise:
                    d_raw = dense_raw.get(exp_id, 0.0)
                    s_raw = sparse_raw.get(exp_id, 0.0)
                    if d_raw < DENSE_SIBLING_FLOOR and s_raw < SPARSE_SIBLING_FLOOR:
                        continue  # too weak/noisy to be included
                payload = payloads.get(exp_id) or {}
                try:
                    node = metadata_dict_to_node(payload)
                    meta = node.metadata or {}
                    text = node.text or ""  # type: ignore[attr-defined]
                except Exception:
                    meta = payload
                    text = payload.get("text", "")
                extra_meta = {
                    mk: mv
                    for mk, mv in meta.items()
                    if mk not in ("document_id", "source", "title", "chunk_index", "token_count")
                }
                chunks.append(
                    RetrievedChunk(
                        chunk_id=str(exp_id),
                        text=text,
                        score=0.0,  # expansion chunks get neutral score
                        hybrid_score=0.0,
                        dense_score=round(dense_raw.get(exp_id, 0.0), 6),
                        sparse_score=sparse_raw.get(exp_id, 0.0),
                        document_id=meta.get("document_id", ""),
                        source=meta.get("source", ""),
                        title=meta.get("title", ""),
                        chunk_index=int(meta.get("chunk_index", 0) or 0),
                        token_count=int(meta.get("token_count", 0) or 0),
                        metadata=extra_meta,
                    )
                )
        except Exception as exc:
            logger.warning(f"Hierarchical expansion fetch failed: {exc}")

    logger.debug(
        "Hybrid retrieval complete",
        results=len(chunks),
        dense_hits=len(dense_points),
        sparse_hits=len(sparse_points),
        pool_max_dense=round(pool_max_dense, 4),
        pool_max_sparse=round(pool_max_sparse, 4),
        dense_available=dense_available,
        sparse_available=sparse_available,
        hierarchy_expanded=_hierarchy_count,
        query=query[:60],
        collection=collection_name,
    )
    return HybridSearchResult(
        chunks=chunks,
        pool_max_dense=round(pool_max_dense, 6),
        pool_max_sparse=round(pool_max_sparse, 6),
        dense_available=dense_available,
        sparse_available=sparse_available,
    )


async def hybrid_search_multi(
    queries: list[str],
    top_k: int | None = None,
    *,
    final_k_multiplier: int = 1,
    query_embeddings: list[list[float] | None] | None = None,
) -> HybridSearchResult:
    """Run one hybrid_search per sub-query in parallel, then merge + dedup.

    For a compound rewrite like ["kemampuan bayar mitra rendah",
    "skill negosiasi Business Partner", "fraud Business Partner"] the old path
    searched ONLY queries[0] — the other two topics were never retrieved, so a
    3-topic question got 5 chunks relevant to one topic only. This searches
    every sub-query, fuses results across queries, and returns
    `top_k * final_k_multiplier` (the COMPOUND_FINAL_TOP_K_MULTIPLIER) so a
    multi-faceted question gets multi-faceted context.

    Merge rule per chunk_id: keep max dense_score / sparse_score (absolute
    signals the NOT-FOUND gate reads), sum the fused hybrid_score across the
    sub-queries that surfaced it (a chunk relevant to 2 sub-queries outranks
    one relevant to 1), and carry the metadata of the highest-scoring hit.
    Pool-level maxes are the max across all sub-query pools; dense_available
    is OR (one sub-query's dense arm succeeding is enough to keep the gate
    out of the C5 sparse-only degraded window).

    Falls back to a single hybrid_search when only one query is passed — no
    merge overhead for the common single-query case.
    """
    if not queries:
        return HybridSearchResult()
    if len(queries) == 1:
        emb = query_embeddings[0] if query_embeddings else None
        return await hybrid_search(queries[0], top_k=top_k, query_embedding=emb)

    k = top_k or settings.retrieval_top_k
    embs = query_embeddings or [None] * len(queries)
    # ponytail: per-subquery top_k stays at final_top_k (not scaled) so each
    # query contributes its own top hits; the post-merge cut scales up.
    results = await asyncio.gather(
        *(
            hybrid_search(q, top_k=k, query_embedding=e)
            for q, e in zip(queries, embs)
        ),
        return_exceptions=True,
    )

    merged: dict[str, RetrievedChunk] = {}
    fused_accum: dict[str, float] = {}
    pool_max_dense = 0.0
    pool_max_sparse = 0.0
    dense_available = False
    sparse_available = False
    sub_ok = 0

    for res in results:
        if isinstance(res, BaseException) or res is None:
            continue
        sub_ok += 1
        dense_available = dense_available or res.dense_available
        sparse_available = sparse_available or res.sparse_available
        pool_max_dense = max(pool_max_dense, res.pool_max_dense)
        pool_max_sparse = max(pool_max_sparse, res.pool_max_sparse)
        for c in res.chunks:
            cid = c.chunk_id
            fused_accum[cid] = fused_accum.get(cid, 0.0) + c.hybrid_score
            prev = merged.get(cid)
            if prev is None:
                merged[cid] = c
            else:
                # keep the strongest absolute signals + richest metadata hit
                if c.dense_score > prev.dense_score:
                    prev.dense_score = c.dense_score
                if c.sparse_score > prev.sparse_score:
                    prev.sparse_score = c.sparse_score
                if c.hybrid_score > prev.hybrid_score:
                    prev.text = c.text or prev.text
                    prev.metadata = c.metadata or prev.metadata
                    prev.document_id = c.document_id or prev.document_id
                    prev.source = c.source or prev.source
                    prev.title = c.title or prev.title

    if sub_ok == 0 or not merged:
        return HybridSearchResult(
            chunks=[],
            pool_max_dense=round(pool_max_dense, 6),
            pool_max_sparse=round(pool_max_sparse, 6),
            dense_available=dense_available,
            sparse_available=sparse_available,
        )

    # Rank by accumulated fused score (a chunk hitting 2 sub-queries wins),
    # then cut to the scaled final_k.
    ranked = sorted(merged.values(), key=lambda c: fused_accum[c.chunk_id], reverse=True)
    final_k = k * max(1, final_k_multiplier) if final_k_multiplier > 1 else k
    chunks = ranked[:final_k]
    # Reflect the cross-query accumulated score on the returned chunks so
    # generate_node's source ordering and the dashboard read the merged rank.
    for c in chunks:
        c.score = round(fused_accum[c.chunk_id], 6)
        c.hybrid_score = c.score

    logger.debug(
        "Multi-query retrieval merged",
        sub_queries=len(queries),
        sub_ok=sub_ok,
        unique_chunks=len(merged),
        final_k=final_k,
        returned=len(chunks),
    )
    return HybridSearchResult(
        chunks=chunks,
        pool_max_dense=round(pool_max_dense, 6),
        pool_max_sparse=round(pool_max_sparse, 6),
        dense_available=dense_available,
        sparse_available=sparse_available,
    )
