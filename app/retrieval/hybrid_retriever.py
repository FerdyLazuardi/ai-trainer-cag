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
at ingestion (`Settings.embed_model`, text-embedding-3-small @1536) and the
sparse side uses the SAME fastembed BM25 encoder the vector store used to write
documents (`Qdrant/bm25`). This keeps query/document encodings aligned.
"""
import asyncio
from functools import lru_cache

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

# Vector names match QdrantManager._create_*_collection (dense + sparse).
_DENSE_VECTOR_NAME = "text-dense"
_SPARSE_VECTOR_NAME = "text-sparse"
# Must match `fastembed_sparse_model` passed to QdrantVectorStore at ingestion.
_SPARSE_MODEL = "Qdrant/bm25"


_sparse_semaphore: asyncio.Semaphore | None = None

def _get_sparse_semaphore() -> asyncio.Semaphore:
    global _sparse_semaphore
    if _sparse_semaphore is None:
        _sparse_semaphore = asyncio.Semaphore(1)
    return _sparse_semaphore


@lru_cache(maxsize=1)
def _get_sparse_encoder():
    """Load the fastembed BM25 sparse query encoder once (process lifetime).

    Same model/encoder the vector store used to write sparse vectors, so query
    and document term weighting are consistent.
    """
    return fastembed_sparse_encoder(model_name=_SPARSE_MODEL)


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
    """
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
                return await asyncio.wait_for(
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
    return None


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
        # Semaphore limits concurrency to 1, preventing fastembed (onnxruntime)
        # from saturating all CPU cores and starving the asyncio event loop.
        async with _get_sparse_semaphore():
            return await asyncio.to_thread(encoder, [query])

    # H5: reuse a precomputed embedding for the SAME query text instead of
    # re-embedding. The caller guarantees (by text equality) it matches `query`.
    async def _dense():
        if query_embedding is not None:
            return query_embedding
        return await _embed_query_resilient(query)

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
    requests = []
    _dense_req_idx = _sparse_req_idx = None
    if dense_available:
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
    dense_raw: dict[str, float] = {p.id: float(p.score) for p in dense_points}
    sparse_raw: dict[str, float] = {p.id: float(p.score) for p in sparse_points}
    payloads: dict[str, dict] = {}
    for p in dense_points:
        payloads.setdefault(p.id, p.payload)
    for p in sparse_points:
        payloads.setdefault(p.id, p.payload)

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

    # 5. Rank by fused score; keep top-k.
    ranked_ids = sorted(fused, key=lambda node_id: fused[node_id], reverse=True)[:k]

    # 6. Materialize RetrievedChunk. `node.metadata` carries the clean
    #    frontmatter dict (course_name, doc_type, etc.); `metadata` on the chunk
    #    excludes the promoted top-level fields, matching the prior contract.
    chunks: list[RetrievedChunk] = []
    for node_id in ranked_ids:
        payload = payloads.get(node_id) or {}
        try:
            node = metadata_dict_to_node(payload)
            meta = node.metadata or {}
            text = node.text or ""
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

    logger.debug(
        "Hybrid retrieval complete",
        results=len(chunks),
        dense_hits=len(dense_points),
        sparse_hits=len(sparse_points),
        pool_max_dense=round(pool_max_dense, 4),
        pool_max_sparse=round(pool_max_sparse, 4),
        dense_available=dense_available,
        sparse_available=sparse_available,
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
