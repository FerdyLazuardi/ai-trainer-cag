"""
Local cross-encoder reranker (replaces Cohere rerank-multilingual-v3.0).

Uses sentence-transformers' CrossEncoder. The model is loaded once via
`lru_cache`; sync `predict()` is wrapped in `asyncio.to_thread` to keep the
existing async API contract untouched.

Defaults to `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` — small (120M)
multilingual cross-encoder that fits a dual-core CPU budget while still
giving a meaningful signal for top-15 → top-K refinement on top of hybrid
BM25 + dense retrieval.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache

from loguru import logger

from app.config.settings import get_settings
from app.retrieval.schemas import RetrievedChunk

settings = get_settings()


@lru_cache(maxsize=1)
def _get_cross_encoder():
    """Load the cross-encoder model once. Cached for the process lifetime."""
    from sentence_transformers import CrossEncoder

    logger.info(
        "Loading cross-encoder reranker",
        model=settings.reranker_model_name,
        device=settings.reranker_device,
    )
    return CrossEncoder(
        settings.reranker_model_name,
        max_length=settings.reranker_max_length,
        device=settings.reranker_device,
    )


def warmup_reranker() -> None:
    """Pre-load model + run a tiny predict so the first request avoids cold start."""
    try:
        model = _get_cross_encoder()
        model.predict([("warmup query", "warmup document")])
        logger.info("Reranker warmed up", model=settings.reranker_model_name)
    except Exception as exc:
        logger.warning(f"Reranker warmup failed (will retry on first request): {exc}")


def _predict_scores(model, query: str, texts: list[str]) -> list[float]:
    """Sync helper run inside a thread; returns one relevance score per text.

    The model outputs raw logits — apply sigmoid so callers get a probability
    in [0, 1], matching the contract Cohere had.
    """
    import math

    pairs = [(query, t) for t in texts]
    raw = model.predict(pairs)
    return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
    deduplicate: bool = True,
) -> list[RetrievedChunk]:
    """
    Rerank retrieved chunks with a local cross-encoder.

    Steps:
        1. Sort by descending initial (hybrid) score.
        2. Score each (query, chunk.text) pair with the cross-encoder.
        3. Replace `chunk.score` with the cross-encoder relevance score
           (`hybrid_score` is preserved on the chunk for downstream metrics).
        4. Return top-K sorted by the new score.

    Args:
        query: User query (already rewritten by pre_processor).
        chunks: Output of hybrid_search.
        top_k: Max results to return (default: settings.reranked_top_k).
        deduplicate: Kept for backwards-compat — unused (Cohere handled this
            implicitly; mMiniLMv2 doesn't, and dedup belongs upstream anyway).

    Returns:
        Reranked list[RetrievedChunk] of length ≤ top_k. When
        `settings.reranker_enabled` is False, returns top-K of the input
        sorted by hybrid_score (no cross-encoder pass).
    """
    k = top_k or settings.reranked_top_k

    if not chunks:
        return []

    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

    # Bypass mode: skip the cross-encoder entirely. `chunk.score` already
    # equals `hybrid_score` (set by hybrid_search), so the existing sort is
    # the right ordering. Used by the eval harness to A/B reranker on/off.
    if not settings.reranker_enabled:
        logger.debug(
            "Reranker disabled — passing through hybrid order",
            input_chunks=len(chunks),
            output_chunks=min(len(sorted_chunks), k),
            top_k=k,
        )
        return sorted_chunks[:k]

    try:
        model = _get_cross_encoder()
        texts = [c.text for c in sorted_chunks]
        scores = await asyncio.to_thread(_predict_scores, model, query, texts)

        for chunk, score in zip(sorted_chunks, scores):
            # hybrid_score is set by the retriever — keep intact for metrics.
            chunk.score = score

        reranked = sorted(sorted_chunks, key=lambda c: c.score, reverse=True)[:k]

        logger.debug(
            "Local reranking complete",
            input_chunks=len(chunks),
            output_chunks=len(reranked),
            top_k=k,
            model=settings.reranker_model_name,
        )
        return reranked
    except Exception as exc:
        logger.warning(f"Local reranking failed, falling back to hybrid order: {exc}")
        return sorted_chunks[:k]
