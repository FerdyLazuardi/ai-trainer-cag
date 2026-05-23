"""
Post-generation follow-up validator.

For each LLM-suggested follow-up question, embed it and run a top-1 search
against the Knowledge_Base Qdrant collection. Drop questions whose top-1
similarity falls below the configured threshold — those would dead-end the
user with "Aku belum menemukan info" because retrieval can't find a useful
chunk.

Fail-open: any embedding/Qdrant error returns the input unchanged so the
validator never blocks a response.
"""
import asyncio

from loguru import logger
from qdrant_client.models import Filter

from app.config.settings import get_settings
from app.config.embedding_config import ensure_llamaindex_configured
from app.database.qdrant_client import get_qdrant_client


_settings = get_settings()


async def _embed_questions(questions: list[str]) -> list[list[float]] | None:
    """Embed all candidate questions in one batched call. Returns None on failure."""
    try:
        ensure_llamaindex_configured()
        from llama_index.core import Settings as LISettings
        embed_model = LISettings.embed_model

        # OpenAI embedding model exposes async batch API
        if hasattr(embed_model, "aget_text_embedding_batch"):
            return await embed_model.aget_text_embedding_batch(questions)

        # Fallback: parallel single-call embeddings
        return await asyncio.gather(*[embed_model.aget_text_embedding(q) for q in questions])
    except Exception as exc:
        logger.warning("Follow-up validator: batch embedding failed", error=str(exc))
        return None


async def _qdrant_top1_score(vector: list[float]) -> float | None:
    """Run Qdrant KB top-1 query and return score. None on failure."""
    try:
        qdrant = get_qdrant_client()
        result = await qdrant.client.query_points(
            collection_name=_settings.qdrant_kb_collection,
            query=vector,
            using="text-dense",   # KB collection uses named hybrid vectors
            limit=1,
            with_payload=False,
            query_filter=Filter(must=[]),
        )
        if not result.points:
            return 0.0
        return float(result.points[0].score or 0.0)
    except Exception as exc:
        logger.warning("Follow-up validator: Qdrant query failed", error=str(exc))
        return None


async def validate_followups(
    questions: list[str],
    threshold: float | None = None,
) -> list[str]:
    """Drop follow-ups whose top-1 KB similarity falls below threshold.

    Order is preserved. On any failure, returns the input unchanged
    (fail-open — validator is a UX guard, not safety-critical).
    """
    if not _settings.followup_validation_enabled:
        return questions
    if not questions:
        return []

    # Strip whitespace + drop empties just in case
    cleaned = [q.strip() for q in questions if q and q.strip()]
    if not cleaned:
        return []

    embeddings = await _embed_questions(cleaned)
    if embeddings is None or len(embeddings) != len(cleaned):
        return questions   # fail-open

    thr = threshold if threshold is not None else _settings.followup_validation_threshold
    scores = await asyncio.gather(*[_qdrant_top1_score(v) for v in embeddings])

    kept: list[str] = []
    dropped: list[tuple[str, float]] = []
    for q, score in zip(cleaned, scores):
        if score is None:
            kept.append(q)   # fail-open per-question
            continue
        if score >= thr:
            kept.append(q)
        else:
            dropped.append((q, score))

    if dropped:
        logger.info(
            "Follow-up validator dropped low-similarity questions",
            dropped_count=len(dropped),
            kept_count=len(kept),
            threshold=thr,
            dropped=[(q[:60], round(s, 3)) for q, s in dropped],
        )

    return kept


def render_followup_block(questions: list[str]) -> str:
    """Render validated questions as a numbered Penasaran tentang block.

    Returns empty string if no questions — caller should omit the block entirely.
    """
    if not questions:
        return ""
    lines = ["**Penasaran tentang:**"]
    for i, q in enumerate(questions, 1):
        lines.append(f"{i}. {q}")
    return "\n".join(lines)
