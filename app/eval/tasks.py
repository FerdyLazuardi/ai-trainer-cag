"""arq background task for post-hoc turn evaluation.

Enqueued from chat handlers after the response has been streamed/returned
to the user. Runs the LLM-as-judge evaluator and writes scores to Phoenix
as span annotations on the original turn's root span.

Cost control:
- Sampling decided at enqueue site (chat.py) — task only runs when picked.
- Judge uses `get_cheap_llm` (Gemini Flash Lite), not the user-facing model.
- Truncated context in the judge prompt (see app/eval/judge.py).
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from app.eval.judge import judge_faithfulness
from app.observability import annotate_span


async def eval_turn_task(
    ctx: dict,
    span_id: str | None,
    query: str,
    answer: str,
    retrieved_context: list[dict[str, Any]],
    intent: str | None = None,
    intent_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate one turn post-hoc and annotate Phoenix with quality scores.

    Returns a small status dict for arq logs. Errors are caught and logged
    so a bad eval never propagates back to user-facing flow.
    """
    if not span_id:
        return {"status": "skipped", "reason": "no_span_id"}
    if not answer or not answer.strip():
        return {"status": "skipped", "reason": "empty_answer"}

    try:
        result = await judge_faithfulness(
            query=query,
            answer=answer,
            retrieved_context=retrieved_context or [],
        )
    except Exception as exc:
        logger.warning(f"eval_turn_task: judge crashed: {exc}")
        return {"status": "error", "reason": "judge_crashed"}

    if result is None:
        return {"status": "skipped", "reason": "judge_no_signal"}

    try:
        annotate_span(span_id, "eval_faithfulness", round(result.score, 4))
    except Exception as exc:
        logger.warning(f"eval_turn_task: Phoenix annotate failed: {exc}")
        return {"status": "error", "reason": "annotate_failed"}

    if result.score < 0.5:
        logger.warning(
            "Low faithfulness score detected",
            span_id=span_id,
            score=round(result.score, 3),
            unsupported=result.unsupported_claims[:2],
            reasoning=result.reasoning[:120],
            query=query[:80],
            intent=intent,
        )
    else:
        logger.debug(
            "Faithfulness eval done",
            span_id=span_id,
            score=round(result.score, 3),
            intent=intent,
        )

    return {
        "status": "annotated",
        "span_id": span_id,
        "score": round(result.score, 4),
    }
