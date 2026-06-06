"""arq background task for post-hoc turn evaluation.

Enqueued from chat handlers after the response has been streamed/returned
to the user. Runs the LLM-as-judge evaluator and writes the score to
Postgres (agent_logs) keyed by turn_id so the Streamlit dashboard can
read it.

Cost control:
- Sampling decided at enqueue site (chat.py) — task only runs when picked.
- Judge uses `get_cheap_llm` (Gemini Flash Lite), not the user-facing model.
- Truncated context in the judge prompt (see app/eval/judge.py).
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from app.eval.judge import judge_faithfulness


async def eval_turn_task(
    query: str,
    answer: str,
    retrieved_context: list[dict[str, Any]],
    intent: str | None = None,
    intent_scores: dict[str, float] | None = None,
    turn_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate one turn post-hoc and persist the faithfulness score.

    Writes the score to Postgres (agent_logs, keyed by turn_id) so the
    Streamlit monitoring dashboard can surface low-faithfulness turns.
    Errors are caught and logged so a bad eval never propagates back to
    user-facing flow.
    """
    if not turn_id:
        return {"status": "skipped", "reason": "no_correlation_key"}
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

    score = round(result.score, 4)

    persisted = await _persist_faithfulness(turn_id, score)

    if result.score < 0.5:
        logger.warning(
            "Low faithfulness score detected",
            turn_id=turn_id,
            score=round(result.score, 3),
            unsupported=result.unsupported_claims[:2],
            reasoning=result.reasoning[:120],
            query=query[:80],
            intent=intent,
        )
    else:
        logger.debug(
            "Faithfulness eval done",
            turn_id=turn_id,
            score=round(result.score, 3),
            intent=intent,
        )

    return {
        "status": "annotated",
        "turn_id": turn_id,
        "score": score,
        "persisted": persisted,
    }


async def _persist_faithfulness(turn_id: str, score: float) -> bool:
    """UPDATE the agent_logs row for this turn with the faithfulness score.

    The eval runs async (after the turn was logged), but the agent_logs write
    is itself batched/deferred — so the row may not exist yet when this fires.
    Retry a couple of times with backoff to absorb that race. Best-effort:
    returns False on persistent miss without raising.
    """
    import asyncio

    from sqlalchemy import update

    from app.database.postgres import AsyncSessionLocal
    from app.database.models import AgentLog

    for attempt in range(3):
        try:
            async with AsyncSessionLocal() as session:
                res = await session.execute(
                    update(AgentLog)
                    .where(AgentLog.turn_id == turn_id)
                    .values(faithfulness_score=score)
                )
                await session.commit()
                if res.rowcount and res.rowcount > 0:
                    return True
        except Exception as exc:
            logger.warning(f"eval_turn_task: faithfulness persist failed (attempt {attempt + 1}): {exc}")
        # Row not flushed yet (batch logger interval) — wait and retry.
        await asyncio.sleep(2 * (attempt + 1))

    logger.warning("eval_turn_task: agent_logs row not found for turn_id", turn_id=turn_id)
    return False
