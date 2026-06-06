"""Faithfulness judge for A-Pedi answers.

Uses the MAIN LLM (`get_llm`) — NOT the cheap LLM the generator uses — to
score whether the answer is grounded in the retrieved context. Output is
in [0, 1]:

    1.0  every claim in the answer is supported by the context
    0.5  partially grounded (some claims unsupported but plausible)
    0.0  hallucination — claims contradict or invent beyond the context

Why a different model than the generator? When judge and generator share
a model they share fabrication patterns, and the eval systematically
undercounts the ungrounded rate. (Was `get_cheap_llm` previously — that
was the bug: same Gemini 2.5 Flash Lite family on both sides meant the
judge was effectively the generator grading itself.) Trade-off: when the
main LLM is swapped (Sonnet → Flash → Flash-Lite) the judge baseline
shifts too, so week-over-week scores need a calibration note. Generator
can still swap freely; the judge baseline only shifts on the main LLM
side, not the cheap side.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from loguru import logger
from pydantic import BaseModel, Field


# Cap context shown to the judge — long context inflates judge tokens
# without improving faithfulness signal (judge only needs enough to verify
# claim support, not full documents).
_MAX_CONTEXT_CHARS = 4000
# 12 chunks covers the production final_top_k (8-10) + 2-4 buffer. The
# previous 6 was below the chunk count the generator saw, so the judge
# literally could not verify any claim grounded in chunk 7+ — directly
# masking the ungrounded rate. Per-chunk budget is still capped at
# _MAX_CONTEXT_CHARS / 12 (~333 chars) by _format_context.
_MAX_CHUNKS = 12


class FaithfulnessResult(BaseModel):
    """Structured output schema for the faithfulness judge."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Faithfulness score in [0, 1]. 1.0 = every factual claim in the "
            "answer is supported by the context. 0.0 = answer hallucinates "
            "facts not in context. 0.5 = partially grounded."
        ),
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 3 short bullets quoting claims from the answer that are "
            "NOT supported by the context. Empty if score >= 0.9."
        ),
    )
    reasoning: str = Field(
        default="",
        description="One short sentence explaining the score. Max 30 words.",
    )


_JUDGE_PROMPT = """You are evaluating whether an AI answer is grounded in retrieved context.

TASK: Score how faithful the answer is to the context. Score in [0, 1]:

- 1.0 — every factual claim in the answer is directly supported by the context
- 0.7-0.9 — mostly grounded, minor paraphrasing or implicit reasoning OK
- 0.4-0.6 — partially grounded; some claims unsupported but plausible
- 0.1-0.3 — answer invents facts not in context (names, numbers, products)
- 0.0 — answer contradicts the context or hallucinates entirely

IMPORTANT:
- Empathy/acknowledgment phrases ("understandable that you feel..."), polite
  framing, and meta-instructions to the user are NOT facts and should not
  count against faithfulness.
- Refusals or "I don't have that info" responses score 1.0 (no claim made).
- General world knowledge that isn't from context can lower score IF the
  answer presents it as if it came from Amartha's KB.

USER QUERY:
{query}

RETRIEVED CONTEXT:
{context}

AI ANSWER:
{answer}

Score the answer's faithfulness to the context."""


def _format_context(retrieved: list[dict[str, Any]]) -> str:
    """Truncate + flatten retrieved chunks into a compact string for the judge.

    Allocates the per-chunk char budget based on the ACTUAL number of chunks
    (not _MAX_CHUNKS), and shrinks proportionally only when the total exceeds
    _MAX_CONTEXT_CHARS. Earlier code divided by _MAX_CHUNKS even when fewer
    chunks were present, which truncated each chunk to ~666 chars and hid
    grounding sentences that lived in the chunk's tail (the ground truth
    detail in a Markdown KB usually sits AFTER the heading + intro).
    """
    if not retrieved:
        return "(no context retrieved)"

    pool = retrieved[:_MAX_CHUNKS]
    n = len(pool)
    per_chunk_budget = max(400, _MAX_CONTEXT_CHARS // n)

    parts: list[str] = []
    used = 0
    for i, c in enumerate(pool):
        text = (c.get("text") or c.get("content") or "").strip()
        if not text:
            continue
        remain = _MAX_CONTEXT_CHARS - used
        if remain <= 200:
            break
        cap = min(per_chunk_budget, remain)
        snippet = text[:cap]
        source = c.get("source") or c.get("course_name") or "?"
        parts.append(f"[chunk {i + 1} | {source}]\n{snippet}")
        used += len(snippet)
    return "\n\n".join(parts) if parts else "(no usable context)"


async def judge_faithfulness(
    *,
    query: str,
    answer: str,
    retrieved_context: list[dict[str, Any]],
) -> FaithfulnessResult | None:
    """Score the faithfulness of `answer` against `retrieved_context`.

    Returns None on judge failure — caller should treat as "no signal" rather
    than infer a default score.
    """
    from app.llm.client import get_llm

    if not answer or not answer.strip():
        return None

    prompt = _JUDGE_PROMPT.format(
        query=query.strip()[:1000],
        context=_format_context(retrieved_context),
        answer=answer.strip()[:3000],
    )

    try:
        judge_llm = get_llm()
        structured = judge_llm.with_structured_output(FaithfulnessResult)
        result = await structured.ainvoke(
            [HumanMessage(content=prompt)],
            config={"run_name": "a-pedi-eval-faithfulness"},
        )
        return result
    except Exception as exc:
        logger.warning(f"Faithfulness judge failed: {exc}")
        return None
