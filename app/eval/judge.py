"""Faithfulness judge for Ava answers.

Uses a DEDICATED judge model (`get_judge_llm` = deepseek/deepseek-v4-pro) —
a different model family than BOTH the generator (`get_generate_llm`) and the
old judge slot (`get_llm`), which are both Gemini 2.5 Flash Lite. Output is
in [0, 1]:

    1.0  every claim in the answer is supported by the context
    0.5  partially grounded (some claims unsupported but plausible)
    0.0  hallucination — claims contradict or invent beyond the context

Why a dedicated model, distinct from the generator? When judge and generator
share a model family they share fabrication patterns, and the eval
systematically undercounts the ungrounded rate (the judge effectively grades
its own output). The previous code used `get_llm` (llm_model) for the judge
and `get_generate_llm` (cheap_llm_model) for generation — but BOTH resolve to
Gemini 2.5 Flash Lite, so it was Flash-Lite grading Flash-Lite (C3). Pinning
the judge to DeepSeek V4 Pro (native provider, no fallbacks, reasoning off)
breaks that shared-family bias. The boot guard
`assert_judge_model_distinct` enforces judge != generator at startup; the
50-query calibration (D3) validates V4 Pro correlation vs gold-grade before
the judge threshold is locked.

Output contract: the judge runs with response_format=json_object (NOT
tool-calling structured output), so it returns a raw JSON string in
AIMessage.content. We json.loads + FaithfulnessResult.model_validate it
here. A reasoning model can occasionally emit an empty content string
(everything routed to reasoning_content, which ChatOpenAI drops) — we retry
once on empty/invalid content, then return None ("no signal") rather than
infer a default score that would skew the mean.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from app.llm.client import get_judge_llm


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

Score the answer's faithfulness to the context. Respond with ONLY a JSON
object (no prose, no markdown fences) matching this schema exactly:
{{"score": <float 0..1>, "unsupported_claims": [<string>, ...], "reasoning": "<one short sentence, max 30 words>"}}"""


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


def _coerce_content_to_text(content: Any) -> str:
    """Flatten an AIMessage.content (str | list[block]) to plain text.

    OpenRouter/ChatOpenAI may return content as a list of content blocks.
    We only want the text payload to json.loads. Returns "" when nothing
    usable is present (the empty-content signal the retry loop checks).
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text") or block.get("content")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts).strip()
    return ""


def _parse_judge_json(raw: str) -> FaithfulnessResult | None:
    """Parse the judge's JSON string into a FaithfulnessResult.

    Tolerates a stray ```json fence (some models still wrap json_object
    output). Returns None on empty/invalid/schema-mismatch so the caller's
    retry loop can fire, and ultimately surfaces "no signal" rather than a
    fabricated default score.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        # strip a leading ```json / ``` fence and trailing ```
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return FaithfulnessResult.model_validate(data)
    except ValidationError:
        return None


async def judge_faithfulness(
    *,
    query: str,
    answer: str,
    retrieved_context: list[dict[str, Any]],
) -> FaithfulnessResult | None:
    """Score the faithfulness of `answer` against `retrieved_context`.

    Returns None on judge failure — caller should treat as "no signal" rather
    than infer a default score.

    Uses the dedicated DeepSeek V4 Pro judge with response_format=json_object
    (not tool-calling structured output). Retries ONCE when the model returns
    empty content (reasoning leaked to a dropped field) or unparseable JSON,
    appending the bad reply + a JSON-only correction so the retry is a true
    repair, not a blind re-roll.
    """
    if not answer or not answer.strip():
        return None

    prompt = _JUDGE_PROMPT.format(
        query=query.strip()[:1000],
        context=_format_context(retrieved_context),
        answer=answer.strip()[:3000],
    )

    try:
        judge_llm = get_judge_llm()
        messages: list[Any] = [HumanMessage(content=prompt)]

        for attempt in range(2):
            reply = await judge_llm.ainvoke(
                messages,
                config={"run_name": "ava-eval-faithfulness"},
            )
            raw = _coerce_content_to_text(getattr(reply, "content", ""))
            parsed = _parse_judge_json(raw)
            if parsed is not None:
                return parsed

            # First failure → append the bad reply + a JSON-only correction
            # and retry once. Empty content (reasoning leaked) and malformed
            # JSON both land here.
            if attempt == 0:
                logger.warning(
                    "Faithfulness judge returned empty/invalid content; retrying once",
                    empty=not raw,
                    preview=raw[:120],
                )
                messages = messages + [
                    AIMessage(content=raw or "(empty)"),
                    HumanMessage(
                        content=(
                            "Your previous reply was not valid JSON. Respond with "
                            "ONLY a JSON object matching: "
                            '{"score": <float 0..1>, "unsupported_claims": '
                            '[<string>...], "reasoning": "<one short sentence>"}. '
                            "No prose, no markdown fences."
                        )
                    ),
                ]

        logger.warning("Faithfulness judge failed to produce valid JSON after retry")
        return None
    except Exception as exc:
        logger.warning(f"Faithfulness judge failed: {exc}")
        return None
