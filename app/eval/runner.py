"""Offline eval runner — executes the RAG graph against a single query (or a
multi-turn conversation) and returns answer + retrieved_context + intent +
scores, ready for the judge.

Differs from the production /chat handler by skipping cache, LTM, user
preferences, and Redis history persistence — the goal is reproducible
regression testing, not session realism. Multi-turn conversations are
replayed in-process: prior turns build up the `messages` list so the
pre-processor sees the same history the production handler would.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger

from app.eval.judge import FaithfulnessResult, judge_faithfulness
from app.graph.pipeline import get_rag_graph


async def run_one_query(
    query: str | None = None,
    *,
    turns: list[str] | None = None,
    conversation_id: str = "eval-runner",
) -> dict[str, Any]:
    """Execute one query — or a multi-turn conversation — through the graph.

    Single-turn (legacy):
        await run_one_query("Apa itu Modal?")

    Multi-turn:
        await run_one_query(turns=[
            "tanya soal Client Protection bleh?",
            "minta penjelasann",
            "apalagi yang perlu aku ketahui",
        ])
        # Each prior turn is replayed; only the FINAL turn's result is
        # returned (and what the judge / golden checks evaluate).

    Returns a flat dict with the fields needed by the judge:
        - answer
        - retrieved_context (list of chunk dicts) from the FINAL turn
        - intent
        - intent_scores
        - rewritten_query
        - latency_ms (final turn only)
    """
    if turns is None:
        if query is None:
            raise ValueError("run_one_query requires either query or turns")
        turns = [query]
    if not turns:
        raise ValueError("turns must not be empty")

    rag_graph = get_rag_graph()

    messages: list[Any] = []
    final_result: dict[str, Any] | None = None
    final_latency_ms = 0.0

    for i, turn_text in enumerate(turns):
        messages.append(HumanMessage(content=turn_text))
        initial_state = {
            "messages": list(messages),
            "conversation_id": conversation_id,
            "conversation_summary": "",
            "user_profile": {"summary": "", "course_names": []},
            "user_preferences": None,
        }

        start = time.perf_counter()
        result = await rag_graph.ainvoke(
            initial_state,
            config={"run_name": f"ava-eval-offline-turn{i + 1}"},
        )
        latency_ms = (time.perf_counter() - start) * 1000

        final_msg = result["messages"][-1]
        ai_answer = (
            final_msg.content if hasattr(final_msg, "content") else str(final_msg)
        )
        messages.append(AIMessage(content=ai_answer))

        if i == len(turns) - 1:
            final_result = result
            final_latency_ms = latency_ms

    assert final_result is not None  # turns is non-empty so this is safe
    final = final_result["messages"][-1]
    answer = final.content if hasattr(final, "content") else str(final)

    return {
        "answer": answer,
        "retrieved_context": final_result.get("retrieved_context") or [],
        "intent": final_result.get("intent"),
        "intent_scores": final_result.get("intent_scores") or {},
        "rewritten_query": final_result.get("rewritten_query"),
        "retrieval_query": final_result.get("retrieval_query"),
        "safety_preserved_query": final_result.get("safety_preserved_query"),
        "latency_ms": round(final_latency_ms, 2),
    }


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    """Load JSONL — one record per line, schema documented in docs/eval.md."""
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.error(f"Bad JSON at line {line_no}: {e}")
    return items


async def evaluate_record(
    record: dict[str, Any],
    *,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    """Run one golden record end-to-end (query → graph → judge → checks).

    Records may be single-turn (`query`) or multi-turn (`turns: [...]`).
    For multi-turn records, only the FINAL turn's output is judged and
    checked — prior turns build up conversational state.
    """
    rid = record.get("id", "?")
    turns: list[str] | None = record.get("turns")
    query: str = record.get("query") or (turns[-1] if turns else "")
    expected_intent: str | None = record.get("expected_intent")
    must_include: list[str] = record.get("must_include") or []
    must_not_include: list[str] = record.get("must_not_include") or []
    min_faithfulness: float = float(record.get("min_faithfulness", 0.0))

    async with sem:
        try:
            run = await run_one_query(query=query if not turns else None, turns=turns)
        except Exception as e:
            logger.error(f"[{rid}] graph crashed: {e}")
            return {
                "id": rid,
                "passed": False,
                "reason": f"graph_crashed: {e}",
                "category": record.get("category"),
            }

        # Faithfulness only matters when the intent actually triggers KB retrieval.
        # Canned/ambiguous intents skip the judge.
        skip_judge = run["intent"] in {"GREETING", "AMBIGUOUS", "MALICIOUS", "TOPIC_LIST", "OFF_SCOPE"}
        judge: FaithfulnessResult | None = None
        if not skip_judge and min_faithfulness > 0:
            judge = await judge_faithfulness(
                query=query,
                answer=run["answer"],
                retrieved_context=run["retrieved_context"],
            )

    failures: list[str] = []

    if expected_intent and run["intent"] != expected_intent:
        failures.append(f"intent={run['intent']} != expected={expected_intent}")

    answer_lower = (run["answer"] or "").lower()
    for needle in must_include:
        if needle.lower() not in answer_lower:
            failures.append(f"missing_required_phrase: {needle!r}")
    for forbidden in must_not_include:
        if forbidden.lower() in answer_lower:
            failures.append(f"contains_forbidden_phrase: {forbidden!r}")

    # Mentor mode assertions — check learning_context range. MENTOR block
    # gating (lc >= threshold + no empathy + no safety + chunks) is tested
    # separately in tests/test_mentor_mode.py using replicated logic.
    expected_lc_min = record.get("expected_learning_context_min")
    expected_lc_max = record.get("expected_learning_context_max")
    actual_lc = float((run.get("intent_scores") or {}).get("learning_context", 0.0))
    if expected_lc_min is not None and actual_lc < expected_lc_min:
        failures.append(
            f"learning_context={actual_lc:.2f} < expected_min={expected_lc_min:.2f}"
        )
    if expected_lc_max is not None and actual_lc > expected_lc_max:
        failures.append(
            f"learning_context={actual_lc:.2f} > expected_max={expected_lc_max:.2f}"
        )

    if judge is not None and judge.score < min_faithfulness:
        failures.append(
            f"faithfulness={judge.score:.2f} < min={min_faithfulness:.2f} — "
            f"unsupported={judge.unsupported_claims[:2]}"
        )

    return {
        "id": rid,
        "category": record.get("category"),
        "query": query,
        "turns": turns,
        "intent": run["intent"],
        "intent_scores": run["intent_scores"],
        "latency_ms": run["latency_ms"],
        "answer_preview": (run["answer"] or "")[:200],
        "faithfulness": round(judge.score, 4) if judge else None,
        # Distinguish "judge ran but couldn't score" (None after C3's retry,
        # i.e. a judge-model outage / unparseable output) from "judge skipped"
        # (canned intent / min_faithfulness==0). A None here means the
        # faithfulness check was REQUESTED but produced no signal — it must NOT
        # be silently counted as a pass, or a judge outage would make every
        # graded turn pass while the mean is computed over a shrinking sample.
        "judge_no_signal": (not skip_judge) and min_faithfulness > 0 and judge is None,
        "unsupported_claims": judge.unsupported_claims if judge else [],
        "passed": not failures,
        "failures": failures,
    }


async def evaluate_dataset(
    records: list[dict[str, Any]],
    *,
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """Evaluate all records with bounded concurrency. Returns per-record results."""
    sem = asyncio.Semaphore(concurrency)
    tasks = [evaluate_record(r, sem=sem) for r in records]
    return await asyncio.gather(*tasks)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize pass/fail counts and per-category faithfulness means."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    # Count turns where the judge was requested but returned no signal (None
    # after C3's retry — a judge-model outage or unparseable output). Surfaced
    # so "no signal" is never silently equivalent to "faithful": a spike here
    # means the faithfulness mean is computed over a shrinking, unreliable sample.
    judge_no_signal = sum(1 for r in results if r.get("judge_no_signal"))
    by_cat: dict[str, dict[str, Any]] = {}

    for r in results:
        cat = r.get("category") or "uncategorized"
        slot = by_cat.setdefault(
            cat,
            {"total": 0, "passed": 0, "faithfulness_sum": 0.0, "faithfulness_n": 0, "judge_no_signal": 0},
        )
        slot["total"] += 1
        if r["passed"]:
            slot["passed"] += 1
        if r.get("judge_no_signal"):
            slot["judge_no_signal"] += 1
        if r.get("faithfulness") is not None:
            slot["faithfulness_sum"] += float(r["faithfulness"])
            slot["faithfulness_n"] += 1

    cat_summary = {}
    for cat, s in by_cat.items():
        mean = (
            round(s["faithfulness_sum"] / s["faithfulness_n"], 4)
            if s["faithfulness_n"]
            else None
        )
        cat_summary[cat] = {
            "total": s["total"],
            "passed": s["passed"],
            "pass_rate": round(s["passed"] / s["total"], 4) if s["total"] else 0.0,
            "faithfulness_mean": mean,
            "judge_no_signal": s["judge_no_signal"],
        }

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "judge_no_signal": judge_no_signal,
        "by_category": cat_summary,
    }
