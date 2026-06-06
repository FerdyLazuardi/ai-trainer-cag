"""Mentor-mode eval driver.

Loads `tests/eval/mentor_mode_cases.json` and runs the offline eval runner
against the live RAG graph. Prints per-case results + a category aggregate
suitable for human review.

Run:  python -m app.eval.run_mentor_eval
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from loguru import logger

from app.eval.runner import aggregate, evaluate_dataset


CASES_PATH = Path("tests/eval/mentor_mode_cases.json")


def load_cases() -> list[dict]:
    with CASES_PATH.open("r", encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list):
        raise ValueError(f"Expected JSON array, got {type(cases).__name__}")
    return cases


async def main() -> int:
    cases = load_cases()
    logger.info(f"Loaded {len(cases)} mentor-mode cases from {CASES_PATH}")

    started = time.perf_counter()
    results = await evaluate_dataset(cases, concurrency=4)
    elapsed = time.perf_counter() - started

    print()
    print("=" * 90)
    print(f"MENTOR MODE EVAL — {len(cases)} cases, {elapsed:.1f}s wall, concurrency=4")
    print("=" * 90)

    for r in results:
        rid = r.get("id", "?")
        cat = r.get("category", "?")
        passed = r.get("passed", False)
        intent = r.get("intent")
        scores = r.get("intent_scores") or {}
        lc = float(scores.get("learning_context", 0.0))
        q = r.get("query") or (r.get("turns") or ["?"])[-1]
        if isinstance(q, str):
            qshort = q[:60] + ("..." if len(q) > 60 else "")
        else:
            qshort = str(q)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {rid:11s}  intent={intent:9s}  L={lc:.2f}  | {qshort}")
        for fail in r.get("failures", []):
            print(f"           -> {fail}")
        if not passed and "answer_preview" in r:
            preview = r["answer_preview"][:120].replace("\n", " ")
            print(f"           preview: {preview}...")

    print()
    print("-" * 90)
    summary = aggregate(results)
    print(
        f"Total: {summary['total']} | "
        f"Passed: {summary['passed']} | "
        f"Failed: {summary['failed']} | "
        f"Pass rate: {summary['pass_rate']:.1%}"
    )
    if summary.get("by_category"):
        for cat, s in summary["by_category"].items():
            print(
                f"  {cat}: {s['passed']}/{s['total']} "
                f"({s['pass_rate']:.1%}) "
                f"faithfulness_mean={s.get('faithfulness_mean')}"
            )
    print("=" * 90)
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
