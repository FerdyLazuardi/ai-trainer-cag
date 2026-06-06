"""CLI runner for JSON-array eval datasets (not JSONL).

Usage:
    python -m app.eval.run_dataset tests/eval/mentor_mode_cases.json
    python -m app.eval.run_dataset tests/eval/mentor_mode_cases.json --concurrency 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from loguru import logger

from app.eval.runner import aggregate, evaluate_dataset


async def main(path: Path, concurrency: int) -> int:
    raw = path.read_text(encoding="utf-8")
    records = json.loads(raw)
    if not isinstance(records, list):
        logger.error(f"Expected JSON array, got {type(records).__name__}")
        return 2

    logger.info(f"Loaded {len(records)} records from {path}")
    start = time.perf_counter()
    results = await evaluate_dataset(records, concurrency=concurrency)
    elapsed = round(time.perf_counter() - start, 2)

    summary = aggregate(results)

    print("\n" + "=" * 72)
    print(f" EVAL SUMMARY — {path.name}")
    print("=" * 72)
    print(f"  total       : {summary['total']}")
    print(f"  passed      : {summary['passed']}")
    print(f"  failed      : {summary['failed']}")
    print(f"  pass_rate   : {summary['pass_rate']:.2%}")
    print(f"  elapsed     : {elapsed}s")
    print("-" * 72)
    for cat, s in sorted(summary["by_category"].items()):
        fm = f"{s['faithfulness_mean']:.2f}" if s["faithfulness_mean"] is not None else "—"
        print(
            f"  {cat:<20s} {s['passed']:>3d}/{s['total']:<3d}  "
            f"({s['pass_rate']:.0%})  faithfulness_mean={fm}"
        )
    print("=" * 72)

    # Per-record failure detail (only failures — saves noise)
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\n FAILED RECORDS ({len(failures)}):")
        for r in failures:
            print(f"  - id={r['id']} intent={r.get('intent')} "
                  f"failures={r.get('failures', [])}")
            lc = (r.get("intent_scores") or {}).get("learning_context")
            if lc is not None:
                print(f"      learning_context={lc:.2f}")
            print(f"      answer_preview: {(r.get('answer_preview') or '')[:120]}...")
    print()
    return 0 if summary["failed"] == 0 else 1


def cli() -> int:
    p = argparse.ArgumentParser(description="Run a JSON eval dataset")
    p.add_argument("path", type=Path, help="Path to JSON array of records")
    p.add_argument("--concurrency", type=int, default=2, help="Parallelism (default 2)")
    args = p.parse_args()
    return asyncio.run(main(args.path, args.concurrency))


if __name__ == "__main__":
    sys.exit(cli())
