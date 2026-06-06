r"""D3 — Judge calibration harness (deepseek/deepseek-v4-pro faithfulness judge).

PURPOSE
After C3 wired a DEDICATED judge model (DeepSeek V4 Pro, distinct from the
Gemini generator), we must VALIDATE that the judge's faithfulness scores
actually track human judgment before we trust them — and LOCK the judge
threshold (provisional 0.70, settings.faithfulness_min / eval gating) on
evidence rather than a guess.

This harness runs a fixed calibration set of ~50 queries through the LIVE RAG
graph, collects the judge's faithfulness score for each, and compares those
scores against HUMAN gold labels you provide in the dataset. It then reports:

  - correlation (Pearson + Spearman) between judge score and gold score
  - at each candidate threshold τ, the confusion matrix (judge says
    faithful/unfaithful vs gold), plus precision/recall/F1 and Cohen's κ
  - the τ that maximizes agreement with the gold labels

DECISION RULE (from the plan):
  - If V4 Pro correlates well with gold (Spearman ≳ 0.6 and best-τ κ ≳ 0.6),
    LOCK the threshold at the reported best τ (≈ 0.70 expected).
  - If correlation is poor, the judge model is not trustworthy → SWAP to the
    fallback (qwen/qwen-2.5-72b-instruct, settings.judge_llm_fallback_model)
    and re-run this harness.

⚠️ THIS IS A LIVE RUN — NOT A UNIT TEST. It:
  - calls the real DeepSeek judge API (costs tokens, needs OPENROUTER creds)
  - calls the real embedding + generator + Qdrant (needs the live KB)
  - requires human `gold_faithful` labels in the dataset (see the seed file)
Do NOT add it to the pytest suite. Run it manually:

    .\.venv\Scripts\python.exe -m app.eval.calibrate_judge \
        data/eval/judge_calibration_seed.json --concurrency 2

DATASET SCHEMA (JSON array; one object per calibration item):
    {
      "id": "cal-001",
      "query": "Apa itu Modal Kerja?",          // or "turns": ["...","..."]
      "gold_faithful": 1.0,                       // HUMAN label in [0,1]:
                                                  //   1.0 fully grounded answer
                                                  //   0.5 partially grounded
                                                  //   0.0 hallucinated
      "note": "optional human rationale"
    }
Items missing `gold_faithful` are RUN (judge score recorded) but EXCLUDED from
correlation/threshold math, and flagged in the report so you can label them.
Items whose intent skips the judge (GREETING/AMBIGUOUS/…/refusals) record a
None judge score and are likewise excluded from the correlation math.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

from app.config.settings import get_settings
from app.eval.judge import judge_faithfulness
from app.eval.runner import run_one_query

# Intents whose answers carry no faithfulness signal — judge is skipped, so
# these can't contribute to calibration (mirrors runner.evaluate_record).
_SKIP_JUDGE_INTENTS = {"GREETING", "AMBIGUOUS", "MALICIOUS", "TOPIC_LIST", "OFF_SCOPE"}

# Candidate thresholds to sweep. Centered on the provisional 0.70 lock.
_CANDIDATE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


async def _run_item(record: dict[str, Any], *, sem: asyncio.Semaphore) -> dict[str, Any]:
    """Run one calibration item through the graph + judge. Records the judge
    score alongside the human gold label for later correlation."""
    rid = record.get("id", "?")
    turns = record.get("turns")
    query = record.get("query") or (turns[-1] if turns else "")
    gold = record.get("gold_faithful")

    async with sem:
        try:
            run = await run_one_query(query=query if not turns else None, turns=turns)
        except Exception as e:
            logger.error(f"[{rid}] graph crashed: {e}")
            return {"id": rid, "query": query, "gold_faithful": gold,
                    "judge_score": None, "intent": None, "error": f"graph_crashed: {e}"}

        intent = run["intent"]
        judge_score: float | None = None
        judge_failed = False
        if intent not in _SKIP_JUDGE_INTENTS:
            judge = await judge_faithfulness(
                query=query,
                answer=run["answer"],
                retrieved_context=run["retrieved_context"],
            )
            judge_score = judge.score if judge is not None else None
            # A gradeable intent that yields no score is a JUDGE FAILURE (API
            # error / unparseable JSON after retry) — NOT a "no-signal" skip.
            # Conflating the two would let a dead judge masquerade as "0 usable,
            # all skipped" and silently corrupt the lock/swap decision.
            judge_failed = judge is None

    return {
        "id": rid,
        "query": query,
        "gold_faithful": gold,
        "judge_score": judge_score,
        "intent": intent,
        "answer_preview": (run["answer"] or "")[:160],
        "error": (
            "judge_returned_none (API failure or unparseable JSON after retry)"
            if judge_failed
            else None
        ),
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    # Epsilon (not == 0): with floating point, a zero-variance series yields a
    # tiny non-zero stddev that would escape an exact check and return a fake
    # 0.0 correlation. A judge that scores everything identically has NO signal
    # → report undefined (None), not 0.0.
    if vx < 1e-12 or vy < 1e-12:
        return None
    return cov / (vx * vy)


def _rankdata(vals: list[float]) -> list[float]:
    """Average-rank transform (ties share the mean of their rank span)."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def _cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    """Cohen's κ for the 2×2 (judge-pass/fail vs gold-pass/fail) table."""
    n = tp + fp + fn + tn
    if n == 0:
        return None
    po = (tp + tn) / n
    p_pass = ((tp + fp) / n) * ((tp + fn) / n)
    p_fail = ((fn + tn) / n) * ((fp + tn) / n)
    pe = p_pass + p_fail
    if pe == 1.0:
        return None
    return (po - pe) / (1 - pe)


def _threshold_report(pairs: list[tuple[float, float]], *, gold_floor: float) -> list[dict[str, Any]]:
    """For each candidate τ, treat judge_score>=τ as 'faithful' and compare to
    the gold label (gold_faithful>=gold_floor is 'faithful'). Reports the 2×2
    table + precision/recall/F1/κ/accuracy."""
    rows = []
    for tau in _CANDIDATE_THRESHOLDS:
        tp = fp = fn = tn = 0
        for judge_s, gold_s in pairs:
            judge_pass = judge_s >= tau
            gold_pass = gold_s >= gold_floor
            if judge_pass and gold_pass:
                tp += 1
            elif judge_pass and not gold_pass:
                fp += 1
            elif not judge_pass and gold_pass:
                fn += 1
            else:
                tn += 1
        total = tp + fp + fn + tn
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall)
            else None
        )
        accuracy = (tp + tn) / total if total else None
        rows.append({
            "threshold": tau,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy,
            "kappa": _cohen_kappa(tp, fp, fn, tn),
        })
    return rows


def _fmt(x: float | None, p: int = 3) -> str:
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"


async def main(path: Path, concurrency: int, gold_floor: float) -> int:
    settings = get_settings()
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        logger.error(f"Expected JSON array, got {type(records).__name__}")
        return 2

    logger.info(
        f"Calibrating judge={settings.judge_llm_model} against {len(records)} items "
        f"from {path.name} (gold_floor={gold_floor})"
    )
    start = time.perf_counter()
    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*[_run_item(r, sem=sem) for r in records])
    elapsed = round(time.perf_counter() - start, 2)

    # Partition results.
    usable: list[tuple[float, float]] = []   # (judge_score, gold_faithful)
    unlabeled, skipped, errored = [], [], []
    for r in results:
        if r["error"]:
            errored.append(r)
        elif r["judge_score"] is None:
            skipped.append(r)            # intent skipped the judge / no signal
        elif r["gold_faithful"] is None:
            unlabeled.append(r)          # judge ran but no human label yet
        else:
            usable.append((float(r["judge_score"]), float(r["gold_faithful"])))

    judge_scores = [j for j, _ in usable]
    gold_scores = [g for _, g in usable]
    pearson = _pearson(judge_scores, gold_scores)
    spearman = _spearman(judge_scores, gold_scores)
    thr_rows = _threshold_report(usable, gold_floor=gold_floor) if usable else []
    best = max(
        (row for row in thr_rows if row["kappa"] is not None),
        key=lambda row: row["kappa"],
        default=None,
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print(f" JUDGE CALIBRATION (D3) — {path.name}")
    print(f" judge_model = {settings.judge_llm_model}")
    print("=" * 76)
    print(f"  items total      : {len(records)}")
    print(f"  usable (labeled) : {len(usable)}")
    print(f"  unlabeled        : {len(unlabeled)}   (judge ran, no gold_faithful — please label)")
    print(f"  judge-skipped    : {len(skipped)}   (intent has no faithfulness signal)")
    print(f"  errored          : {len(errored)}")
    print(f"  elapsed          : {elapsed}s")
    print("-" * 76)
    print(f"  Pearson  r (judge vs gold) : {_fmt(pearson)}")
    print(f"  Spearman ρ (judge vs gold) : {_fmt(spearman)}")
    print("-" * 76)
    if thr_rows:
        print("  τ      TP  FP  FN  TN   prec   recall  f1     acc    kappa")
        for row in thr_rows:
            marker = "  <-- best κ" if best and row["threshold"] == best["threshold"] else ""
            print(
                f"  {row['threshold']:.2f}  "
                f"{row['tp']:>2d}  {row['fp']:>2d}  {row['fn']:>2d}  {row['tn']:>2d}   "
                f"{_fmt(row['precision'],2):>5s}  {_fmt(row['recall'],2):>6s}  "
                f"{_fmt(row['f1'],2):>5s}  {_fmt(row['accuracy'],2):>5s}  "
                f"{_fmt(row['kappa'],3):>6s}{marker}"
            )
    print("=" * 76)

    # ── Decision guidance (per plan D3 rule) ───────────────────────────────────
    print("\n DECISION (plan D3):")
    if len(usable) < 30:
        print(f"  ⚠ Only {len(usable)} labeled items — aim for ≳40-50 for a stable lock.")
    if spearman is None:
        print("  ⚠ Not enough labeled data to assess correlation. Label more items and re-run.")
    elif spearman >= 0.6 and best and best["kappa"] is not None and best["kappa"] >= 0.6:
        print(f"  ✓ V4 Pro tracks gold (Spearman ρ={_fmt(spearman)}, best κ={_fmt(best['kappa'])}).")
        print(f"    → LOCK judge threshold at τ={best['threshold']:.2f} "
              f"(update settings.faithfulness_min / eval gate).")
    else:
        print(f"  ✗ Weak correlation (Spearman ρ={_fmt(spearman)}, "
              f"best κ={_fmt(best['kappa']) if best else '—'}).")
        print(f"    → SWAP to fallback judge ({settings.judge_llm_fallback_model}) and re-run.")
    if unlabeled:
        print(f"\n  {len(unlabeled)} UNLABELED items to grade (add gold_faithful):")
        for r in unlabeled[:15]:
            print(f"    - {r['id']}: judge={_fmt(r['judge_score'],2)}  "
                  f"q={r['query'][:50]!r}  ans={r['answer_preview'][:60]!r}")
    print()

    # Persist raw results for offline analysis / re-thresholding without re-running.
    out_path = path.with_name(path.stem + "_judge_run.json")
    out_path.write_text(
        json.dumps(
            {
                "judge_model": settings.judge_llm_model,
                "gold_floor": gold_floor,
                "pearson": pearson,
                "spearman": spearman,
                "thresholds": thr_rows,
                "best_threshold": best["threshold"] if best else None,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f" Raw run + per-item scores written to: {out_path}")
    return 0


def cli() -> int:
    # Windows consoles/pipes default to cp1252, which cannot encode the
    # report's ρ/κ/✓/✗/— glyphs and crashes the print mid-report (before the
    # results JSON is written). Force UTF-8 so the report always renders.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
    p = argparse.ArgumentParser(description="D3 — calibrate the faithfulness judge vs human gold labels")
    p.add_argument("path", type=Path, help="JSON array of calibration items (see module docstring)")
    p.add_argument("--concurrency", type=int, default=2, help="Parallelism (default 2)")
    p.add_argument(
        "--gold-floor", type=float, default=0.7,
        help="gold_faithful >= this counts as 'faithful' for the confusion matrix (default 0.7)",
    )
    args = p.parse_args()
    return asyncio.run(main(args.path, args.concurrency, args.gold_floor))


if __name__ == "__main__":
    sys.exit(cli())
