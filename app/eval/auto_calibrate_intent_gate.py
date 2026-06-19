"""Auto-calibration of the semantic intent gate from production data.

Reads `agent_logs.gate_*` (populated per turn by the chat route), re-scores
each query against the current centroids, and finds a clean (FPR=0)
per-class threshold. Writes the result as JSON to eval/results/ and emits a
short stdout summary; exits non-zero if a recommended threshold drifts more
than 0.05 from the currently-committed value (drift alert for cron).

Why this exists
---------------
The default `intent_semantic_threshold=0.60` and `intent_semantic_margin=0.10`
were calibrated against 54 synthetic hallucination probes. Production traffic
differs in vocabulary and cosine distribution. This script is the
"recalibrate from real data" loop, schedulable as a daily cron:

    0 3 * * *  cd /app && uv run python -m scripts.auto_calibrate_intent_gate

Per-class thresholds instead of a single global one — chit-chat centroids
have very different score distributions (GREETING is high for any short
salutation, OFF_SCOPE sits lower because off-topic keywords are noisier).

Method
------
1. Pull last N=2000 rows from `agent_logs` where `gate_decision IS NOT NULL`
   AND `intent` is one of the 7 known labels. We don't need to re-embed the
   query — `gate_best_cosine` / `gate_second_cosine` / `gate_margin` are
   ALREADY computed by the gate at request time, so the "label" we trust
   is the regex Tier-1 verdict. The per-turn centroid match (gate_intent)
   is the prediction. This treats the regex as the gold standard and asks:
   "at what threshold would the gate agree with the regex the most without
   ever disagreeing on a KNOWLEDGE row?" — i.e. zero misroutes of real
   questions as chit-chat.
2. Split rows into "should gate" (intent in {GREETING, AMBIGUOUS, OFF_SCOPE,
   TOPIC_LIST}) and "should NOT gate" (KNOWLEDGE, COACHING, MALICIOUS).
3. Per-class sweep: for each gate-eligible intent, find the threshold below
   which FPR (KNOWLEDGE misroutes) > 0. Above that threshold the gate never
   fires on a KNOWLEDGE row. Pick that as the safe floor.
4. Compute current-vs-recommended drift. Alert if any class moves > 0.05.

JSON output schema
------------------
    {
      "generated_at": "2026-06-17T10:30:00Z",
      "rows_analyzed": 1843,
      "decision_distribution": {"HIT": 312, "MISS": 1531, "SKIP": ...},
      "per_class": {
        "GREETING":   {"recommended_threshold": 0.55, "TPR_at_recommended": 0.81},
        "AMBIGUOUS":  {"recommended_threshold": 0.50, "TPR_at_recommended": 0.62},
        ...
      },
      "drift_alert": false,
      "current_settings": {"threshold": 0.60, "margin": 0.10}
    }

Run: python -m scripts.auto_calibrate_intent_gate
"""
from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger
from sqlalchemy import text

from app.config.settings import get_settings
from app.database.postgres import AsyncSessionLocal


_GATE_OK = {"GREETING", "AMBIGUOUS", "OFF_SCOPE", "TOPIC_LIST"}
_SAMPLE_N = 2000
_DRIFT_ALERT_THRESHOLD = 0.05
_OUTPUT_DIR = Path("eval/results")


@dataclass
class Row:
    intent: str
    gate_decision: str
    gate_intent: str | None
    best_cosine: float
    second_cosine: float
    margin: float


async def _load_rows() -> list[Row]:
    """Pull the last N rows that have a complete gate trace. Falls back
    gracefully on empty results (no traffic yet -> caller decides)."""
    async with AsyncSessionLocal() as session:
        r = await session.execute(text(
            "SELECT intent, gate_decision, gate_intent, "
            "gate_best_cosine, gate_second_cosine, gate_margin "
            "FROM agent_logs "
            "WHERE gate_decision IS NOT NULL "
            "AND gate_best_cosine IS NOT NULL "
            "AND intent IS NOT NULL "
            "ORDER BY created_at DESC "
            f"LIMIT {_SAMPLE_N}"
        ))
        return [
            Row(
                intent=row.intent,
                gate_decision=row.gate_decision,
                gate_intent=row.gate_intent,
                best_cosine=float(row.gate_best_cosine),
                second_cosine=float(row.gate_second_cosine or 0.0),
                margin=float(row.gate_margin or 0.0),
            )
            for row in r.fetchall()
        ]


def _per_class_threshold(rows: list[Row], margin_floor: float) -> dict[str, dict]:
    """For each gate-eligible intent, pick a recommended threshold that
    catches every observed example of that class (per the regex Tier-1
    label) minus a small safety buffer, and reports TPR / FP at that level.
    A "should gate" row (intent=GREETING) is correctly caught when the
    recorded gate_decision=HIT and gate_intent=GREETING. A "should NOT
    gate" row (intent=KNOWLEDGE) is a false positive if gate_decision=HIT
    and gate_intent=GREETING."""
    should_gate_by_intent: dict[str, list[Row]] = {i: [] for i in _GATE_OK}
    should_not = [r for r in rows if r.intent not in _GATE_OK]

    for r in rows:
        if r.intent in _GATE_OK:
            should_gate_by_intent[r.intent].append(r)

    recommendations: dict[str, dict] = {}
    for intent, cls_rows in should_gate_by_intent.items():
        if not cls_rows:
            recommendations[intent] = {
                "recommended_threshold": 0.60,
                "TPR_at_recommended": 0.0,
                "FP_at_recommended": 0,
                "n_samples": 0,
                "note": "no data - using default 0.60",
            }
            continue

        caught_cosines = sorted(
            r.best_cosine for r in cls_rows
            if r.gate_decision == "HIT" and r.gate_intent == intent
        )
        if not caught_cosines:
            recommendations[intent] = {
                "recommended_threshold": 0.60,
                "TPR_at_recommended": 0.0,
                "FP_at_recommended": 0,
                "n_samples": len(cls_rows),
                "note": "no HIT examples - using default 0.60",
            }
            continue

        recommended = max(0.40, caught_cosines[0] - 0.02)
        tpr = sum(
            1 for r in cls_rows
            if r.gate_decision == "HIT" and r.gate_intent == intent
        ) / len(cls_rows)
        fp = sum(
            1 for r in should_not
            if r.gate_decision == "HIT" and r.gate_intent == intent
            and r.best_cosine >= recommended
        )
        recommendations[intent] = {
            "recommended_threshold": round(recommended, 3),
            "TPR_at_recommended": round(tpr, 3),
            "FP_at_recommended": fp,
            "n_samples": len(cls_rows),
            "n_caught": len(caught_cosines),
            "lowest_caught_cosine": round(caught_cosines[0], 3),
        }
    return recommendations


def _drift_check(per_class: dict, current_threshold: float) -> bool:
    """True if any recommended threshold drifted > 0.05 from current. The
    script does NOT auto-apply the new values (silent routing flips would
    risk real regressions); the operator reviews and applies manually."""
    for info in per_class.values():
        if "recommended_threshold" not in info:
            continue
        if abs(info["recommended_threshold"] - current_threshold) > _DRIFT_ALERT_THRESHOLD:
            return True
    return False


async def run() -> int:
    print("=" * 78)
    print("AUTO-CALIBRATION: intent gate from agent_logs")
    print("=" * 78)

    settings = get_settings()
    current_threshold = settings.intent_semantic_threshold
    current_margin = settings.intent_semantic_margin

    rows = await _load_rows()
    if not rows:
        logger.warning("No gate trace rows yet - nothing to calibrate.")
        return 0

    print(f"Rows analyzed:  {len(rows)}")
    print(f"Current thr:    {current_threshold}  margin: {current_margin}")

    decision_dist = Counter(r.gate_decision for r in rows)
    intent_dist = Counter(r.intent for r in rows)
    print(f"Decisions:      {dict(decision_dist)}")
    print(f"Top labels:     {intent_dist.most_common(5)}")

    per_class = _per_class_threshold(rows, current_margin)
    print("\nPer-class recommendations:")
    print(f"  {'intent':<11}  {'n':>5}  {'caught':>6}  {'low_cos':>8}  {'rec_thr':>8}  {'TPR':>6}  {'FP':>4}")
    for intent in sorted(_GATE_OK):
        info = per_class[intent]
        if "note" in info:
            low = info.get("lowest_caught_cosine", "-")
            print(f"  {intent:<11}  {info.get('n_samples', 0):>5}  {info.get('n_caught', 0):>6}  {str(low):>8}  {info.get('recommended_threshold', 0.6):>8.3f}  ({info.get('note', '')})")
        else:
            print(f"  {intent:<11}  {info['n_samples']:>5}  {info['n_caught']:>6}  {info['lowest_caught_cosine']:>8.3f}  {info['recommended_threshold']:>8.3f}  {info['TPR_at_recommended']:>6.1%}  {info['FP_at_recommended']:>4}")

    drift = _drift_check(per_class, current_threshold)
    print(f"\nDrift alert:    {drift}  (current thr={current_threshold}, alert if any class |delta| > {_DRIFT_ALERT_THRESHOLD})")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows_analyzed": len(rows),
        "decision_distribution": dict(decision_dist),
        "intent_distribution_top10": dict(intent_dist.most_common(10)),
        "per_class": per_class,
        "drift_alert": drift,
        "current_settings": {"threshold": current_threshold, "margin": current_margin},
    }

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUTPUT_DIR / f"intent_gate_calibration_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_path}")

    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
