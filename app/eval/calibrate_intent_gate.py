"""Calibration harness for the semantic intent gate.

Measures the embedding-similarity distribution of the 54 hallucination probes
(labeled by their expected intent) and finds a clean threshold/margin pair
that catches the chit-chat the regex misses WITHOUT mis-routing real KNOWLEDGE
queries as chit-chat.

Method
------
1. Load the 54 probes from hallucination_probe.
2. Embed each query via bge-m3 (same model the gate uses).
3. Compute cosine similarity to every intent centroid from intent_seed.yaml.
4. For each query, record (best_intent, best_sim, second_sim, margin).
5. Split into two groups:
     - "should gate"   (label in {GREETING, AMBIGUOUS, OFF_SCOPE, TOPIC_LIST})
     - "should NOT gate" (label in {KNOWLEDGE, COACHING, MALICIOUS})
6. Sweep threshold from 0.40 to 0.80 in 0.01 steps. For each, count:
     TP = should-gate queries correctly committed to their label
     FP = should-NOT-gate queries that the gate still committed on
   Print the (TPR, FPR) table + the operating point that maximizes TPR with
   FPR = 0. That's the cleanest cutoff - no chit-chat leaks, all real
   questions pass through untouched.

Run: python -m app.eval.calibrate_intent_gate
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml

from app.eval.hallucination_probe import PROBES
from app.config.embedding_config import ensure_llamaindex_configured
from app.graph.intent_classifier import _compute_centroids, _embed_one, _l2_normalize, _cosine


# Gate-eligible intents (matches the local Intent Literal in intent_classifier).
_GATE_OK = {"GREETING", "AMBIGUOUS", "OFF_SCOPE", "TOPIC_LIST"}


@dataclass
class Row:
    q: str
    label: str
    best_intent: str
    best_sim: float
    second_sim: float
    margin: float


async def _measure() -> list[Row]:
    ensure_llamaindex_configured()
    centroids = await _compute_centroids()
    if not centroids:
        raise SystemExit("No centroids - is intent_seed.yaml populated?")

    rows: list[Row] = []
    for p in PROBES:
        label = p.intent_hint or "?"
        vec = await _embed_one(p.q)
        vec = _l2_normalize(vec)
        sims = {intent: _cosine(vec, c) for intent, c in centroids.items()}
        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        best_intent, best_sim = ranked[0]
        second_sim = ranked[1][1] if len(ranked) > 1 else 0.0
        rows.append(Row(
            q=p.q, label=label, best_intent=best_intent,
            best_sim=best_sim, second_sim=second_sim,
            margin=best_sim - second_sim,
        ))
    return rows


def _gate_decision(r: Row, threshold: float, margin: float) -> str | None:
    """Mirror classify_semantic's commit rule. Returns the committed intent or None."""
    if r.best_intent not in _GATE_OK:
        return None
    if r.best_sim >= threshold and r.margin >= margin:
        return r.best_intent
    return None


def _sweep(rows: list[Row]) -> None:
    print(f"\n{'thr':>6}  {'mar':>5}  {'TP':>3}  {'FP':>3}  {'TPR':>5}  {'FPR':>5}  notes")
    print("-" * 70)
    should_gate = [r for r in rows if r.label in _GATE_OK]
    should_not = [r for r in rows if r.label not in _GATE_OK]
    n_gate, n_nogate = len(should_gate), len(should_not)

    best_clean = None
    for thr_x10 in range(40, 85):
        for mar_x10 in (8, 10, 12, 15):
            thr = thr_x10 / 100
            mar = mar_x10 / 100
            tp = sum(1 for r in should_gate if _gate_decision(r, thr, mar) == r.label)
            fp = sum(1 for r in should_not if _gate_decision(r, thr, mar) is not None)
            tpr = tp / n_gate if n_gate else 0
            fpr = fp / n_nogate if n_nogate else 0
            if fpr == 0 and (best_clean is None or tpr > best_clean[0]):
                best_clean = (tpr, thr, mar, tp, fp)
            if mar_x10 == 10 and thr_x10 % 5 == 0:
                note = ""
                if fpr == 0:
                    note = "clean"
                elif fp > 0:
                    note = f"({fp} misroutes)"
                print(f"{thr:6.2f}  {mar:5.2f}  {tp:3d}  {fp:3d}  {tpr*100:4.0f}%  {fpr*100:4.0f}%  {note}")

    if best_clean:
        tpr, thr, mar, tp, fp = best_clean
        print("\n" + "=" * 70)
        print(f"BEST CLEAN (FPR=0): thr={thr}  mar={mar}  TPR={tpr*100:.0f}% ({tp}/{n_gate})")
    else:
        print("\n" + "=" * 70)
        print("No threshold achieves FPR=0 - relax or grow probe set.")


def _dump_per_query(rows: list[Row]) -> None:
    print(f"\n{'label':<13}  {'best':<13}  {'sim':>5}  {'mar':>5}  q")
    print("-" * 90)
    for r in rows:
        print(f"{r.label:<13}  {r.best_intent:<13}  {r.best_sim:5.3f}  {r.margin:5.3f}  {r.q[:40]!r}")


async def run() -> int:
    print("=" * 70)
    print("SEMANTIC INTENT-GATE CALIBRATION")
    print("=" * 70)
    print(f"Probes: {len(PROBES)} (from hallucination_probe)")
    print(f"Gate-eligible intents: {sorted(_GATE_OK)}")

    rows = await _measure()

    rows.sort(key=lambda r: r.margin, reverse=True)
    _dump_per_query(rows)

    print(f"\n[Centroid sizes] intent -> #seed examples")
    seed_path = Path(__file__).parent.parent / "graph" / "intent_seed.yaml"
    with open(seed_path, encoding="utf-8") as f:
        seed = yaml.safe_load(f) or {}
    for intent, examples in seed.items():
        print(f"  {intent:<13}  {len(examples)}")

    _sweep(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
