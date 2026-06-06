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
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from app.config.settings import get_settings
from app.eval.judge import judge_faithfulness
from app.eval.runner import run_one_query

# Intents whose answers carry no faithfulness signal — judge is skipped, so
# these can't contribute to calibration (mirrors runner.evaluate_record).
_SKIP_JUDGE_INTENTS = {"GREETING", "AMBIGUOUS", "MALICIOUS", "TOPIC_LIST", "OFF_SCOPE"}

# Candidate thresholds to sweep. Centered on the provisional 0.70 lock.
_CANDIDATE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# The provisional lock from the plan (D3). When the κ-maximizing τ is
# statistically indistinguishable from this value (its κ falls inside the best
# τ's bootstrap CI), we PREFER this stable plateau point over a knife-edge max.
_PROVISIONAL_TAU = 0.70

# Per-class minimum on the DISCRIMINATING (non-refusal) labeled subset. A
# faithfulness judge is a discriminator: κ is meaningless unless BOTH a faithful
# and an unfaithful class are populated. Below this we refuse to LOCK or SWAP
# and instead demand more negatives (oracle D3 guidance: ≥15-20/class ideal).
_MIN_PER_CLASS = 10
_HEALTHY_PER_CLASS = 15

# Canned NOT-FOUND refusals emitted by the graph (pipeline.py:1116 LLM-prompt
# path + pipeline.py:1150 gate path). A refusal makes NO factual claim, so the
# judge scores it 1.0 by rule (faithful-by-vacuity). These are trivially
# gold=1.0=judge=1.0 and would DOMINATE both observed and expected agreement in
# κ — manufacturing a high-looking number that is pure artifact. They are
# EXCLUDED from the κ/ρ math and audited as their own bucket (a refusal WITH
# retrieved context is a possible over-refusal signal, relevant to H9).
_REFUSAL_MARKERS = (
    "belum menemukan info soal itu",   # ID — both canned templates lead with this
    "belum menemukan info",            # ID — looser fallback
    "couldn't find any info",          # EN equivalent
    "could not find any info",
)


def _is_refusal(answer: str | None) -> bool:
    """True when the answer is a canned NOT-FOUND refusal (no factual claim)."""
    if not answer:
        return False
    a = answer.strip().lower()
    return any(marker in a for marker in _REFUSAL_MARKERS)


async def _run_item(record: dict[str, Any], *, sem: asyncio.Semaphore) -> dict[str, Any]:
    """Run one calibration item through the judge. Records the judge score
    alongside the human gold label for later correlation.

    TWO MODES:
    - GENERATED (default): run the query through the LIVE RAG graph, judge the
      generator's answer against the retrieved context. Measures the real
      end-to-end system but a well-behaved RAG on in-domain queries produces
      almost no negatives (it answers faithfully or refuses) — single-class,
      can't calibrate a discriminator.
    - PROVIDED-ANSWER (record carries a pre-written `answer`): skip the
      generator and judge the SUPPLIED answer against the SUPPLIED
      `retrieved_context`. This is how we inject a real NEGATIVE class
      (realistic synthetic hallucinations, gold=0.0) and partials (gold=0.5).
      Methodologically sound: the judge is a pure function of
      (query, context, answer) — it neither knows nor cares whether Gemini or a
      curator wrote the answer, so validating "does the judge track human
      faithfulness judgment?" is provenance-agnostic. The answer must be a
      CONFIDENT assertion, never a refusal (refusals auto-score 1.0 and poison
      the negative class)."""
    rid = record.get("id", "?")
    turns = record.get("turns")
    query = record.get("query") or (turns[-1] if turns else "")
    gold = record.get("gold_faithful")
    provided_answer = record.get("answer")
    provided_mode = provided_answer is not None

    async with sem:
        if provided_mode:
            # PROVIDED-ANSWER: judge the supplied answer/context directly. No
            # generator, no graph — so `intent` is recorded as the synthetic
            # marker and the judge is always invoked (the item exists precisely
            # to be graded).
            answer = provided_answer or ""
            retrieved_context = record.get("retrieved_context") or []
            intent = record.get("intent") or "PROVIDED"
            judge = await judge_faithfulness(
                query=query,
                answer=answer,
                retrieved_context=retrieved_context,
            )
            judge_score = judge.score if judge is not None else None
            judge_failed = judge is None
        else:
            try:
                run = await run_one_query(query=query if not turns else None, turns=turns)
            except Exception as e:
                logger.error(f"[{rid}] graph crashed: {e}")
                return {"id": rid, "query": query, "gold_faithful": gold,
                        "judge_score": None, "intent": None, "mode": "generated",
                        "is_refusal": False, "error": f"graph_crashed: {e}"}
            answer = run["answer"] or ""
            retrieved_context = run.get("retrieved_context") or []
            intent = run["intent"]
            judge_score = None
            judge_failed = False
            if intent not in _SKIP_JUDGE_INTENTS:
                judge = await judge_faithfulness(
                    query=query,
                    answer=answer,
                    retrieved_context=retrieved_context,
                )
                judge_score = judge.score if judge is not None else None
                # A gradeable intent that yields no score is a JUDGE FAILURE (API
                # error / unparseable JSON after retry) — NOT a "no-signal" skip.
                # Conflating the two would let a dead judge masquerade as "0
                # usable, all skipped" and silently corrupt the lock/swap call.
                judge_failed = judge is None

    return {
        "id": rid,
        "query": query,
        "gold_faithful": gold,
        "judge_score": judge_score,
        "intent": intent,
        "mode": "provided" if provided_mode else "generated",
        # Refusals make no factual claim → judge=1.0 by rule. Flagged here so the
        # partition can EXCLUDE them from κ/ρ (they're trivially 1.0=1.0 and
        # would inflate agreement) and audit them separately. A synthetic
        # provided-answer must never be refusal-shaped, so this should only fire
        # on generated items.
        "is_refusal": _is_refusal(answer),
        # Full answer + context are persisted so a human (or reviewer) can
        # actually ground-truth the gold label offline WITHOUT re-running the
        # live graph. answer_preview is kept only for the compact console list.
        "answer_preview": answer[:160],
        "answer": answer,
        "retrieved_context": retrieved_context,
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


def _bin3(score: float) -> float:
    """Snap a continuous [0,1] score to the nearest 3-level ordinal grade
    {0.0, 0.5, 1.0}. Gold labels are already on this grid; judge scores are
    binned to it so the ordinal (weighted-κ) comparison is apples-to-apples."""
    if score < 0.25:
        return 0.0
    if score < 0.75:
        return 0.5
    return 1.0


_GRADES3 = (0.0, 0.5, 1.0)


def _quadratic_weighted_kappa(judge_bins: list[float], gold_bins: list[float]) -> float | None:
    """Quadratic-weighted Cohen's κ over the 3-level ordinal grid {0,0.5,1.0}.

    Unlike the binary κ (which lumps 0.5 partials in with 0.0 hallucinations and
    can't tell "judge caught a partial" from "judge caught outright fabrication"),
    weighted κ uses the FULL ordinal signal and penalizes a 1.0-vs-0.0 miss far
    more than a 1.0-vs-0.5 miss (penalty ∝ squared grade distance). Reported
    ALONGSIDE the binary κ: if the two disagree, investigate before locking
    (oracle D3 guidance). Returns None when either rater has no variance."""
    n = len(judge_bins)
    if n == 0 or n != len(gold_bins):
        return None
    idx = {g: i for i, g in enumerate(_GRADES3)}
    k = len(_GRADES3)
    # Observed confusion matrix O[gold][judge].
    O = [[0.0] * k for _ in range(k)]
    for jb, gb in zip(judge_bins, gold_bins):
        O[idx[gb]][idx[jb]] += 1.0
    # Marginals → expected matrix E under independence.
    gold_marg = [sum(O[r]) for r in range(k)]
    judge_marg = [sum(O[r][c] for r in range(k)) for c in range(k)]
    if all(m == 0 for m in gold_marg) or all(m == 0 for m in judge_marg):
        return None
    # Quadratic weights w[i][j] = ((i-j)/(k-1))^2.
    W = [[((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)]
    num = sum(W[i][j] * O[i][j] for i in range(k) for j in range(k))
    den = sum(W[i][j] * (gold_marg[i] * judge_marg[j] / n) for i in range(k) for j in range(k))
    if den == 0:
        return None
    return 1.0 - num / den


def _bootstrap_ci(
    items: list[Any],
    stat_fn: Callable[[list[Any]], float | None],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI for a statistic computed over `items`.

    Resamples `items` WITH replacement `n_resamples` times, recomputes the
    statistic on each resample, and returns the (alpha/2, 1-alpha/2) percentiles.
    Used so the lock/swap call rests on an INTERVAL, not a knife-edge point
    estimate that one flipped item could swing (oracle D3 guidance). Returns
    (None, None) when there aren't enough items or the statistic is degenerate."""
    n = len(items)
    if n < 3:
        return (None, None)
    rng = random.Random(seed)
    stats: list[float] = []
    for _ in range(n_resamples):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        val = stat_fn(sample)
        if val is not None:
            stats.append(val)
    if len(stats) < max(20, n_resamples // 10):
        # Too many degenerate resamples (e.g. zero-variance) → CI not meaningful.
        return (None, None)
    stats.sort()
    lo_i = int((alpha / 2) * len(stats))
    hi_i = min(len(stats) - 1, int((1 - alpha / 2) * len(stats)))
    return (stats[lo_i], stats[hi_i])


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


def _kappa_at_tau(pairs: list[tuple[float, float]], tau: float, gold_floor: float) -> float | None:
    """Binary Cohen's κ at a fixed τ — used as the bootstrap statistic so the
    lock/swap call rests on a κ CONFIDENCE INTERVAL, not a point estimate."""
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
    return _cohen_kappa(tp, fp, fn, tn)


def _fmt(x: float | None, p: int = 3) -> str:
    return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"


def _ci_str(ci: tuple[float | None, float | None]) -> str:
    """Render a bootstrap CI as ' [lo,hi]', or '' when undefined."""
    lo, hi = ci
    if lo is None or hi is None:
        return ""
    return f" [{lo:.2f},{hi:.2f}]"


async def main(path: Path, concurrency: int, gold_floor: float, from_run: Path | None = None) -> int:
    settings = get_settings()
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        logger.error(f"Expected JSON array, got {type(records).__name__}")
        return 2

    if from_run is not None:
        # OFFLINE re-threshold: reuse the judge scores from a prior live run and
        # re-merge the (freshly edited) gold_faithful labels from the seed by id.
        # Zero API calls — labels don't change judge scores, so re-running the
        # live graph + judge just to relabel would waste ~100 calls per pass.
        if not from_run.exists():
            logger.error(f"--from-run file not found: {from_run}")
            return 2
        prior = json.loads(from_run.read_text(encoding="utf-8"))
        gold_by_id = {r.get("id"): r.get("gold_faithful") for r in records}
        results = []
        for r in prior.get("results", []):
            merged = dict(r)
            if merged.get("id") in gold_by_id:
                merged["gold_faithful"] = gold_by_id[merged["id"]]
            # Backfill fields added after older runs were written, so the new
            # refusal-exclusion partition works on pre-upgrade run data too. The
            # full `answer` was always persisted, so is_refusal is recomputable.
            if "is_refusal" not in merged:
                merged["is_refusal"] = _is_refusal(merged.get("answer"))
            if "mode" not in merged:
                merged["mode"] = "provided" if merged.get("answer") is not None and not merged.get("intent") else "generated"
            results.append(merged)
        logger.info(
            f"Offline re-threshold from {from_run.name}: reusing {len(results)} judge "
            f"scores, re-merged gold labels from {path.name} (gold_floor={gold_floor})"
        )
        elapsed = 0.0
    else:
        logger.info(
            f"Calibrating judge={settings.judge_llm_model} against {len(records)} items "
            f"from {path.name} (gold_floor={gold_floor})"
        )
        start = time.perf_counter()
        sem = asyncio.Semaphore(concurrency)
        results = await asyncio.gather(*[_run_item(r, sem=sem) for r in records])
        elapsed = round(time.perf_counter() - start, 2)

    # Partition results.
    #
    # REFUSALS are split out and EXCLUDED from the κ/ρ math: a canned NOT-FOUND
    # refusal makes no factual claim, the judge scores it 1.0 by rule, and the
    # gold is 1.0 — a trivial 1.0=1.0 agreement that carries zero discrimination
    # signal but would DOMINATE both observed and expected agreement in κ
    # (oracle D3 guidance). They are audited as their own bucket instead.
    usable: list[tuple[float, float]] = []   # (judge_score, gold) — NON-refusal
    unlabeled, skipped, errored, refusals = [], [], [], []
    for r in results:
        if r["error"]:
            errored.append(r)
        elif r["judge_score"] is None:
            skipped.append(r)            # intent skipped the judge / no signal
        elif r.get("is_refusal"):
            refusals.append(r)           # excluded from κ/ρ, audited separately
        elif r["gold_faithful"] is None:
            unlabeled.append(r)          # judge ran but no human label yet
        else:
            usable.append((float(r["judge_score"]), float(r["gold_faithful"])))

    judge_scores = [j for j, _ in usable]
    gold_scores = [g for _, g in usable]
    pearson = _pearson(judge_scores, gold_scores)
    spearman = _spearman(judge_scores, gold_scores)          # 3-level ordinal ρ
    spearman_ci = _bootstrap_ci(
        usable, lambda s: _spearman([j for j, _ in s], [g for _, g in s])
    )
    # Quadratic-weighted κ over the 3-level grid — corroborates the binary κ
    # using the full ordinal signal (penalizes 1.0-vs-0.0 more than 1.0-vs-0.5).
    weighted_kappa = _quadratic_weighted_kappa(
        [_bin3(j) for j in judge_scores], [_bin3(g) for g in gold_scores]
    )
    thr_rows = _threshold_report(usable, gold_floor=gold_floor) if usable else []
    best = max(
        (row for row in thr_rows if row["kappa"] is not None),
        key=lambda row: row["kappa"],
        default=None,
    )

    # Per-class population on the DISCRIMINATING subset (binarized at gold_floor).
    # A faithfulness judge is a discriminator — κ is meaningless unless BOTH
    # classes are present and adequately sized.
    faithful_n = sum(1 for _, g in usable if g >= gold_floor)
    unfaithful_n = sum(1 for _, g in usable if g < gold_floor)
    both_classes_ok = faithful_n >= _MIN_PER_CLASS and unfaithful_n >= _MIN_PER_CLASS

    # Bootstrap CI on κ at the κ-max τ, and the plateau check: prefer the
    # provisional τ=0.70 when its κ is statistically indistinguishable from the
    # max (inside the CI) — locking a stable plateau, not a knife-edge point.
    best_kappa_ci: tuple[float | None, float | None] = (None, None)
    kappa_at_provisional: float | None = None
    lock_tau: float | None = None
    if best is not None and both_classes_ok:
        best_tau = best["threshold"]
        best_kappa_ci = _bootstrap_ci(
            usable, lambda s: _kappa_at_tau(s, best_tau, gold_floor)
        )
        kappa_at_provisional = _kappa_at_tau(usable, _PROVISIONAL_TAU, gold_floor)
        lo = best_kappa_ci[0]
        if (
            kappa_at_provisional is not None
            and lo is not None
            and kappa_at_provisional >= lo
        ):
            lock_tau = _PROVISIONAL_TAU      # plateau: prefer the provisional
        else:
            lock_tau = best_tau              # distinct max — lock it

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 76)
    print(f" JUDGE CALIBRATION (D3) — {path.name}")
    print(f" judge_model = {settings.judge_llm_model}")
    print("=" * 76)
    print(f"  items total      : {len(records)}")
    print(f"  usable (κ/ρ)     : {len(usable)}   (non-refusal, labeled)")
    print(f"    ├─ faithful    : {faithful_n}   (gold ≥ {gold_floor})")
    print(f"    └─ unfaithful  : {unfaithful_n}   (gold < {gold_floor})")
    print(f"  refusals (audit) : {len(refusals)}   (excluded from κ/ρ — no claim, judge=1.0 by rule)")
    print(f"  unlabeled        : {len(unlabeled)}   (judge ran, no gold_faithful — please label)")
    print(f"  judge-skipped    : {len(skipped)}   (intent has no faithfulness signal)")
    print(f"  errored          : {len(errored)}")
    print(f"  elapsed          : {elapsed}s")
    print("-" * 76)
    print(f"  Pearson  r (judge vs gold)      : {_fmt(pearson)}")
    print(f"  Spearman ρ (judge vs gold, 3-lvl): {_fmt(spearman)}{_ci_str(spearman_ci)}")
    print(f"  Quadratic-weighted κ (3-level)  : {_fmt(weighted_kappa)}")
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

    # ── Decision (per plan D3 rule, with oracle methodology guards) ────────────
    # LOCK iff: both classes populated AND Spearman ρ ≥ 0.6 AND best-τ κ ≥ 0.6.
    #   → lock at the plateau τ (prefers provisional 0.70 when indistinguishable).
    # SWAP iff: both classes populated AND the κ CI UPPER bound < 0.6 (the judge
    #   is confidently sub-threshold — not just unlucky on a small sample).
    # Otherwise INCONCLUSIVE: never discard a possibly-good judge on noise, and
    #   never lock a τ without a real negative class.
    decision = "inconclusive"
    print("\n DECISION (plan D3):")
    if not both_classes_ok:
        decision = "insufficient_classes"
        print(f"  ⚠ DISCRIMINATING set too thin: faithful={faithful_n}, "
              f"unfaithful={unfaithful_n} (need ≥{_MIN_PER_CLASS}/class, "
              f"≥{_HEALTHY_PER_CLASS} healthy).")
        print("    A faithfulness judge is a DISCRIMINATOR — κ is meaningless")
        print("    without both a faithful AND an unfaithful class. Add realistic")
        print("    synthetic negatives (provided-answer items, gold=0.0) and re-run.")
        print("    → NO LOCK, NO SWAP until the negative class is populated.")
    elif spearman is None or best is None or best["kappa"] is None:
        print("  ⚠ Not enough signal to assess correlation. Label more items and re-run.")
    else:
        best_kappa = best["kappa"]
        ci_hi = best_kappa_ci[1]
        if spearman >= 0.6 and best_kappa >= 0.6:
            decision = "lock"
            plateau = " (provisional plateau)" if lock_tau == _PROVISIONAL_TAU else ""
            print(f"  ✓ V4 Pro tracks gold: Spearman ρ={_fmt(spearman)}{_ci_str(spearman_ci)}, "
                  f"best κ={_fmt(best_kappa)}{_ci_str(best_kappa_ci)}, "
                  f"weighted κ={_fmt(weighted_kappa)}.")
            print(f"    → LOCK judge threshold at τ={lock_tau:.2f}{plateau} "
                  f"(set settings.faithfulness_min / eval gate).")
            if weighted_kappa is not None and weighted_kappa < 0.4:
                print(f"    ⚠ but weighted κ={_fmt(weighted_kappa)} is weak — the judge may be")
                print("      mis-ranking partials vs hallucinations. Inspect before trusting.")
        elif ci_hi is not None and ci_hi < 0.6:
            decision = "swap"
            print(f"  ✗ Judge confidently sub-threshold: best κ={_fmt(best_kappa)} "
                  f"(CI upper {_fmt(ci_hi,2)} < 0.6), Spearman ρ={_fmt(spearman)}.")
            print(f"    → SWAP to fallback judge ({settings.judge_llm_fallback_model}) and re-run.")
        else:
            print(f"  ~ INCONCLUSIVE: Spearman ρ={_fmt(spearman)}{_ci_str(spearman_ci)}, "
                  f"best κ={_fmt(best_kappa)}{_ci_str(best_kappa_ci)}.")
            print("    κ CI straddles 0.6 — neither a confident lock nor a confident")
            print("    swap. Add more labeled discriminating items (esp. negatives) and re-run.")

    # Refusal audit — a refusal WITH retrieved context is a possible OVER-refusal
    # (the gate fired despite usable chunks). The faithfulness axis is blind to
    # this (a false refusal is still "no claim = faithful"), so surface it here.
    if refusals:
        with_ctx = sum(1 for r in refusals if r.get("retrieved_context"))
        print(f"\n  REFUSAL AUDIT: {len(refusals)} refusals, {with_ctx} WITH retrieved context")
        print("    (possible over-refusal — relevant to H9 colloquial false-pass; "
              "not a faithfulness defect).")

    if unlabeled:
        print(f"\n  {len(unlabeled)} UNLABELED items to grade (add gold_faithful):")
        for r in unlabeled[:15]:
            print(f"    - {r['id']}: judge={_fmt(r['judge_score'],2)}  "
                  f"q={r['query'][:50]!r}  ans={r['answer_preview'][:60]!r}")
    print()

    # Persist raw results for offline analysis / re-thresholding without re-running.
    # An offline re-threshold writes to a DISTINCT file so it can never clobber
    # the expensive live-run source it read from (--from-run).
    if from_run is not None:
        out_path = path.with_name(path.stem + "_judge_run_relabeled.json")
    else:
        out_path = path.with_name(path.stem + "_judge_run.json")
    out_path.write_text(
        json.dumps(
            {
                "judge_model": settings.judge_llm_model,
                "gold_floor": gold_floor,
                "decision": decision,
                "lock_tau": lock_tau,
                "pearson": pearson,
                "spearman": spearman,
                "spearman_ci": list(spearman_ci),
                "weighted_kappa": weighted_kappa,
                "best_threshold": best["threshold"] if best else None,
                "best_kappa": best["kappa"] if best else None,
                "best_kappa_ci": list(best_kappa_ci),
                "kappa_at_provisional": kappa_at_provisional,
                "provisional_tau": _PROVISIONAL_TAU,
                "faithful_n": faithful_n,
                "unfaithful_n": unfaithful_n,
                "both_classes_ok": both_classes_ok,
                "refusals_n": len(refusals),
                "usable_n": len(usable),
                "thresholds": thr_rows,
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
    p.add_argument(
        "--from-run", type=Path, default=None,
        help=(
            "Offline re-threshold: reuse judge scores from a prior "
            "*_judge_run.json and re-merge gold_faithful from the seed by id. "
            "Zero API calls — use after editing labels."
        ),
    )
    args = p.parse_args()
    return asyncio.run(main(args.path, args.concurrency, args.gold_floor, args.from_run))


if __name__ == "__main__":
    sys.exit(cli())
