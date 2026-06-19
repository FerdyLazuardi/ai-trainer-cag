"""70-query intent-gating quality test — realistic FO Amartha Indonesian queries.

End-to-end eval of the cascade (regex Tier-1 + semantic gate) BEFORE
deploying per-class threshold tuning. Each query is labeled with the
routing we WANT; the script measures what the cascade DID and grades it.

What it measures
----------------
1. Regex TPR/FPR (does Tier-1 catch the obvious cases?)
2. Gate HIT/MISS/SKIP (does the semantic gate add value beyond regex?)
3. Latency per routing path (bge-m3 embed cost)
4. Quality verdict per query:
   - "OK"     = cascade did the right thing
   - "FP"     = gate committed chitchat on a knowledge query (CRITICAL,
                skips KB and answers conversationally — wrong topic risk)
   - "FN"     = cascade let a chitchat query fall through to KNOWLEDGE
                (lost a small save, no info lost)
5. JSON output to eval/results/ for the dashboard's Gate Monitor.

The "safe to deploy" gate is FP=0. Any FP = the gate is too eager and
needs threshold tuning or better centroids.

Run: python -m app.eval.intent_gate_quality_70
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config.embedding_config import ensure_llamaindex_configured
from app.graph.intent_classifier import (
    classify_semantic_with_scores,
    _compute_centroids,
)
from app.graph.intent_rules import classify as rule_classify


@dataclass
class Q:
    q: str
    # "chitchat" = regex Tier-1 OR semantic gate SHOULD commit
    # "knowledge" = should fall through to KNOWLEDGE + retrieval
    expected_routing: str
    expected_intent: str = ""
    cluster: str = ""


# 70 realistic FO Indonesian queries. Mix of:
#   20 chit-chat fillers (halo/info/eh/dll) — gate should commit
#   15 borderline queries that look like chit-chat but ARE real questions
#   25 standard KNOWLEDGE queries (definitive, procedural)
#   10 typo/slang variants of KNOWLEDGE — should fall through
PROBES: list[Q] = [
    # ── 20 chit-chat fillers (gate SHOULD commit) ────────────────────────
    Q("halo", "chitchat", "GREETING", "salam"),
    Q("halo ka", "chitchat", "GREETING", "salam"),
    Q("met pagi ava", "chitchat", "GREETING", "salam"),
    Q("pagi semua", "chitchat", "GREETING", "salam"),
    Q("assalamualaikum", "chitchat", "GREETING", "salam"),
    Q("hallo", "chitchat", "GREETING", "salam"),
    Q("halooo", "chitchat", "GREETING", "salam"),
    Q("test", "chitchat", "GREETING", "salam"),
    Q("p", "chitchat", "GREETING", "salam"),
    Q("siapa kamu", "chitchat", "GREETING", "identity"),
    Q("kamu bot ya", "chitchat", "GREETING", "identity"),
    Q("harga emas hari ini brp", "chitchat", "OFF_SCOPE", "off-scope"),
    Q("iphone 15 bagus ga", "chitchat", "OFF_SCOPE", "off-scope"),
    Q("kapan indonesia merdeka", "chitchat", "OFF_SCOPE", "off-scope"),
    Q("resep nasi goreng dong", "chitchat", "OFF_SCOPE", "off-scope"),
    Q("skor bola tadi malam gmn", "chitchat", "OFF_SCOPE", "off-scope"),
    Q("hmm", "chitchat", "AMBIGUOUS", "filler"),
    Q("tanya dong", "chitchat", "AMBIGUOUS", "filler"),
    Q("bantuin", "chitchat", "AMBIGUOUS", "filler"),
    Q("ok", "chitchat", "AMBIGUOUS", "filler"),
    # ── 15 borderline queries — look like chit-chat but are real questions ─
    Q("lapor BM gimana caranya", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("lapor fraud ke mana", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("cara validasi dong", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("gimana cara ngitung MO", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("tanya soal komite lapang", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("info soal client protection", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("bantu aku jelasin PAR dong", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("tolong rinci mekanisme pengaduan", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("mo tu apa sie", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("cp amartha tu apaan", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("pinjaman modal bunga brp persen", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("produk modal tu fitur nya apa aja", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("mekanisme pengaduan amartha gimana", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("satsgas ppks no wa nya berapa", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    Q("syarat pinjam modal apa aja", "knowledge", "KNOWLEDGE", "looks-chitchat"),
    # ── 25 standard KNOWLEDGE (definitive, procedural) ───────────────────
    Q("apa itu Maximum Outstanding", "knowledge", "KNOWLEDGE", "definisi"),
    Q("jelaskan 6 prinsip client protection", "knowledge", "KNOWLEDGE", "definisi"),
    Q("apa saja 8 prinsip client protection", "knowledge", "KNOWLEDGE", "definisi"),
    Q("apa itu mekanisme pengaduan di amartha", "knowledge", "KNOWLEDGE", "definisi"),
    Q("jelaskan tentang business process modal", "knowledge", "KNOWLEDGE", "definisi"),
    Q("apa itu modal cycle 0", "knowledge", "KNOWLEDGE", "definisi"),
    Q("apa itu Komite Lapang", "knowledge", "KNOWLEDGE", "definisi"),
    Q("prosedur onboarding mitra baru", "knowledge", "KNOWLEDGE", "prosedur"),
    Q("cara survey mitra baru", "knowledge", "KNOWLEDGE", "prosedur"),
    Q("proses approval pinjaman gimana", "knowledge", "KNOWLEDGE", "prosedur"),
    Q("tahapan validasi UK apa saja", "knowledge", "KNOWLEDGE", "prosedur"),
    Q("alur pengajuan pinjaman dari awal sampai cair", "knowledge", "KNOWLEDGE", "prosedur"),
    Q("proses komite lapang gimana caranya", "knowledge", "KNOWLEDGE", "prosedur"),
    Q("cara menentukan mitra eligible dapat pinjaman", "knowledge", "KNOWLEDGE", "prosedur"),
    Q("berapa bunga pinjaman modal amartha", "knowledge", "KNOWLEDGE", "fakta"),
    Q("berapa lama pencairan modal biasanya", "knowledge", "KNOWLEDGE", "fakta"),
    Q("MO dihitung dari pendapatan berapa persen", "knowledge", "KNOWLEDGE", "fakta"),
    Q("siapa target pelanggan produk modal", "knowledge", "KNOWLEDGE", "fakta"),
    Q("amartha fokus di daerah mana saja", "knowledge", "KNOWLEDGE", "fakta"),
    Q("berapa minimum pinjaman modal", "knowledge", "KNOWLEDGE", "fakta"),
    Q("produk apa saja yang ada di amartha", "knowledge", "KNOWLEDGE", "definisi"),
    Q("jenis layanan apa saja di amartha", "knowledge", "KNOWLEDGE", "definisi"),
    Q("prinsip-prinsip apa yang harus dipatuhi FO", "knowledge", "KNOWLEDGE", "definisi"),
    Q("bagaimana cara mencegah fraud di lapangan", "knowledge", "KNOWLEDGE", "prosedur"),
    Q("kebijakan soal penagihan amartha gimana", "knowledge", "KNOWLEDGE", "definisi"),
    # ── 10 typo/slang variants of KNOWLEDGE ──────────────────────────────
    Q("mo tu apa sih", "knowledge", "KNOWLEDGE", "typo"),
    Q("cp amartha tu gmna sih cr kerjanya", "knowledge", "KNOWLEDGE", "typo"),
    Q("knp mo 30 persen bs lebih gede ga", "knowledge", "KNOWLEDGE", "typo"),
    Q("produk modal fiturnya apa aja sih", "knowledge", "KNOWLEDGE", "typo"),
    Q("kl mitra udh lewat mo gmn", "knowledge", "KNOWLEDGE", "typo"),
    Q("nmbr bisa ngadu kemana aja sih", "knowledge", "KNOWLEDGE", "typo"),
    Q("gimana cara ngitung maximum outstanding", "knowledge", "KNOWLEDGE", "typo"),
    Q("klo dilecehan di kantor ngapain", "knowledge", "KNOWLEDGE", "typo"),
    Q("amartha fokus di sektor apa", "knowledge", "KNOWLEDGE", "typo"),
    Q("pencairan modal biasanya berapa hari", "knowledge", "KNOWLEDGE", "typo"),
]


@dataclass
class Row:
    q: str
    expected_routing: str
    expected_intent: str
    cluster: str
    regex_intent: str | None
    gate_decision: str
    gate_intent: str | None
    gate_best_cosine: float
    gate_second_cosine: float
    gate_margin: float
    final_intent: str
    gate_latency_ms: float
    verdict: str = ""


async def run() -> int:
    print("=" * 84)
    print(f"INTENT-GATE QUALITY TEST - {len(PROBES)} realistic FO queries")
    print("=" * 84)

    ensure_llamaindex_configured()
    await _compute_centroids()

    rows: list[Row] = []
    for q in PROBES:
        t0 = time.perf_counter()
        regex_intent = rule_classify(q.q)
        gate = await classify_semantic_with_scores(q.q)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if regex_intent is not None and regex_intent != "MALICIOUS":
            final_intent = regex_intent
        elif regex_intent == "MALICIOUS":
            final_intent = "MALICIOUS"
        elif gate.committed is not None:
            final_intent = gate.committed
        else:
            final_intent = "KNOWLEDGE"

        if q.expected_routing == "chitchat":
            if final_intent in ("KNOWLEDGE", "COACHING"):
                verdict = "FN"
            else:
                verdict = "OK"
        else:
            if final_intent in ("KNOWLEDGE", "COACHING"):
                verdict = "OK"
            else:
                verdict = "FP"

        rows.append(Row(
            q=q.q,
            expected_routing=q.expected_routing,
            expected_intent=q.expected_intent,
            cluster=q.cluster,
            regex_intent=regex_intent,
            gate_decision=gate.decision,
            gate_intent=gate.best_intent,
            gate_best_cosine=gate.best_cosine,
            gate_second_cosine=gate.second_cosine,
            gate_margin=gate.margin,
            final_intent=final_intent,
            gate_latency_ms=round(elapsed_ms, 1),
            verdict=verdict,
        ))

    print()
    print(f"{'#':<3}  {'verdict':<4}  {'regex':<12}  {'gate':<5}  {'best':<5}  {'mar':<5}  {'final':<12}  {'q':<48}")
    print("-" * 130)
    for i, r in enumerate(rows, 1):
        print(f"{i:<3}  {r.verdict:<4}  {str(r.regex_intent or '-'):<12}  {r.gate_decision:<5}  {r.gate_best_cosine:5.3f}  {r.gate_margin:5.3f}  {r.final_intent:<12}  {r.q[:46]!r}")

    n = len(rows)
    n_ok = sum(1 for r in rows if r.verdict == "OK")
    n_fp = sum(1 for r in rows if r.verdict == "FP")
    n_fn = sum(1 for r in rows if r.verdict == "FN")
    avg_lat = sum(r.gate_latency_ms for r in rows) / n
    print()
    print("=" * 84)
    print("AGGREGATE")
    print("=" * 84)
    print(f"  Total queries:           {n}")
    print(f"  OK (correct routing):    {n_ok}  ({n_ok/n*100:.0f}%)")
    print(f"  FP (gate over-fired):    {n_fp}  ({n_fp/n*100:.0f}%)  <- CRITICAL: gate committed chitchat on a knowledge query")
    print(f"  FN (chitchat let thru):  {n_fn}  ({n_fn/n*100:.0f}%)  <- minor: missed opportunity, no info lost")
    print(f"  Avg gate+regex latency:  {avg_lat:.1f} ms")

    print()
    print("Per-cluster breakdown:")
    by_cluster: dict[str, list[Row]] = {}
    for r in rows:
        by_cluster.setdefault(r.cluster, []).append(r)
    print(f"  {'cluster':<22}  {'n':>3}  {'OK':>4}  {'FP':>4}  {'FN':>4}")
    for c in sorted(by_cluster):
        crs = by_cluster[c]
        ok = sum(1 for r in crs if r.verdict == "OK")
        fp = sum(1 for r in crs if r.verdict == "FP")
        fn = sum(1 for r in crs if r.verdict == "FN")
        print(f"  {c:<22}  {len(crs):>3}  {ok:>4}  {fp:>4}  {fn:>4}")

    print()
    print("Gate HIT distribution (when gate did commit):")
    hits = [r for r in rows if r.gate_decision == "HIT"]
    hit_intents = Counter(r.gate_intent for r in hits)
    for intent, cnt in hit_intents.most_common():
        print(f"  {intent}: {cnt}")

    fp_rows = [r for r in rows if r.verdict == "FP"]
    print()
    print("=" * 84)
    if fp_rows:
        print(f"FP DETAIL - gate committed chitchat on {len(fp_rows)} knowledge query(ies):")
        for r in fp_rows:
            print(f"  q={r.q!r}")
            print(f"     regex={r.regex_intent}  gate={r.gate_decision}/{r.gate_intent}@{r.gate_best_cosine:.3f}  mar={r.gate_margin:.3f}  -> {r.final_intent}")
            print(f"     expected: {r.expected_intent}  cluster: {r.cluster}")
    else:
        print("FP: 0 - gate never mis-routed a real question as chitchat. SAFE TO DEPLOY.")

    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"intent_gate_quality_70_{int(time.time())}.json"
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": n, "ok": n_ok, "fp": n_fp, "fn": n_fn,
        "ok_pct": round(n_ok/n*100, 1),
        "fp_pct": round(n_fp/n*100, 1),
        "fn_pct": round(n_fn/n*100, 1),
        "avg_gate_latency_ms": round(avg_lat, 1),
        "verdict_safe_to_deploy": n_fp == 0,
        "per_query": [
            {
                "q": r.q,
                "verdict": r.verdict,
                "regex_intent": r.regex_intent,
                "gate_decision": r.gate_decision,
                "gate_intent": r.gate_intent,
                "gate_best_cosine": r.gate_best_cosine,
                "gate_margin": r.gate_margin,
                "final_intent": r.final_intent,
                "cluster": r.cluster,
                "expected_intent": r.expected_intent,
            } for r in rows
        ],
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_path}")
    return 0 if n_fp == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
