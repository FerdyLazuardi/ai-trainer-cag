"""
Retrieval evaluation harness — measures hybrid_search quality against the LIVE
Knowledge Base.

WHY THIS IS A STANDALONE SCRIPT (not in tests/):
    It requires a live Qdrant + the embedding provider (bge-m3 via OpenRouter).
    The pytest suite is deterministic and infra-free, so a network-dependent
    retrieval eval would break it in CI/dev without infra. This mirrors the
    existing app/eval/ harnesses (run_dataset.py, run_mentor_eval.py).

WHAT IT MEASURES:
    1. Hit@1 / Hit@3   — is the expected COURSE in the top-1 / top-3 results?
    2. MRR             — mean reciprocal rank of the first correct-course chunk.
    3. Title hit       — when a case names a specific chunk, did it surface?
    4. NOT-FOUND gate  — off-scope queries MUST fail the gate; in-scope MUST pass
                         (mirrors _route_after_rag's KNOWLEDGE OR-gate).

DETERMINISM:
    bge-m3 embeddings + BM25 + a fixed HNSW index are deterministic for a fixed
    KB, so results are reproducible and valid as a regression gate.

RUN:
    .\\.venv\\Scripts\\python.exe -m app.eval.run_retrieval_eval
    (exit code 0 = all gates passed, 1 = regression — CI-friendly)
"""
import asyncio
import sys
from dataclasses import dataclass, field

from loguru import logger

from app.config.settings import get_settings
from app.config.embedding_config import ensure_llamaindex_configured
from app.retrieval.hybrid_retriever import hybrid_search

s = get_settings()


# ─── Golden set (grounded in the real KB: 5 courses, 42 chunks) ───────────────
@dataclass
class Case:
    query: str
    category: str
    expected_course: str | None          # None = off-scope → gate MUST reject
    expect_title_contains: str | None = None  # optional: a chunk heading substring
    note: str = ""


# Course-name constants (verbatim as stored in chunk.metadata["course_name"]).
CP = "Client Protection"
AH = "Anti-Harassment"
PK = "Product Knowledge Amartha"
DM = "AmarthaLink & Digital Mindset"
CO = "Company Amartha"

GOLDEN: list[Case] = [
    # A. Direct factual lookup — unambiguous; expect the right course at rank 1.
    Case("apa itu Client Protection", "factual", CP, "Tentang Client Protection"),
    Case("8 prinsip client protection apa saja", "factual", CP, "8 Prinsip Client Protection"),
    Case("apa itu anti harassment di Amartha", "factual", AH),
    Case("produk apa saja yang ada di Amartha", "factual", PK),
    Case("visi dan misi Amartha", "factual", CO),
    Case("apa itu digital mindset", "factual", DM, "Digital Mindset"),

    # B. Exact-term / proper nouns — BM25/sparse should fire hard (sparse >= 1.0).
    Case("Poket", "exact-term", PK, "Poket"),
    Case("Celengan", "exact-term", PK, "Celengan"),
    Case("AmarthaFin", "exact-term", PK, "AmarthaFin"),
    Case("Responsible Pricing", "exact-term", CP, "Responsible Pricing"),
    Case("Governance and HR", "exact-term", CP, "Governance and HR"),

    # C. Paraphrase / semantic — low lexical overlap; leans on dense (bge-m3).
    Case("gimana kalau aku diganggu sama rekan kerja", "paraphrase", AH,
         note="no literal 'harassment' term — dense must bridge"),
    Case("cara menjaga kerahasiaan data nasabah", "paraphrase", CP,
         note="maps to Privacy of Client Data (Prinsip 6)"),
    Case("pinjaman modal buat usaha kecil", "paraphrase", PK,
         note="maps to Produk Modal"),

    # D. Cross-course disambiguation — shared vocab; must pick the RIGHT course.
    Case("prinsip transparansi informasi ke nasabah", "disambiguation", CP,
         note="'transparansi' could pull other courses; must be CP Prinsip 3"),
    Case("sanksi bagi pelaku pelecehan", "disambiguation", AH,
         note="must be Anti-Harassment, not Client Protection"),

    # E. Off-scope — NOT-FOUND gate MUST reject (no KB grounding exists).
    Case("cuaca hari ini di Jakarta gimana", "off-scope", None),
    Case("resep nasi goreng yang enak", "off-scope", None),
    Case("siapa presiden Amerika Serikat", "off-scope", None),
]


def gate_passes(pool_max_dense: float, pool_max_sparse: float, dense_ok: bool) -> bool:
    """Mirror _route_after_rag's NOT-FOUND gate (pipeline.py).

    Normal operation: DENSE is the mandatory floor (sparse NOT consulted — raw
    BM25 doesn't separate scope on a small KB). When dense degraded (embedding
    outage), fail OPEN on sparse alone (C5 window).
    """
    if not dense_ok:
        return pool_max_sparse >= s.kb_min_sparse_score   # degraded: fail-open
    return pool_max_dense >= s.kb_min_dense_score          # dense mandatory


def _norm(x: str) -> str:
    return (x or "").strip().casefold()


@dataclass
class CaseResult:
    case: Case
    top_courses: list[str] = field(default_factory=list)
    hit_rank: int | None = None          # 1-based rank of first expected-course chunk
    title_hit: bool = False
    pool_max_dense: float = 0.0
    pool_max_sparse: float = 0.0
    dense_ok: bool = True
    gate: bool = False
    top_rows: list[str] = field(default_factory=list)  # human-readable top-k lines
    passed: bool = False
    reason: str = ""


async def _eval_case(case: Case) -> CaseResult:
    res = await hybrid_search(query=case.query, top_k=s.final_top_k)
    r = CaseResult(case=case)
    r.pool_max_dense = round(res.pool_max_dense, 4)
    r.pool_max_sparse = round(res.pool_max_sparse, 4)
    r.dense_ok = res.dense_available
    r.gate = gate_passes(res.pool_max_dense, res.pool_max_sparse, res.dense_available)

    for rank, ch in enumerate(res.chunks, start=1):
        course = ch.metadata.get("course_name", "?")
        head = (ch.text or "").strip().split("\n", 1)[0][:60]
        r.top_courses.append(course)
        r.top_rows.append(
            f"      #{rank} d={ch.dense_score:.3f} sp={ch.sparse_score:6.2f} "
            f"hy={ch.hybrid_score:.3f} | {course} | {head}"
        )
        if case.expected_course and _norm(course) == _norm(case.expected_course):
            if r.hit_rank is None:
                r.hit_rank = rank
            if case.expect_title_contains and _norm(case.expect_title_contains) in _norm(ch.text):
                r.title_hit = True

    # Pass/fail logic per category.
    if case.expected_course is None:
        # Off-scope: success = gate REJECTS (no false grounding).
        r.passed = not r.gate
        r.reason = "gate correctly rejected" if r.passed else "FALSE-POSITIVE: gate passed an off-scope query"
    else:
        in_top3 = r.hit_rank is not None
        gate_ok = r.gate
        title_ok = (case.expect_title_contains is None) or r.title_hit
        r.passed = in_top3 and gate_ok and title_ok
        if r.passed:
            r.reason = f"hit@{r.hit_rank}" + (" +title" if case.expect_title_contains else "")
        else:
            probs = []
            if not in_top3:
                probs.append(f"expected '{case.expected_course}' NOT in top-{s.final_top_k}")
            if not gate_ok:
                probs.append("gate FAILED on in-scope query")
            if not title_ok:
                probs.append(f"title '{case.expect_title_contains}' missing")
            r.reason = "; ".join(probs)
    return r


def _print_report(results: list[CaseResult]) -> bool:
    in_scope = [r for r in results if r.case.expected_course is not None]
    off_scope = [r for r in results if r.case.expected_course is None]

    print("\n" + "=" * 78)
    print("RETRIEVAL EVAL — per-case results")
    print("=" * 78)
    cats: dict[str, list[CaseResult]] = {}
    for r in results:
        cats.setdefault(r.case.category, []).append(r)

    for cat, rows in cats.items():
        print(f"\n── {cat} ──")
        for r in rows:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  [{mark}] {r.case.query!r}")
            print(f"      → {r.reason}   "
                  f"(gate={'PASS' if r.gate else 'reject'}, "
                  f"pool_dense={r.pool_max_dense}, pool_sparse={r.pool_max_sparse}"
                  + ("" if r.dense_ok else ", DENSE-DEGRADED") + ")")
            for line in r.top_rows:
                print(line)
            if r.case.note:
                print(f"      note: {r.case.note}")

    # ── Aggregate metrics ──
    n_in = len(in_scope)
    hit1 = sum(1 for r in in_scope if r.hit_rank == 1)
    hit3 = sum(1 for r in in_scope if r.hit_rank is not None)
    mrr = sum((1.0 / r.hit_rank) for r in in_scope if r.hit_rank) / n_in if n_in else 0.0
    title_cases = [r for r in in_scope if r.case.expect_title_contains]
    title_hits = sum(1 for r in title_cases if r.title_hit)
    gate_in_ok = sum(1 for r in in_scope if r.gate)
    gate_off_ok = sum(1 for r in off_scope if not r.gate)

    print("\n" + "=" * 78)
    print("AGGREGATE METRICS")
    print("=" * 78)
    print(f"  In-scope cases:        {n_in}")
    print(f"  Hit@1:                 {hit1}/{n_in}  ({hit1/n_in*100:.0f}%)")
    print(f"  Hit@3:                 {hit3}/{n_in}  ({hit3/n_in*100:.0f}%)")
    print(f"  MRR:                   {mrr:.3f}")
    if title_cases:
        print(f"  Title-specific hit:    {title_hits}/{len(title_cases)}  ({title_hits/len(title_cases)*100:.0f}%)")
    print(f"  Gate pass (in-scope):  {gate_in_ok}/{n_in}  (want {n_in})")
    print(f"  Gate reject (off):     {gate_off_ok}/{len(off_scope)}  (want {len(off_scope)})")

    # ── Pass/fail thresholds (regression gate) ──
    THRESH = {
        "hit3_rate": 1.0,     # every in-scope query must surface its course in top-3
        "hit1_rate": 0.80,    # at least 80% rank-1
        "mrr": 0.85,
        "gate_in": 1.0,       # all in-scope must clear the NOT-FOUND gate
        "gate_off": 1.0,      # all off-scope must be rejected
    }
    checks = {
        "Hit@3 == 100%": (hit3 / n_in) >= THRESH["hit3_rate"],
        f"Hit@1 >= {THRESH['hit1_rate']:.0%}": (hit1 / n_in) >= THRESH["hit1_rate"],
        f"MRR >= {THRESH['mrr']}": mrr >= THRESH["mrr"],
        "all in-scope pass gate": gate_in_ok == n_in,
        "all off-scope rejected": gate_off_ok == len(off_scope),
    }
    print("\n" + "-" * 78)
    all_ok = True
    for name, ok in checks.items():
        print(f"  [{'OK ' if ok else 'XX '}] {name}")
        all_ok = all_ok and ok

    n_pass = sum(1 for r in results if r.passed)
    print("-" * 78)
    print(f"  CASES: {n_pass}/{len(results)} passed   |   VERDICT: {'PASS' if all_ok and n_pass == len(results) else 'FAIL'}")
    print("=" * 78)
    return all_ok and n_pass == len(results)


async def main() -> int:
    # Windows consoles default to cp1252 and choke on the box-drawing / arrow
    # glyphs in the report. Force UTF-8 so the harness runs cross-platform.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    logger.remove()  # quiet the retriever's debug logs for a clean report
    logger.add(sys.stderr, level="WARNING")
    ensure_llamaindex_configured()
    print(f"Running retrieval eval: {len(GOLDEN)} cases against "
          f"'{s.qdrant_kb_collection}' (final_top_k={s.final_top_k}, "
          f"gate: dense>={s.kb_min_dense_score} mandatory, degraded->sparse>={s.kb_min_sparse_score})")
    results = [await _eval_case(c) for c in GOLDEN]
    ok = _print_report(results)

    # ── Hard gates + margin report (regression guard) ──
    off = [r for r in results if r.case.expected_course is None]
    ins = [r for r in results if r.case.expected_course is not None]
    leaked = [r.case.query for r in off if r.gate]
    rejected = [r.case.query for r in ins if not r.gate]
    if off:
        max_off = max(r.pool_max_dense for r in off)
        min_in = min(r.pool_max_dense for r in ins) if ins else 0.0
        D = s.kb_min_dense_score
        print("\n" + "-" * 78)
        print(f"  DENSE GATE D={D} | off_ceiling={max_off:.3f} (margin {D-max_off:+.3f}) "
              f"| in_floor={min_in:.3f} (margin {min_in-D:+.3f})")
        if not (max_off < D < min_in):
            print("  WARN: D is OUTSIDE the separable gap — recalibrate (see settings comment).")
        elif (D - max_off) < 0.05 or (min_in - D) < 0.05:
            print("  WARN: gate margin < 0.05 — sample is thin, expand the eval set.")
    if leaked:
        print(f"  XX off-scope LEAKED the gate: {leaked}")
    if rejected:
        print(f"  XX in-scope REJECTED by the gate: {rejected}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
