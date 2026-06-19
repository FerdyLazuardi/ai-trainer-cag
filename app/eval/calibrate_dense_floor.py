"""Calibrate kb_min_dense_score (the NOT-FOUND dense floor in _route_after_rag).

Method (data-driven, not eyeballed):
1. IN-SCOPE labels self-derive from Qdrant: for every course actually ingested,
   probe "apa itu {course}" + "jelasin {course}". Their pool_max_dense is the
   in-scope distribution — the floor must NOT reject these.
2. OFF-SCOPE labels are a held-out list: topics NOT in the KB (BMDP /
   Basic Leadership, still un-ingested) + clearly out-of-domain queries. Their
   pool_max_dense is the off-scope distribution — the floor SHOULD reject these.
3. Sweep candidate floors; at each, count in-scope false-rejects and off-scope
   false-accepts. Recommend the highest floor with ZERO in-scope false-reject
   (protect real questions first), and report the cleanest-separation floor too.

Run: python -m app.eval.calibrate_dense_floor
Re-run after KB growth or an embedding-model swap. Does NOT auto-apply — prints
a recommendation; the operator edits settings.kb_min_dense_score.
"""
from __future__ import annotations

import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config.settings import get_settings
from app.database.qdrant_client import get_qdrant_client
from app.retrieval.hybrid_retriever import hybrid_search

_s = get_settings()

# Topics known to be ABSENT from the KB (un-ingested BMDP / Basic Leadership
# material) plus clearly out-of-domain queries. Update if the KB changes.
OFF_SCOPE_QUERIES = [
    "apa itu BMDP",
    "4 gaya kepemimpinan BMDP apa aja",
    "apa itu basic leadership",
    "apa itu flexible leadership",
    "apa itu sales funnel",
    "apa itu early warning signal",
    "apa itu conversion tracker",
    "metode GROW coaching itu apa",
    "apa itu bitcoin",
    "resep nasi goreng",
    "siapa presiden indonesia",
    "cuaca jakarta hari ini gimana",
]


async def _course_names() -> list[str]:
    q = get_qdrant_client()
    offset = None
    names: set[str] = set()
    while True:
        pts, offset = await q.client.scroll(
            collection_name=_s.qdrant_kb_collection, limit=256,
            offset=offset, with_payload=True, with_vectors=False,
        )
        for p in pts:
            pl = p.payload or {}
            cn = pl.get("course_name") or (pl.get("metadata") or {}).get("course_name")
            if cn:
                names.add(cn)
        if offset is None:
            break
    return sorted(names)


async def _pool_dense(query: str) -> float:
    r = await hybrid_search(query=query, top_k=_s.final_top_k)
    return float(r.pool_max_dense or 0.0)


async def main():
    courses = await _course_names()
    in_queries = [f"apa itu {c}" for c in courses] + [f"jelasin {c}" for c in courses]

    in_scores = await asyncio.gather(*[_pool_dense(q) for q in in_queries])
    off_scores = await asyncio.gather(*[_pool_dense(q) for q in OFF_SCOPE_QUERIES])

    in_scores = sorted(in_scores)
    off_scores = sorted(off_scores)

    print(f"\nIN-SCOPE  ({len(in_scores)} probes over {len(courses)} courses): "
          f"min={in_scores[0]:.3f} p10={in_scores[len(in_scores)//10]:.3f} "
          f"median={in_scores[len(in_scores)//2]:.3f} max={in_scores[-1]:.3f}")
    print(f"OFF-SCOPE ({len(off_scores)} probes): "
          f"min={off_scores[0]:.3f} median={off_scores[len(off_scores)//2]:.3f} "
          f"max={off_scores[-1]:.3f}")

    print("\nOFF-SCOPE detail (these SHOULD be rejected):")
    for q, sc in sorted(zip(OFF_SCOPE_QUERIES, off_scores if False else [None]*len(OFF_SCOPE_QUERIES))):
        pass  # printed below from the unsorted pairing
    # re-run pairing for readable detail
    pairs = sorted(zip(await asyncio.gather(*[_pool_dense(q) for q in OFF_SCOPE_QUERIES]),
                       OFF_SCOPE_QUERIES), reverse=True)
    for sc, q in pairs:
        print(f"  {sc:.3f}  {q!r}")

    print(f"\n{'floor':>6} {'in_reject':>10} {'off_accept':>11}  verdict")
    best_clean = None
    for f in [round(0.35 + 0.01 * i, 2) for i in range(31)]:  # 0.35..0.65
        in_rej = sum(1 for s in in_scores if s < f)
        off_acc = sum(1 for s in off_scores if s >= f)
        if in_rej == 0 and best_clean is None and off_acc < len(off_scores):
            pass
        if in_rej == 0:
            best_clean = f  # highest floor with zero in-scope rejects
        mark = ""
        if in_rej == 0 and off_acc == 0:
            mark = "  <- perfect separation"
        print(f"{f:>6} {in_rej:>10} {off_acc:>11}{mark}")

    cur = _s.kb_min_dense_score
    print(f"\nCURRENT floor = {cur}")
    if best_clean is not None:
        in_rej_cur = sum(1 for s in in_scores if s < cur)
        off_acc_cur = sum(1 for s in off_scores if s >= cur)
        off_acc_best = sum(1 for s in off_scores if s >= best_clean)
        print(f"  at current {cur}: in-scope rejected={in_rej_cur}, off-scope leaked={off_acc_cur}")
        print(f"RECOMMEND floor = {best_clean}  "
              f"(highest with 0 in-scope false-reject; off-scope leaked={off_acc_best})")
    else:
        print("  NO floor achieves zero in-scope false-reject — distributions overlap; "
              "single dense floor insufficient, consider a rerank/margin second signal.")


asyncio.run(main())
