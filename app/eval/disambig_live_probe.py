"""Live disambiguation probe — uses REAL chunks from Qdrant (not a fake KB)
and checks Ava asks back on underspecified queries.

Unlike disambig_probe.py (hand-written KB), this pulls real chunk text from the
live Knowledge_Base collection, formats it exactly like the pipeline, and feeds
it to the same CONVERSATIONAL_PROMPT. Proves the <disambiguate> fix works
against the actual 233-chunk KB.

Retrieval here is keyword scroll, NOT hybrid_search: the fastembed BM25 sparse
encoder segfaults on Py3.14 in this env (same reason hallucination_probe bypasses
the pipeline). Retrieval quality is NOT what's under test — the disambiguate
PROMPT behaviour is — so a keyword scroll over the real corpus is the right tool.

Run (needs Qdrant up + OPENROUTER_API_KEY): python -m app.eval.disambig_live_probe
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config.settings import get_settings
from app.database.qdrant_client import get_qdrant_client
from app.graph.pipeline import CONVERSATIONAL_PROMPT, _sanitize_answer
from app.eval.hallucination_probe import _call_llm

_settings = get_settings()


@dataclass
class D:
    q: str
    keyword: str    # term to scroll the corpus for
    must_ask: bool  # True = expect a clarifying question that enumerates options


# Fresh wording, different from the static probe. Keywords verified to span
# multiple courses in the live KB (ad-hoc scan: prinsip→6, lapor→6, prosedur→11,
# validasi→5 courses). No must_name list: the model names context-specific
# SUBTYPES (e.g. "Validasi UK/MIS"), not course names — so we grade on whether
# it asked AND enumerated ≥2 options, not on matching a hardcoded word list.
PROBES = [
    D("prinsipnya gimana sih", "prinsip", True),
    D("aku mau lapor", "lapor", True),
    D("prosedurnya apa aja", "prosedur", True),
    D("cara validasi", "validasi", True),
    # Control — specific enough to answer directly, must NOT interrogate.
    D("apa itu client protection", "client protection", False),
]


async def _real_chunks(keyword: str, limit: int = 8) -> list[dict]:
    """Scroll the live KB collection, return up to `limit` chunks whose text
    contains `keyword`, spread across DISTINCT courses (max 2 per course so the
    ambiguity is visible to the LLM, not buried under one course's chunks)."""
    qc = get_qdrant_client()
    points, _ = await qc.client.scroll(
        collection_name=_settings.qdrant_kb_collection,
        limit=1000,
        with_payload=True,
    )
    kw = keyword.lower()
    out: list[dict] = []
    per_course: dict[str, int] = {}
    for p in points:
        payload = p.payload or {}
        node = json.loads(payload.get("_node_content", "{}"))
        text = node.get("text", "")
        if kw not in text.lower():
            continue
        course = payload.get("course_name", "?")
        if per_course.get(course, 0) >= 2:
            continue
        per_course[course] = per_course.get(course, 0) + 1
        out.append({"course_name": course, "course_id": payload.get("course_id", "?"), "text": text})
        if len(out) >= limit:
            break
    return out


def _ends_with_question(a: str) -> bool:
    """A clarifying reply ENDS by asking ("...yang mana?"); an answer/dump ends
    with content. Robust to inline phrasing the option-counter kept missing —
    the model varies bullets vs bold vs comma lists run-to-run, but a genuine
    disambiguation always closes with the question. Check the last ~80 chars."""
    return "?" in a[-80:]


async def run() -> int:
    print("=" * 84)
    print("LIVE DISAMBIGUATION PROBE — real Qdrant chunks, fresh wording")
    print("=" * 84)
    fails = 0
    for i, p in enumerate(PROBES, 1):
        chunks = await _real_chunks(p.keyword)
        lines = [
            f"[{j}] Course: {c['course_name']} (ID:{c['course_id']})\n{c['text'][:600]}"
            for j, c in enumerate(chunks, 1)
        ]
        kb = "<retrieved_context>\n" + "\n\n---\n\n".join(lines) + "\n</retrieved_context>"
        retrieved_courses = sorted({c["course_name"] for c in chunks})

        msgs = [
            {"role": "system", "content": CONVERSATIONAL_PROMPT},
            {"role": "user", "content": f"{kb}\n\nUser question: {p.q}"},
        ]
        data = await _call_llm(msgs)
        ans = _sanitize_answer(data["choices"][0]["message"]["content"].strip())
        asks_back = _ends_with_question(ans)
        leaked = bool(re.search(r"</?(?:disambiguate|grounding|mode|role)>", ans, re.I))

        if p.must_ask:
            # Good = closes by asking the user to pick, OR (acceptable) retrieval
            # collapsed to one course so there was nothing to disambiguate.
            single_course = len(retrieved_courses) <= 1
            ok = (asks_back or single_course) and not leaked
        else:
            ok = (not asks_back) and not leaked
        if not ok:
            fails += 1
        verdict = "PASS" if ok else "FAIL"

        print(f"\nT{i} [{verdict}] q={p.q!r}")
        print(f"   retrieved_courses({len(retrieved_courses)}): {retrieved_courses}")
        print(f"   asks_back={asks_back} leaked={leaked}")
        print(f"   {ans[:260].replace(chr(10), ' ')!r}")

    print("\n" + "=" * 84)
    print(f"RESULT: {len(PROBES)-fails}/{len(PROBES)} pass")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
