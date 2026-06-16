"""Disambiguation probe — does Ava ASK BACK on underspecified queries instead
of dumping all sets or guessing one?

Tests the <disambiguate>-vs-<grounding> precedence fix in pipeline.py. The KB
here deliberately carries THREE distinct "prinsip" sets (Fraud / Client
Protection / Penagihan) and THREE distinct report channels, so a bare "prinsip"
or "cara lapor" is genuinely ambiguous and the prompt MUST ask one clarifying
question naming the candidates — not list everything, not pick one.

Run: python -m app.eval.disambig_probe
"""
from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.graph.pipeline import CONVERSATIONAL_PROMPT, _sanitize_answer, _sanitize_answer
from app.eval.hallucination_probe import _call_llm  # reuse live OpenRouter caller

# KB with MULTIPLE distinct sets behind the same bare term.
KB = """<retrieved_context>
[1] Course: Pencegahan Fraud
5 Prinsip Non-Negotiable Pencegahan Fraud: (1) zero tolerance, (2) ...

[2] Course: Client Protection
8 Prinsip Client Protection: Prinsip 1 Appropriate Product Design, ...
Prinsip 3 Transparency, Prinsip 4 Responsible Pricing, ...

[3] Course: Proses Penagihan
3 Prinsip Proses Penagihan: jaga mitra lancar, cegah keterlambatan, ...

[4] Course: Leadership
Prinsip Leadership Amartha: lead by example, ...

[5] Course: Pelaporan
Lapor kendala lapangan ke Branch Manager. Lapor fraud ke kanal Whistleblower.
Lapor pelecehan ke People Care (Satgas PPKS).
</retrieved_context>"""


@dataclass
class D:
    q: str
    must_ask: bool          # True = expect a clarifying question
    candidates: list[str]   # ≥2 of these named => it disambiguated correctly


PROBES = [
    D("prinsip", True, ["Fraud", "Client Protection", "Penagihan"]),
    D("cara lapor", True, ["kendala", "fraud", "pelecehan"]),
    D("prinsip 3", True, ["Fraud", "Client Protection", "Penagihan"]),
    # Control: unambiguous → must NOT ask, must answer.
    D("apa itu prinsip client protection", False, []),
]


def _is_question(a: str) -> bool:
    return "?" in a


def _names_candidates(a: str, cands: list[str], n: int = 2) -> int:
    al = a.lower()
    return sum(1 for c in cands if c.lower() in al)


async def run() -> int:
    print("=" * 80)
    print("DISAMBIGUATION PROBE — bare terms with multiple KB sets")
    print("=" * 80)
    fails = 0
    for i, p in enumerate(PROBES, 1):
        msgs = [
            {"role": "system", "content": CONVERSATIONAL_PROMPT},
            {"role": "user", "content": f"{KB}\n\nUser question: {p.q}"},
        ]
        data = await _call_llm(msgs)
        raw = data["choices"][0]["message"]["content"].strip()
        ans = _sanitize_answer(raw)  # same strip the pipeline applies
        asked = _is_question(ans)
        named = _names_candidates(ans, p.candidates)
        leaked = bool(re.search(r"</?(?:disambiguate|grounding|mode|role)>", ans, re.I))

        if p.must_ask:
            ok = asked and named >= 2 and not leaked
            verdict = "PASS" if ok else "FAIL"
        else:
            ok = (not asked or named == 0) and not leaked
            verdict = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1

        print(f"\nT{i} [{verdict}] q={p.q!r}  asked={asked} named={named}/{len(p.candidates)} leaked={leaked}")
        print(f"   {ans[:240].replace(chr(10), ' ')!r}")

    print("\n" + "=" * 80)
    print(f"RESULT: {len(PROBES)-fails}/{len(PROBES)} pass")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
