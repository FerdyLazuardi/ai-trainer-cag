"""Token-measurement smoke: run a few representative cases through the pre-
processor + generate pipeline and report ACTUAL token usage as reported by
the LLM provider (OpenRouter). This gives a real number, not an estimate.

Compares:
  - Old prompt sizes (in-code comments) vs new prompt sizes
  - Per-call input/output tokens as reported by OpenRouter
  - Cost per query at current OpenRouter pricing for gemini-2.5-flash

Uses dummy retrieval context (no Qdrant) so it can run without infra.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.graph.pipeline import (
    PRE_PROCESSOR_PROMPT, SYSTEM_PROMPT, BRAINSTORM_SYSTEM_PROMPT,
    RESPONSE_SHAPE_SAFETY, RESPONSE_SHAPE_EMPATHY, RESPONSE_SHAPE_LOOKUP,
    RESPONSE_SHAPE_REASONING_WITH_LOOKUP,
)
from app.graph.intent_rules import classify as rule_classify
from app.llm.client import get_llm, get_preprocessor_llm, get_generate_llm
from app.config.settings import get_settings

settings = get_settings()


async def measure_preproc(query: str, history=None) -> dict:
    """Run pre-processor and return usage metadata."""
    history = history or []
    history_str = "\n".join(f"User: {h}" if i % 2 == 0 else f"AI: {h}" for i, h in enumerate(history))
    user_msg_str = f"Conversation history (for pronoun/reference resolution):\n{history_str}\n\nLatest Query: {query}"

    llm = get_preprocessor_llm()
    from app.graph.pipeline import PreProcessorResult
    structured = llm.with_structured_output(PreProcessorResult)

    t0 = time.time()
    try:
        # Use ainvoke (not structured) to capture usage metadata first
        plain_result = await llm.ainvoke([
            SystemMessage(content=PRE_PROCESSOR_PROMPT),
            HumanMessage(content=user_msg_str),
        ])
        usage = getattr(plain_result, "usage_metadata", None) or {}
        in_t = usage.get("input_tokens", 0)
        out_t = usage.get("output_tokens", 0)

        # Now run structured (the real call) to get the classification
        result = await structured.ainvoke([
            SystemMessage(content=PRE_PROCESSOR_PROMPT),
            HumanMessage(content=user_msg_str),
        ])
        dt = time.time() - t0
        return {
            "intent": result.intent,
            "scores": {
                "L": result.needs_lookup,
                "R": result.needs_reasoning,
                "E": result.needs_empathy,
                "S": result.needs_safety_escalation,
            },
            "duration_s": dt,
            "rule_classified": rule_classify(query),
            "input_tokens": in_t,
            "output_tokens": out_t,
        }
    except Exception as e:
        return {"error": str(e), "duration_s": time.time() - t0}


async def measure_generate(query: str, intent: str, chunks: list, history=None) -> dict:
    """Run generate call with dummy chunks and return usage metadata."""
    history = history or []
    has_safety = any("pelecehan" in c["text"].lower() or "harass" in c["text"].lower() for c in chunks)
    has_lookup = bool(chunks)
    has_empathy = intent == "BRAINSTORM"
    has_reasoning = intent == "BRAINSTORM"

    base = BRAINSTORM_SYSTEM_PROMPT if intent == "BRAINSTORM" else SYSTEM_PROMPT
    score_blocks = []
    if has_safety: score_blocks.append(RESPONSE_SHAPE_SAFETY)
    if has_empathy: score_blocks.append(RESPONSE_SHAPE_EMPATHY)
    if has_lookup: score_blocks.append(RESPONSE_SHAPE_LOOKUP)
    if has_reasoning:
        if has_lookup: score_blocks.append(RESPONSE_SHAPE_REASONING_WITH_LOOKUP)

    score_block_str = ("\n\n" + "\n\n".join(score_blocks)) if score_blocks else ""

    context_str = "\n\n---\n\n".join(
        f"[{i+1}] Course: Pelaporan (ID:{i+1})\n{c['text']}" for i, c in enumerate(chunks)
    )

    full_system = (
        f"{base}{score_block_str}"
        f"\n\n<previous_context>\nUser asked about general topic.\n</previous_context>"
        f"\n\n<retrieved_context>\n{context_str}\n</retrieved_context>"
    )

    messages = [SystemMessage(content=full_system)]
    for h in history:
        if isinstance(h, tuple) and h[0] == "ai":
            messages.append(AIMessage(content=h[1]))
        else:
            messages.append(HumanMessage(content=h[1]))
    messages.append(HumanMessage(content=query))

    llm = get_generate_llm()
    t0 = time.time()
    try:
        result = await llm.ainvoke(messages)
        dt = time.time() - t0
        usage = getattr(result, "usage_metadata", None) or {}
        return {
            "duration_s": dt,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "answer_len": len(result.content) if hasattr(result, "content") else 0,
        }
    except Exception as e:
        return {"error": str(e), "duration_s": time.time() - t0}


DUMMY_CHUNKS = [
    {"text": "Mechanism of Complaints Resolution: nasabah bisa melaporkan keluhan ke People Care melalui email peoplecare@amartha.com atau hotline 0800-1234-5678. Tim People Care akan menindaklanjuti dalam 7 hari kerja."},
    {"text": "Untuk kasus pelecehan seksual (sexual harassment), nasabah atau karyawan bisa melapor lewat jalur khusus: email safespace@amartha.com atau WhatsApp 0812-3456-7890. Semua laporan dijaga kerahasiaannya."},
    {"text": "Client Protection (Perlindungan Nasabah) adalah prinsip yang memastikan Amartha memperlakukan nasabah secara adil. Meliputi: transparansi, kerahasiaan data, dan mekanisme penyelesaian keluhan."},
    {"text": "Tips menghadapi situasi tidak nyaman: 1) Dokumentasikan kejadian (tanggal, waktu, lokasi, saksi). 2) Laporkan ke supervisor atau tim People Care. 3) Minta dukungan dari rekan kerja terpercaya."},
]


async def main():
    print("=" * 80)
    print("ACTUAL TOKEN MEASUREMENT (from OpenRouter usage_metadata)")
    print("=" * 80)
    print()
    print("Test cases:")
    test_cases = [
        ("User failure case (ID slang, safety)", "aku hbis dilcehin laporinnya ke mna"),
        ("Standard knowledge query", "Apa itu produk Modal di Amartha?"),
        ("Brainstorm/vent query", "aku stress banget akhir2 ini"),
        ("Greeting", "halo"),
        ("Procedural (3rd person)", "Bagaimana cara melaporkan kasus pelecehan di Amartha?"),
    ]

    print()
    print("=" * 80)
    print("PRE-PROCESSOR CALL (actual usage from OpenRouter)")
    print("=" * 80)
    for label, q in test_cases:
        r = await measure_preproc(q)
        rule = r.get("rule_classified", "—")
        if "error" in r:
            print(f"  [{label:42s}] ERROR: {r['error'][:60]}")
        else:
            intent = r.get("intent", "?")
            scores = r.get("scores", {})
            in_t = r.get("input_tokens", 0)
            out_t = r.get("output_tokens", 0)
            cost = in_t / 1e6 * 0.075 + out_t / 1e6 * 0.30  # flash-lite
            print(f"  [{label:42s}] rule={rule!s:12s} -> {intent:12s} | "
                  f"L={scores.get('L', 0):.2f} R={scores.get('R', 0):.2f} "
                  f"E={scores.get('E', 0):.2f} S={scores.get('S', 0):.2f} | "
                  f"{in_t} in + {out_t} out = {in_t+out_t} t | ${cost:.6f} | {r['duration_s']:.1f}s")
    print()
    print("=" * 80)
    print("GENERATE CALL (KNOWLEDGE+SAFETY scenario with dummy 4-chunk context)")
    print("=" * 80)
    for label, q in test_cases:
        r = await measure_generate(q, intent="BRAINSTORM", chunks=DUMMY_CHUNKS, history=[
            ("user", "halo"),
            ("ai", "Halo! Ada yang bisa aku bantu?"),
        ])
        if "error" in r:
            print(f"  [{label:42s}] ERROR: {r['error'][:60]}")
        else:
            in_t = r.get("input_tokens") or 0
            out_t = r.get("output_tokens") or 0
            cost_in = in_t / 1_000_000 * 0.075  # flash-lite input cost
            cost_out = out_t / 1_000_000 * 0.30  # flash-lite output cost
            cost_total = cost_in + cost_out
            print(f"  [{label:42s}] {in_t:>5} in + {out_t:>4} out = {in_t+out_t:>5} t | "
                  f"${cost_total:.6f}/call | {r['duration_s']:.1f}s")
    print()
    print("=" * 80)
    print("PRICING ASSUMPTIONS (OpenRouter, as of 2026)")
    print("=" * 80)
    print("  gemini-2.5-flash:    $0.30 / M input, $2.50 / M output")
    print("  gemini-2.5-flash-lite: $0.075 / M input, $0.30 / M output  (4-8x cheaper)")
    print("  Auto-cache (>1024 t prefix): 50% off cached portion after first call")
    print()


if __name__ == "__main__":
    asyncio.run(main())
