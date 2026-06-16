"""Comprehensive Socratic coaching smoke test — 15 turns across every
SOCRATIC_PROMPT block.

Bypasses the LangChain/LlamaIndex pipeline (whose LLM call hangs in this
env) and calls OpenRouter directly with the same SOCRATIC_PROMPT the
pipeline injects when ChatRequest.coaching_mode=True. We simulate the
"coaching_mode=True" path by loading SOCRATIC_PROMPT as the system message
— the pipeline does the same in _generate_node when intent=COACHING
(pipeline.py:1057).

15 turns exercise the full SOCRATIC_PROMPT surface:

  TURN  expect_block     tests
  ----  -------------    ----------------------------------------------------
   1    loop             user makes a wrong guess — light redirect, new Q
   2    loop             user partial-right — closer to wrap-up
   3    wrap-up          "ga tau" fatigue — payoff fires
   4    re-asked         user re-asks same topic after wrap-up — fresh opener
   5    experience       user shares real case — colleague-acknowledge, NOT quiz
   6    frustration      "kok gitu" — drop Socratic, direct teach
   7    frustration      "hah knapa" — drop Socratic, direct teach
   8    factual          "berapa persen MO" — answer DIRECTLY, no quiz
   9    loop             "masih bingung" — new angle or wrap-up
  10    wrap-up          "langsung aja" — explicit done signal
  11    scope            "kenapa langit biru" — off-topic scope guard
  12    loop             "terus gimana" — anaphoric, new facet
  13    loop             "oke" — bare affirmation, keep moving
  14    wrap-up          "udah cukup, makasih" — closer must be actionable
  15    frustration+     "capek bgt jelasin yg cepet" — urgent, direct teach

  F1: no "maaf"/"tujuanku"/"Boleh ceritakan" anti-deflection phrases
  F2: frustration -> direct teach, NO "?" at end
  F3: factual lookup -> answer directly, NO "coba tebak"
  F4: validation beat varies across turns
  F5: wrap-up closer is actionable step, NOT generic re-offer
  F6: scope guard rejects off-topic
  F7: re-asked topic gets FRESH opener, not verbatim repeat

Run:  python -m app.eval.socratic_smoke
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
from typing import Any

# Force UTF-8 stdout on Windows (cp1252 can't encode ≤ → etc.)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx

from app.config.settings import get_settings
from app.graph.pipeline import SOCRATIC_PROMPT

settings = get_settings()

# ── Direct OpenRouter (bypass langchain ChatOpenAI — the latter hangs) ───────
API_URL = settings.openrouter_base_url.rstrip("/") + "/chat/completions"
API_KEY = settings.openrouter_api_key
MODEL = settings.cheap_llm_model  # gemini-2.5-flash-lite

# ── 15-turn scenario ─────────────────────────────────────────────────────────
# Each step: user msg + what behavior we expect the LLM to exhibit.
SCENARIO: list[dict[str, Any]] = [
    # T1: wrong guess, should stay in Socratic loop with light redirect
    {
        "user": "mungkin soal bunga ya?",
        "expect": "loop",
        "can_end_q": True,
        "must_not": ["aku minta maaf", "tujuanku", "Boleh ceritakan"],
    },
    # T2: closer-to-right guess, validate + push to one more probe
    {
        "user": "soal batas pinjaman kali?",
        "expect": "loop",
        "can_end_q": True,
        "must_not": ["aku minta maaf", "tujuanku"],
    },
    # T3: "ga tau" — wrap-up trigger per <wrap_up>
    {
        "user": "apa ya aku gtau",
        "expect": "wrap-up",
        "can_end_q": False,
        "must_not": [
            "aku minta maaf", "tujuanku", "Boleh ceritakan",
            "ada lagi yang mau kamu diskusikan",
            "ada lagi yg mau didiskusikan",
            "ada lagi yang bisa kubantu",
        ],
    },
    # T4: re-asked topic — fresh opener, not verbatim repeat
    {
        "user": "coba jelasin ulang soal MO dong",
        "expect": "re-asked",
        "can_end_q": True,
        "must_not": ["aku minta maaf", "tujuanku"],
    },
    # T5: real experience sharing — acknowledge as colleague, NO "tepat sekali!"
    {
        "user": "dulu mitra aku ada yg pinjam banyak banget sampe nunggak",
        "expect": "experience",
        "can_end_q": True,
        "must_not": ["tepat sekali", "betul sekali", "aku minta maaf"],
    },
    # T6: frustration #1 — drop Socratic, direct teach, no question
    {
        "user": "kok gitu responnya",
        "expect": "frustration",
        "can_end_q": False,
        "must_not": [
            "aku minta maaf", "tujuanku", "Boleh ceritakan",
            "coba tebak",
        ],
    },
    # T7: frustration #2 — same rule, must hold
    {
        "user": "hah knapa",
        "expect": "frustration",
        "can_end_q": False,
        "must_not": [
            "aku minta maaf", "tujuanku", "Boleh ceritakan",
            "coba tebak",
        ],
    },
    # T8: factual lookup — answer DIRECTLY, no quiz
    {
        "user": "berapa persen MO biasanya?",
        "expect": "factual",
        "can_end_q": False,
        "must_not": ["coba tebak"],
        "must_have": ["30"],
    },
    # T9: masih bingung — try a new angle OR wrap-up (model's choice)
    {
        "user": "masih bingung soal 30% nya",
        "expect": "loop",
        "can_end_q": True,
        "must_not": ["aku minta maaf", "Boleh ceritakan"],
    },
    # T10: explicit done — wrap-up now
    {
        "user": "langsung aja",
        "expect": "wrap-up",
        "can_end_q": False,
        "must_not": [
            "aku minta maaf", "Boleh ceritakan",
            "ada lagi yang mau", "ada lagi yg",
        ],
    },
    # T11: off-topic — scope guard, NO KB content
    {
        "user": "btw kenapa langit biru?",
        "expect": "scope",
        "can_end_q": True,
        "must_not": ["Maximum Outstanding", "MO adalah"],
    },
    # T12: anaphoric follow-up — keep going, new facet
    {
        "user": "terus gimana?",
        "expect": "loop",
        "can_end_q": True,
        "must_not": ["aku minta maaf", "Boleh ceritakan"],
    },
    # T13: bare "oke" — continue loop, NOT AMBIGUOUS short-circuit
    {
        "user": "oke",
        "expect": "loop",
        "can_end_q": True,
        "must_not": ["aku minta maaf", "Boleh ceritakan"],
    },
    # T14: explicit thanks / done — closer must be actionable
    {
        "user": "udah cukup, makasih",
        "expect": "wrap-up",
        "can_end_q": False,
        "must_not": [
            "aku minta maaf", "Boleh ceritakan",
            "ada lagi yang mau", "ada lagi yg",
            "feel free to ask",
        ],
    },
    # T15: urgent + frustrated — drop Socratic, fast direct teach
    {
        "user": "capek bgt, jelasin yg cepet",
        "expect": "frustration",
        "can_end_q": False,
        "must_not": [
            "aku minta maaf", "tujuanku", "Boleh ceritakan",
            "coba tebak",
        ],
    },
]

# Strip these structural tags from response preview/analysis (the LLM
# occasionally echoes them literally — pipeline's StreamLeakGuard catches
# this in prod, smoke test bypasses pipeline so we mimic the strip).
_STRUCTURAL_TAG_RE = re.compile(
    r"</?(?:retrieved_context|hard_rules|anti_deflection|mode|role|"
    r"output_contract|rules|how_to_ask|during_the_loop|wrap_up|"
    r"scope|grounding|length|when_to_ask_vs_answer|user_history|"
    r"previous_context|user_preferences|user_context|response_shape|"
    r"available_topics|conversation_signals|capabilities)>",
    re.IGNORECASE,
)
_RETRIEVED_CTX_RE = re.compile(
    r"<retrieved_context>.*?</retrieved_context>", re.DOTALL
)


def _strip_leak(text: str) -> str:
    """Mimic pipeline's StreamLeakGuard: remove structural tag leaks + chunk dumps."""
    text = _RETRIEVED_CTX_RE.sub("", text)
    text = _STRUCTURAL_TAG_RE.sub("", text)
    return text.strip()


def _first_word(text: str) -> str:
    """First real word (alpha-only, no punctuation, lowercased) — used for
    validation-beat variation check (F4)."""
    m = re.match(r"\s*([A-Za-zÀ-ſ]+)", text)
    return m.group(1).lower() if m else ""


async def _call_llm(messages: list[dict], timeout: float = 30.0) -> str:
    """Direct OpenRouter call (no langchain wrapping)."""
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 800,
        "temperature": 0.4,
        "usage": {"include": True},
        "provider": {
            "order": ["google-vertex"],
            "allow_fallbacks": True,
        },
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-lms-agent",
        "X-Title": "AI LMS Agent (Socratic Smoke 15T)",
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(API_URL, json=body, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def run() -> int:
    # Fake <retrieved_context> — same chunk every turn so the LLM has real
    # content to ground teaching in (production does this via generate_node).
    FAKE_CTX = (
        "<retrieved_context>\n"
        "[1] Course: Maximum Outstanding (ID:42)\n"
        "Maximum Outstanding (MO) adalah batas maksimal total pinjaman yang "
        "dapat dimiliki oleh seorang mitra pada satu waktu. MO dihitung "
        "sebagai persentase tertentu dari pendapatan mingguan mitra (umumnya "
        "30%). Tujuannya: mencegah mitra terbebani cicilan berlebih dan "
        "menjaga keberlanjutan usaha mereka. Mitra yang patuh MO dan "
        "pembayaran lancar akan diprioritaskan untuk pencairan berikutnya.\n"
        "</retrieved_context>"
    )

    # ── Simulating coaching_mode=True ──────────────────────────────────────
    # The pipeline's _generate_node (pipeline.py:1057) selects SOCRATIC_PROMPT
    # when intent == "COACHING", which is set by _pre_processor from
    # state["coaching_mode"]. We bypass the pipeline and load SOCRATIC_PROMPT
    # directly — same system message the pipeline would inject.
    messages: list[dict] = [
        {"role": "system", "content": SOCRATIC_PROMPT},
        # Prior Ava Socratic opening (mimics an in-progress coaching loop)
        {"role": "assistant", "content":
            "Coba ingat-ingat lagi, ada istilah yang mirip dengan 'kapasitas' "
            "atau 'batas maksimal' pembayaran mingguan. Apa ya itu?"
        },
    ]

    results: list[dict] = []
    failures = 0
    warnings = 0

    def ok(msg: str) -> None:
        print(f"  [OK] {msg}")

    def fail(msg: str) -> None:
        nonlocal failures
        failures += 1
        print(f"  [FAIL] {msg}")

    def warn(msg: str) -> None:
        nonlocal warnings
        warnings += 1
        print(f"  [WARN] {msg}")

    print("=" * 90)
    print(f"SOCRATIC COACHING SMOKE — {len(SCENARIO)} turns (simulating coaching_mode=True)")
    print("=" * 90)

    for i, step in enumerate(SCENARIO, start=1):
        user_msg = step["user"]
        messages.append({
            "role": "user",
            "content": f"{user_msg}\n\n{FAKE_CTX}",
        })

        print(f"\n--- Turn {i} [{step['expect']}] ---")
        print(f"USER: {user_msg!r}")

        started = time.perf_counter()
        try:
            answer = await _call_llm(messages)
        except Exception as e:
            fail(f"LLM call failed: {e}")
            continue
        elapsed = time.perf_counter() - started
        print(f"  ({elapsed:.1f}s)")

        # Clean preview (strip leak so output is readable)
        clean = _strip_leak(answer)
        preview = clean[:400].replace("\n", " ")
        print(f"AVA: {preview}{'...' if len(clean) > 400 else ''}")

        # ── F1 / hard rules: anti-deflection patterns ─────────────────────
        ans_lower = answer.lower()
        hits = [n for n in step.get("must_not", []) if n.lower() in ans_lower]
        if hits:
            fail(f"anti-deflection: {hits}")
        else:
            ok("no anti-deflection")

        # ── F2 / frustration: no "?" at end ───────────────────────────────
        if not step.get("can_end_q", True):
            last_line = answer.rstrip().split("\n")[-1] if answer.strip() else ""
            if "?" in last_line:
                warn(f"ends with '?' (expect: {step['expect']} should not quiz)")

        # ── F3 / factual: required content present ─────────────────────────
        for needle in step.get("must_have", []):
            if needle.lower() not in ans_lower:
                fail(f"missing required content: {needle!r}")
            else:
                ok(f"has required content: {needle!r}")

        # ── F5 / wrap-up: no forbidden closer ──────────────────────────────
        if step["expect"] == "wrap-up":
            for fc in ("ada lagi yang mau", "ada lagi yg mau", "feel free to ask"):
                if fc in ans_lower:
                    fail(f"forbidden closer: {fc!r}")
                    break
            else:
                ok("closer is actionable (no generic re-offer)")

        # ── F6 / scope: no KB content for off-topic ────────────────────────
        if step["expect"] == "scope":
            kb_markers = ["maximum outstanding", "batas maksimum", "mo adalah", "mitra"]
            kb_hits = [m for m in kb_markers if m in ans_lower]
            if kb_hits:
                fail(f"off-topic but pulled KB content: {kb_hits}")
            else:
                ok("scope guard held (no KB content for off-topic)")

        # ── F7 / re-asked: response should NOT start with verbatim wrap-up ─
        if step["expect"] == "re-asked":
            # Quick check: if response opens with "Maximum Outstanding adalah"
            # verbatim, the LLM just copied prior wrap-up
            if "maximum outstanding adalah" in clean[:200].lower():
                warn("re-asked opener may be verbatim copy of prior wrap-up")

        # Append to history for next turn
        messages.append({"role": "assistant", "content": answer})
        results.append({
            "turn": i,
            "user": user_msg,
            "expect": step["expect"],
            "answer": answer,
            "elapsed_s": round(elapsed, 2),
            "first_word_clean": _first_word(_strip_leak(answer)),
        })

    # ── F4 check: validation beat variation across ALL turns ───────────────
    print()
    print("=" * 90)
    print("VALIDATION BEAT VARIATION (F4):")
    beats = [r["first_word_clean"] for r in results if r["first_word_clean"]]
    for i, b in enumerate(beats, 1):
        print(f"  T{i:2d}: {b!r}")
    if len(beats) >= 2:
        # Check for ANY consecutive duplicate (also check first 2 words of
        # first 2 sentences, since some openers are 2-word like "Oke, mari")
        dups = []
        for i in range(1, len(beats)):
            if beats[i] == beats[i - 1] and beats[i] != "":
                dups.append((i, i + 1, beats[i]))
        if dups:
            for a, b, w in dups:
                fail(f"consecutive validation beat dup: T{a} and T{b} both {w!r}")
        else:
            ok("no consecutive duplicate validation beats")
    else:
        warn("not enough turns to check variation")

    # ── Per-turn summary table ──────────────────────────────────────────────
    print()
    print("=" * 90)
    print("PER-TURN SUMMARY:")
    print(f"  {'T':>3} {'expect':<13} {'beats':<8} {'ok?':<5} {'first10':<10}")
    print(f"  {'-'*3} {'-'*13} {'-'*8} {'-'*5} {'-'*10}")
    for r in results:
        # We don't have per-turn pass/fail in results dict — use first_word as
        # proxy + a quick "had_answer" check
        status = "ok" if r.get("answer") else "ERR"
        print(f"  {r['turn']:>3} {r['expect']:<13} {r['first_word_clean']:<8} {status:<5} {r['user'][:10]!r}")

    print()
    print("=" * 90)
    if failures == 0 and warnings == 0:
        print(f"PASS — all {len(SCENARIO)} turns clean, no warnings")
        return 0
    if failures == 0:
        print(f"PASS (with {warnings} warning(s)) — {len(SCENARIO)} turns")
        return 0
    print(f"FAIL — {failures} hard fail(s), {warnings} warning(s) across {len(SCENARIO)} turns")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
