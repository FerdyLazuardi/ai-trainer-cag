"""Deterministic intent pre-classifier.

Fast, regex-based rules that handle the highest-confidence intent cases
WITHOUT calling the LLM. The LLM-based classifier in `_pre_processor`
remains the fallback for everything that doesn't match a rule here —
this module just short-circuits the unambiguous 60-70% of traffic.

Why: the LLM classifier (Gemini Flash Lite) was bouncing on edge cases
when the prompt grew to cover every intent. Each prompt iteration that
fixed one case broke another. Pulling deterministic patterns out lets us:
  1. Eliminate flakiness for things that were never genuinely ambiguous
     (math, single emoji, "kamu siapa").
  2. Save LLM cost + latency on those calls.
  3. Keep the LLM prompt small + focused on truly ambiguous middle cases.

Design:
  - Each rule is a function (text, low) -> Optional[Intent]
  - Rules check IN ORDER, first match wins.
  - Rules are conservative — false negatives (no match) fall through to
    the LLM; false positives (wrong rule fires) are user-visible bugs.
  - No history scanning here. History-binding stays in the LLM path
    where it can reason about prior turns.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

Intent = Literal["GREETING", "AMBIGUOUS", "OFF_SCOPE"]

# ── Rule 1: pure punctuation / filler ────────────────────────────────────────
# Matches messages that are ONLY punctuation, whitespace, or a single short
# emoji-like token. Things like "??", "...", ".", "??!", "🤔", "🙏".
_PURE_PUNCT_RE = re.compile(r"^[\s\W_]{1,5}$", re.UNICODE)

# ── Rule 2: math expressions ─────────────────────────────────────────────────
# Pure arithmetic queries: "2+2", "5 x 10 berapa", "100/4", "berapa 7-3".
# Conservative: requires a digit-operator-digit pattern with WORD BOUNDARIES
# on both sides — so "[218cb0-3]" or "user_id123-4" do NOT trigger. Without
# the \b anchors the regex was greedy and ate alphanumeric suffixes.
_MATH_OP_RE = re.compile(r"\b\d+\s*[+\-*x×/÷]\s*\d+\b")

# ── Rule 3: off-Amartha topical keywords ─────────────────────────────────────
# Words that are unambiguously about NON-Amartha topics. Each entry is a
# whole-word match (space-padded) to avoid catching substrings of legitimate
# Amartha terms.
#
# CRITICAL — keep this list to terms that are NEVER part of a legitimate
# Amartha question. Do NOT add bank/e-wallet/competitor names, "mandiri"
# (= Indonesian "independent", core to Amartha's mission), "menteri" (govt
# partnerships are in-scope), or "masak" (slang "really?!"). Those produce
# false OFF_SCOPE blocks on real questions like "apakah Amartha kerja sama
# BRI", "cara jadi mitra mandiri", "integrasi gopay". Genuine competitor
# questions ("bunga deposito BCA") are still caught by the LLM classifier's
# STEP 1 — which, unlike this regex, can tell an in-scope partnership question
# from an off-scope one.
_OFF_SCOPE_KEYWORDS = (
    # Weather
    "cuaca", "weather", "hujan deras",
    # News / current affairs
    "berita terkini", "berita hari ini", "headline berita", "breaking news",
    # Food / recipes
    "resep", "recipe", "menu makan", "restoran",
    # Sports
    "skor bola", "liga inggris", "premier league", "world cup", "piala dunia",
    # Politics / world facts (national leaders — never an Amartha topic)
    "siapa presiden", "presiden indonesia", "ibu kota negara",
    # Celebrity / entertainment
    "selebriti", "selebgram", "gosip artis",
)

# ── Rule 4: bot-identity / app-purpose questions ─────────────────────────────
# Exact-ish matches for "what/who is this" addressed to the bot/app.
# CRITICAL: every phrase MUST anchor to an explicit bot/app reference
# (kamu/lu/lo/apps/bot/aplikasi/who-are-you). Do NOT add bare phrases like
# "ini apa" or "what is this" — those are greedy and wrongly swallow legitimate
# topic lookups ("modal ini apa", "celengan ini apa", "BP ini apa sih"), routing
# them to a self-introduction instead of an answer. Anything genuinely
# ambiguous ("ini apa?" alone) must fall through to the LLM, which has context.
_IDENTITY_PHRASES = (
    # ID — anchored to bot/app reference
    "kamu siapa", "lu siapa", "lo siapa", "elu siapa", "siapa kamu",
    "ini apps apa", "ini apps buat apa", "ini aplikasi apa", "ini bot apa",
    "kamu bot apa", "kamu apa sih", "lu apa sih", "perkenalkan diri",
    # EN
    "who are you", "what are you", "what is this app",
    "what is this bot", "introduce yourself",
)

# ── Rule 5: greeting prefixes ────────────────────────────────────────────────
# Already in the existing pipeline; centralised here for one-stop classifier.
_GREETING_PREFIXES = (
    "halo", "hai", "hi ", "hi,", "hi.", "hey", "hello",
    "pagi", "siang", "sore", "malam",
    "good morning", "good afternoon", "good evening",
    "selamat", "test ", "test,", "test.",
)

# Greeting/filler tokens used to decide whether a message is a PURE greeting or
# a greeting that PREFIXES a real question ("pagi, modal itu apa"). Only the
# former should short-circuit to GREETING; the latter must reach the LLM so the
# actual question gets answered.
_GREETING_WORDS = {
    "halo", "hai", "hi", "hey", "hello", "pagi", "siang", "sore", "malam",
    "selamat", "datang", "test", "good", "morning", "afternoon", "evening",
}
# Conversational fluff that may trail a greeting without making it a question.
_GREETING_FLUFF = {
    "ya", "yaa", "dong", "donk", "kak", "ka", "min", "bang", "gan", "ges",
    "gaes", "nih", "ni", "aja", "deh", "sih", "kok", "aku", "saya", "gw", "gue",
}


def _is_pure_filler(low: str) -> bool:
    """Filler = no semantic content. '??', '...', single emoji, single word
    that is not a topic name (we conservatively only fire on punctuation/
    very short tokens — single Indonesian words are routed to the LLM)."""
    if not low:
        return True
    return bool(_PURE_PUNCT_RE.match(low))


def _is_math(text: str) -> bool:
    """Pure arithmetic query — at least one digit-operator-digit, and the
    surrounding text is mostly numbers + a question word ("berapa", "=").

    Strips `[...]` bracketed segments first so test suffixes like `[218cb0-3]`
    or random ID tokens don't trigger the regex spuriously. We also drop any
    isolated alphanumeric tokens with internal digits (e.g. `abc-123`) for
    the same reason — those are typically IDs, not math.
    """
    cleaned = re.sub(r"\[[^\]]*\]", " ", text)  # strip [...]-wrapped suffixes
    # Drop ID-shaped tokens (alphanumeric with internal digits, dash-joined) —
    # but require at least one alpha char so plain math like "7-3" survives.
    cleaned = re.sub(r"\b[A-Za-z][\w]*-\d+\w*\b", " ", cleaned)
    cleaned = re.sub(r"\b\w*\d+[A-Za-z]+\w*-\d+\w*\b", " ", cleaned)
    if not _MATH_OP_RE.search(cleaned):
        return False
    stripped = _MATH_OP_RE.sub(" ", cleaned)
    word_count = len(re.findall(r"[A-Za-zÀ-ſ]+", stripped))
    return word_count <= 3


def _is_off_scope_keyword(low: str) -> bool:
    """Match any off-scope keyword. Whole-word for short keywords, raw
    substring for multi-word phrases (less false-positive risk)."""
    padded = " " + low + " "
    for kw in _OFF_SCOPE_KEYWORDS:
        if " " in kw:
            if kw in low:
                return True
        else:
            if f" {kw} " in padded:
                return True
    return False


def _is_identity_question(low: str) -> bool:
    """Bot-identity / app-purpose question. Length-bounded so 'siapa target
    pelanggan Modal' doesn't trigger.

    Every phrase in _IDENTITY_PHRASES is anchored to an explicit bot/app
    reference (kamu/lu/apps/bot/who-are-you), so topic lookups like "modal ini
    apa" or "amartha ini apaan" do NOT match here — no company-name guard
    needed. Anything genuinely ambiguous falls through to the LLM.
    """
    if len(low) > 60:
        return False
    return any(p in low for p in _IDENTITY_PHRASES)


def _is_greeting(low: str) -> bool:
    """PURE greeting only — fires when the whole message is just salutation
    (+ optional conversational fluff). A greeting that PREFIXES a real question
    ("pagi, modal itu apa", "hi gimana cara daftar") must NOT short-circuit; it
    falls through to the LLM so the actual question gets answered.
    """
    if len(low) > 40:
        return False
    if not any(low.startswith(p) for p in _GREETING_PREFIXES):
        return False
    # Tokenize, strip punctuation, and check every token is a greeting word or
    # harmless fluff. Any "real" token (a topic/question word) → not pure → LLM.
    tokens = re.findall(r"[a-zA-ZÀ-ſ]+", low)
    leftover = [t for t in tokens if t not in _GREETING_WORDS and t not in _GREETING_FLUFF]
    return len(leftover) == 0


def classify(text: str) -> Optional[Intent]:
    """Return a high-confidence intent, or None if no rule fires.

    Order matters: filler before greeting (so '??' doesn't match nothing
    and fall through to LLM); identity before off-scope (so 'kamu siapa'
    isn't accidentally caught by a future keyword); greeting last (the
    most permissive prefix match).
    """
    if not text:
        return "AMBIGUOUS"
    low = text.lower().strip()

    if _is_pure_filler(low):
        return "AMBIGUOUS"
    if _is_identity_question(low):
        return "GREETING"
    if _is_math(text):
        return "OFF_SCOPE"
    if _is_off_scope_keyword(low):
        return "OFF_SCOPE"
    if _is_greeting(low):
        return "GREETING"
    return None
