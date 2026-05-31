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
# Amartha terms ("modal" must NOT match "modal usaha").
_OFF_SCOPE_KEYWORDS = (
    # Weather
    "cuaca", "weather", "hujan deras", "panas banget hari ini",
    # News / current affairs
    "berita terkini", "berita hari ini", "headline", "breaking news",
    # Food / recipes
    "resep", "recipe", "masak", "menu makan", "restoran",
    # Sports
    "skor bola", "pertandingan", "liga inggris", "premier league", "world cup",
    # Politics / world facts
    "siapa presiden", "ibu kota negara", "presiden indonesia",
    "menteri keuangan", "menteri",
    # Other companies / banks (non-Amartha)
    "bca", "mandiri", "bri ", "bni ", "gopay", "ovo", "dana app",
    "shopeepay", "tokopedia", "shopee", "grab", "gojek",
    # Celebrity / entertainment
    "artis", "selebriti", "selebgram",
)

# ── Rule 4: bot-identity / app-purpose questions ─────────────────────────────
# Exact-ish matches for "what/who is this" addressed to the bot/app.
# Casual variants covered. Length-bounded to avoid catching long sentences
# that just happen to contain "siapa".
_IDENTITY_PHRASES = (
    # ID
    "kamu siapa", "lu siapa", "lo siapa", "elu siapa",
    "ini apa", "ini apps apa", "ini apps buat apa", "ini bot apa",
    "kamu bot apa", "kamu apa sih", "lu apa sih",
    "siapa kamu", "perkenalkan diri",
    # EN
    "who are you", "what are you", "what is this", "what is this app",
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


# ── Company-name guard ───────────────────────────────────────────────────────
# Standalone "amartha" = the COMPANY (a KB topic), not the assistant. Word
# boundaries ensure this matches "amartha" but NOT "amarthapedia" (the LMS) or
# "a-pedi" (the bot) — those remain assistant-identity references.
_COMPANY_REF_RE = re.compile(r"\bamartha\b", re.IGNORECASE)


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

    Guard: if the user names the COMPANY "amartha" (standalone word), it's a
    factual lookup about the company — NOT a question about the assistant.
    Without this, greedy phrases like "ini apa" wrongly catch "amartha ini
    apaan" (the company) and route it to GREETING instead of KNOWLEDGE.
    `\\bamartha\\b` matches the company word but NOT "amarthapedia"/"a-pedi"
    (the assistant), since there's no word boundary inside "amarthapedia".
    """
    if len(low) > 60:
        return False
    if _COMPANY_REF_RE.search(low):
        return False
    return any(p in low for p in _IDENTITY_PHRASES)


def _is_greeting(low: str) -> bool:
    """Short message that begins with a greeting prefix."""
    if len(low) > 40:
        return False
    return any(low.startswith(p) for p in _GREETING_PREFIXES)


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
