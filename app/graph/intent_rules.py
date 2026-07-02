"""Deterministic intent pre-classifier.

Fast, regex-based rules that handle the highest-confidence intent cases
WITHOUT calling any LLM. Queries that don't match a rule here return None
and fall through to the KNOWLEDGE retrieval path (the pipeline default) —
this module just short-circuits the unambiguous 60-70% of traffic.

NOTE (current state): this module is Ava-only, and Ava's `_pre_processor`
(app/graph/pipeline.py) NO LONGER calls an LLM or embedding classifier for
intent. An unmatched query now defaults to KNOWLEDGE. Askfer has its own
separate LLM pre-processor and does NOT use this module.

History (why this module exists): the original LLM classifier (Gemini Flash
Lite) was bouncing on edge cases when its prompt grew to cover every intent —
each prompt iteration that fixed one case broke another. Pulling deterministic
patterns out eliminated flakiness for things that were never genuinely
ambiguous (math, single emoji, "kamu siapa") and saved LLM cost + latency.
The LLM classification step was later dropped completely, leaving these rules
as the only pre-generation classifier.

Design:
  - Each rule is a function (text, low) -> Optional[Intent]
  - Rules check IN ORDER, first match wins.
  - Rules are conservative — false negatives (no match) fall through to
    the KNOWLEDGE retrieval default; false positives (wrong rule fires)
    are user-visible bugs.
  - No history scanning here — a terse anaphoric follow-up is handled by
    the retrieval-query prepend in _pre_processor, not in this module.
"""
from __future__ import annotations
from pathlib import Path

import re
from typing import Literal, Optional

Intent = Literal["GREETING", "AMBIGUOUS", "OFF_SCOPE", "TOPIC_LIST", "MALICIOUS"]


# ── Load intent patterns from YAML (config-driven, no hardcode) ──────────────
# All keyword/regex/tuple/set constants below are populated from
# intent_patterns.yaml. To add or remove patterns, edit the YAML file
# — DO NOT add them to this .py file. Single source of truth principle.
import yaml as _yaml
_PATTERNS_PATH = Path(__file__).parent / "intent_patterns.yaml"
_PATTERNS = _yaml.safe_load(_PATTERNS_PATH.read_text(encoding="utf-8"))
_OFF_SCOPE_KEYWORDS = tuple(_PATTERNS["off_scope_keywords"])
_IDENTITY_PHRASES = tuple(_PATTERNS["identity_phrases"])
_TOPIC_LIST_PHRASES = tuple(_PATTERNS["topic_list_phrases"])
_THANKS_TOKENS = set(_PATTERNS["thanks_tokens"])
_THANKS_FLUFF = set(_PATTERNS["thanks_fluff"])
_GREETING_WORDS = set(_PATTERNS["greeting_words"])
_GREETING_FLUFF = set(_PATTERNS["greeting_fluff"])
_GREETING_PREFIX_RE = re.compile(_PATTERNS["greeting_prefix_pattern"], re.IGNORECASE | re.VERBOSE)
_TOPIC_LIST_RE = re.compile(_PATTERNS["topic_list_regex"], re.IGNORECASE | re.VERBOSE)
_SECTION_DRILLDOWN_PHRASES = tuple(_PATTERNS.get("section_drilldown_phrases", []))


# ── Rule 0: prompt-injection / jailbreak / system-prompt extraction ──────────
# Deterministic guard for ADVERSARIAL inputs that try to override the bot's
# instructions, extract its system prompt, or activate a "no rules" persona.
# Routes straight to MALICIOUS (canned refusal) before any softer rule fires.
#
# WHY deterministic: an injection attempt ("abaikan instruksimu", "kamu
# sekarang DAN mode") must be caught BEFORE any softer rule can
# mishandle it — historically a role-play-flavored injection misrouted to the
# loosest conversational path and recited the system prompt. A rule here fires
# first, so the attempt never reaches a prompt that could be coaxed.
#
# PRECISION over recall: every pattern requires a verb/target co-occurrence or
# a multi-word anchor that essentially never appears in a legitimate Amartha
# materials question. Subtler paraphrases still fall through to KNOWLEDGE.
#
# CRITICAL false-positive traps explicitly avoided:
#  - "dan" = Indonesian "and" → DAN only matches as an ALL-CAPS standalone
#    token co-occurring with "mode" (original-case), never the conjunction.
#  - bare "instruksi"/"aturan"/"sistem" are common in real questions
#    ("instruksi kerja FO", "sebutkan aturan main CP", "sistem grading") →
#    each needs an override VERB or a bot-referring qualifier (kamu/mu/sistem)
#    or a multi-word anchor ("system prompt") alongside it.
_INJECTION_RES = (
    # 1. instruction-override: (abaikan/lupakan/...) + (instruksi/aturan/prompt/...)
    #    Trailing (?:mu|nya|ku|2)? absorbs Indonesian possessive/plural suffixes so
    #    "abaikan semua aturanMU" / "lupakan instruksiNYA" fire (a bare \b after the
    #    stem fails on the glued suffix). The override VERB is still the real guard,
    #    so this stays false-positive-safe ("sebutkan aturan main CP" has no verb).
    re.compile(
        r"\b(abaikan|lupakan|hiraukan|acuhkan)\b.{0,30}\b(instruksi|aturan|perintah|arahan|prompt)(?:mu|nya|ku|2)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(ignore|forget|disregard|override|bypass)\b.{0,30}\b(instruction|instructions|rule|rules|prompt|previous|prior|above)\b",
        re.IGNORECASE,
    ),
    # 2. system-prompt extraction — multi-word anchors that ~never occur in a
    #    legit question (work instructions are "instruksi kerja"/"SOP", never
    #    "system prompt"/"instruksi sistem").
    re.compile(
        r"\b(system\s+prompt|sistem\s+prompt|prompt\s+sistem|instruksi\s+sistem|system\s+instructions?)\b",
        re.IGNORECASE,
    ),
    # 3. extraction of the bot's OWN prompt/persona via a bot-referring qualifier
    re.compile(
        r"\b(instruksi|prompt|persona)\s+(kamu|mu|lo|lu|asli|rahasia|sistem)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(instruksimu|promptmu|personamu|instruksinya)\b", re.IGNORECASE),
    # 4. jailbreak personas / no-rules mode
    re.compile(
        r"\b(jailbreak|developer\s+mode|dev\s+mode|tanpa\s+aturan|tanpa\s+batasan|tanpa\s+filter|mode\s+tanpa)\b",
        re.IGNORECASE,
    ),
    # 5. DAN persona — ALL-CAPS standalone token co-occurring with "mode" (so
    #    the Indonesian conjunction "dan" never fires). NO IGNORECASE here.
    re.compile(r"\bDAN\b.{0,20}\bmode\b|\bmode\b.{0,20}\bDAN\b"),
    # 6. English jailbreak phrasings that rule 2 misses (it requires an
    #    instruction/rule/prompt object; "bypass your safety filters" and
    #    "unrestricted AI" name no such object). These AI-safety terms
    #    (unrestricted/uncensored/safety filter/guardrail/content policy) are
    #    NEVER part of a legitimate Amartha materials question, so a hard block
    #    here is safe. "do anything now" / "act as DAN" are canonical jailbreaks.
    re.compile(
        r"\b(unrestricted|uncensored|jailbroken|no[\s-]?restrictions?|without\s+restrictions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(bypass|disable|turn\s+off|ignore|remove)\b.{0,30}\b(safety|filter|filters|guardrail|guardrails|moderation|content\s+polic(?:y|ies)|restriction)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdo\s+anything\s+now\b|\bact\s+as\s+(?:a\s+)?(?:dan|an?\s+unrestricted|jailbroken)\b", re.IGNORECASE),
)
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
# questions ("bunga deposito BCA") are deliberately left to fall through to
# KNOWLEDGE retrieval — the dense-floor relevance gate in the pipeline rejects
# a genuinely off-scope competitor query when no in-scope chunk clears the
# floor, so a hard regex block here (which can't tell an in-scope partnership
# question from an off-scope one) would do more harm than good.
    # (loaded from intent_patterns.yaml — see top of file)

# ── Rule 4: bot-identity / app-purpose questions ─────────────────────────────
# Exact-ish matches for "what/who is this" addressed to the bot/app.
# CRITICAL: every phrase MUST anchor to an explicit bot/app reference
# (kamu/lu/lo/apps/bot/aplikasi/who-are-you). Do NOT add bare phrases like
# "ini apa" or "what is this" — those are greedy and wrongly swallow legitimate
# topic lookups ("modal ini apa", "celengan ini apa", "BP ini apa sih"), routing
# them to a self-introduction instead of an answer. Anything genuinely
# ambiguous ("ini apa?" alone) must fall through to KNOWLEDGE retrieval.
    # (loaded from intent_patterns.yaml — see top of file)

# ── Rule 5: greeting prefixes ────────────────────────────────────────────────
# Regex-based to catch common Indonesian typo elongations:
# "halooo", "haiiii", "heiii", "heyy", "helloooo", etc.
# Also covers religious salaams, regional/slang openers, and chat shorthand
# ("p", "pp" = the Indonesian "ping/halo"), since 13k FO greet in many forms.
# Pattern anchored at start of string, case-insensitive applied at call site.
    # (loaded from intent_patterns.yaml — see top of file)

# Keep tuple for backward compat (used in _handle_greeting language detect)
_GREETING_PREFIXES = (
    "halo", "hai", "hei", "hi ", "hi,", "hi.", "hey", "hello",
    "pagi", "siang", "sore", "malam",
    "good morning", "good afternoon", "good evening",
    "selamat", "test ", "test,", "test.",
)

# ── Rule 6: topic-list meta-questions ─────────────────────────────────────────
# Phrases that ask "what topics/courses/materials do you have?" — must NOT
# trigger if user already names a topic (those go to KNOWLEDGE). All phrases
# are anchored to a meta-marker (topik/tema/materi/course/pelatihan/available)
# to avoid false positives on real questions.
#
# NOTE: phrase list is loaded from intent_patterns.yaml (see top of file). Do NOT
# hardcode a tuple here, it will shadow the YAML source-of-truth.

# Regex backstop for TOPIC_LIST phrasings the fixed-substring list above misses:
# the `-nya` suffix ("topiknya apa aja", "materinya apa saja") and the reversed
# word order ("apa aja topiknya"). These are the exact forms that were falling
# through to KNOWLEDGE — where only `final_top_k` chunks are retrieved, spanning
# a SUBSET of courses, so the model listed 3 topics instead of all 5. Routing
# them to TOPIC_LIST instead pulls the complete, ground-truth course list from
# Postgres. Anchored to a topic-meta marker on BOTH sides so content questions
# ("produk apa aja", "prinsip apa aja") do NOT match and stay KNOWLEDGE.
_TOPIC_MARKER = r"(?:topik|tema|materi|konten|course|kursus|pelatihan|modul|pembelajaran)"
# Tolerant trailing suffix: catches the clean word AND typo'd possessives
# ("materinya", "materiny", "topikny", "materinyaa") without requiring a strict
# \b right after the stem — \bmateri\b fails on "materiny" because 'i'→'n' is no
# boundary, so those typos used to fall through to the flaky semantic path.
_MK = rf"{_TOPIC_MARKER}\w{{0,3}}"
_TOPIC_LIST_RE = re.compile(
    # marker(nya) … apa aja/saja      "topiknya apa aja", "materiny aada apa aja"
    rf"\b{_MK}[^.?!\n]{{0,6}}\bapa\s*(?:aja|saja|yg ada|yang ada)\b"
    # apa aja/saja … marker(nya)      "apa aja topiknya"
    rf"|\bapa\s*(?:aja|saja)\b[^.?!\n]{{0,6}}\b{_MK}\b"
    # marker(nya) + apa at END        "topiknya apa", "materinya apa sih"
    rf"|\b{_MK}\s+apa\b(?:\s+(?:sih|dong|donk|ya|yaa|jir|woi|kak|ka|min))?\s*\??\s*$"
    # apa + marker(nya) at END         "apa topiknya", "apa materinya", "apa topik nya"
    # END-anchored so "apa materi client protection" (names a topic) stays KNOWLEDGE.
    rf"|\bapa\s+{_TOPIC_MARKER}(?:\s*nya)?\b(?:\s+(?:sih|dong|donk|ya|yaa|jir|woi|kak|ka|min|nih))?\s*\??\s*$"
    # belajar/pelajari + apa aja/saja "bisa belajar apa aja", "mau belajar apa saja"
    rf"|\b(?:belajar|pelajari|dipelajari|mempelajari)\b[^.?!\n]{{0,6}}\bapa\s*(?:aja|saja)\b"
    # belajar/pelajari + apa at END   "mau belajar apa", "bisa belajar apa sih",
    # "aku bisa belajar apa di sini" (trailing location fluff allowed, but NOT
    # a content noun — "belajar apa itu modal" stays KNOWLEDGE since "itu" isn't
    # in the trailing set).
    rf"|\b(?:belajar|pelajari|dipelajari|mempelajari)\s+apa\b(?:\s+(?:aja|saja|sih|dong|donk|ya|yaa|nih|di\s*sini|disini|disitu|di\s*amartha\w*))*\s*\??\s*$"
    # apa aja/saja … belajar/dipelajari (reversed)  "apa aja yang bisa dipelajari"
    rf"|\bapa\s*(?:aja|saja)\b[^.?!\n]{{0,18}}\b(?:belajar|pelajari|dipelajari)\b",
    re.IGNORECASE,
)

# Fluff/connective tokens that carry no topic content. Used by the bare-"apa aja"
# whole-message check below: if stripping these leaves NOTHING, the message is a
# context-free "what's available?" → topic list. A content noun like "produk" in
# "produk apa aja" survives the strip, so that stays KNOWLEDGE.
_BARE_APAAJA_FLUFF = {
    "emang", "emng", "mang", "ada", "apa", "aja", "saja", "kah", "sih", "dong",
    "donk", "ya", "yaa", "yah", "kak", "ka", "min", "bang", "gan", "jir", "woi",
    "tu", "itu", "yg", "yang", "ini", "tuh", "deh", "nih", "ni", "kira", "kirakira",
}

# Greeting/filler tokens used to decide whether a message is a PURE greeting or
# a greeting that PREFIXES a real question ("pagi, modal itu apa"). Only the
# former should short-circuit to GREETING; the latter must fall through to
# KNOWLEDGE so the actual question gets answered.
    # (loaded from intent_patterns.yaml — see top of file)
# Conversational fluff that may trail a greeting without making it a question.
    # (loaded from intent_patterns.yaml — see top of file)


def _is_injection(text: str) -> bool:
    """Adversarial prompt-injection / jailbreak / system-prompt extraction.

    Matches on the ORIGINAL-case text (not lowercased) so the DAN rule can
    require ALL-CAPS — `dan` (Indonesian "and") must never fire. All other
    patterns are case-insensitive via their own re.IGNORECASE flag.

    Length-bounded generously (attacks can be long); the patterns are precise
    enough that length isn't the guard — co-occurrence is.
    """
    return any(rx.search(text) for rx in _INJECTION_RES)


def _is_pure_filler(low: str) -> bool:
    """Filler = no semantic content. '??', '...', single emoji, single word
    that is not a topic name (we conservatively only fire on punctuation/
    very short tokens — single Indonesian words default to KNOWLEDGE)."""
    if not low:
        return True
    return bool(_PURE_PUNCT_RE.match(low))


# ── Thanks / closing acknowledgment ───────────────────────────────────────────
# Pure "thank you" / sign-off turns that END a thread. Routed to AMBIGUOUS so the
# hot path skips embedding + retrieval and replies conversationally ("sama-sama").
# DELIBERATELY NARROW: only thanks tokens (+ fluff). It does NOT include bare
# "oke"/"iya"/"ya"/"lanjut"/"terus" — those can mean "okay, go on" in the middle
# of a Coaching loop, and this regex tier has no mode/history context to tell the
# difference, so hijacking them to AMBIGUOUS would silently break a coaching turn.
# Those bare affirmations stay KNOWLEDGE (safe, just pay one wasted embed).
    # (loaded from intent_patterns.yaml — see top of file)
    # (loaded from intent_patterns.yaml — see top of file)


def _is_thanks_closer(low: str) -> bool:
    """Pure thanks/sign-off ('makasih', 'thanks ya', 'makasih banyak infonya').

    Conservative: SHORT, must contain a thanks token, and every other token must
    be fluff — so 'makasih tapi aku masih bingung soal X' (a real follow-up) does
    NOT match. See _THANKS_TOKENS note on why 'oke'/'iya'/'lanjut' are excluded.
    """
    if len(low) > 40:
        return False
    tokens = re.findall(r"[a-zA-ZÀ-ſ]+", low)
    if not tokens:
        return False
    if not any(t in _THANKS_TOKENS for t in tokens):
        return False
    return all(t in _THANKS_TOKENS or t in _THANKS_FLUFF for t in tokens)


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
    """Match any off-scope keyword. Uses regex word boundary (\b) so single-
    word tokens like "iphone" match "iphone 15" (trailing digit is not a
    word boundary) and multi-word phrases like "mobile legend" match
    "main mobile legend yuk" (boundary at space)."""
    for kw in _OFF_SCOPE_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", low):
            return True
    return False


def _is_identity_question(low: str) -> bool:
    """Bot-identity / app-purpose question. Length-bounded so 'siapa target
    pelanggan Modal' doesn't trigger.

    Every phrase in _IDENTITY_PHRASES is anchored to an explicit bot/app
    reference (kamu/lu/apps/bot/who-are-you), so topic lookups like "modal ini
    apa" or "amartha ini apaan" do NOT match here — no company-name guard
    needed. Anything genuinely ambiguous falls through to KNOWLEDGE.
    """
    if len(low) > 60:
        return False
    return any(p in low for p in _IDENTITY_PHRASES)


def _is_greeting(low: str) -> bool:
    """PURE greeting only — fires when the whole message is just salutation
    (+ optional conversational fluff). A greeting that PREFIXES a real question
    ("pagi, modal itu apa", "hi gimana cara daftar") must NOT short-circuit; it
    falls through to KNOWLEDGE so the actual question gets answered.

    Uses regex to catch elongated typos: "halooo", "haiiii", "heiii", "heyy".
    """
    if len(low) > 60:
        return False
    m = _GREETING_PREFIX_RE.match(low)
    if not m:
        return False
    # Check the REMAINDER after the greeting prefix — if empty or only fluff,
    # it's a pure greeting. This avoids the token-set lookup failing on
    # elongated forms like "heiii" or "halooo" that aren't in _GREETING_WORDS.
    remainder = low[m.end():].strip()
    if not remainder:
        return True
    # Tokenize remainder, check every token is harmless fluff.
    tokens = re.findall(r"[a-zA-ZÀ-ſ]+", remainder)
    leftover = [t for t in tokens if t not in _GREETING_WORDS and t not in _GREETING_FLUFF]
    return len(leftover) == 0


def _is_topic_list(low: str) -> bool:
    """Meta-question about available topics/courses. Length-bounded so long
    questions about specific topics don't trigger. Every phrase is anchored
    to a topic-meta marker (topik/materi/course/available/list) so a real
    knowledge question like "apa itu Modal" or "jelasin CP" does NOT match.
    """
    if len(low) > 80:
        return False
    if any(p in low for p in _TOPIC_LIST_PHRASES):
        return True
    if _TOPIC_LIST_RE.search(low):
        return True
    # Bare context-free "what's available?" — "apa aja", "emng ada apa aja".
    # Only fires when the WHOLE message is fluff + "apa aja" with NO content
    # noun. A content word ("produk apa aja", "prinsip apa aja") survives the
    # strip → stays KNOWLEDGE. Requires "apa" present so a pure greeting/filler
    # ("ada", "emang") doesn't match.
    if "apa" in low:
        tokens = re.findall(r"[a-zA-ZÀ-ſ]+", low)
        if tokens and all(t in _BARE_APAAJA_FLUFF for t in tokens):
            return True
    return False


def classify(text: str) -> Optional[Intent]:
    """Return a high-confidence intent, or None if no rule fires.

    Order matters: filler before greeting (so '??' doesn't match nothing
    and fall through to KNOWLEDGE); identity before off-scope (so 'kamu siapa'
    isn't accidentally caught by a future keyword); greeting last (the
    most permissive prefix match).
    """
    if not text:
        return "AMBIGUOUS"
    low = text.lower().strip()

    # Rule 0: adversarial injection/jailbreak — checked FIRST, on original-case
    # text (the DAN rule needs ALL-CAPS). Hard-routes to the canned MALICIOUS
    # refusal before any softer rule can mis-handle it.
    if _is_injection(text):
        return "MALICIOUS"

    if _is_pure_filler(low):
        return "AMBIGUOUS"
    if _is_thanks_closer(low):
        return "AMBIGUOUS"
    if _is_identity_question(low):
        return "GREETING"
    if _is_math(text):
        return "OFF_SCOPE"
    if _is_off_scope_keyword(low):
        return "OFF_SCOPE"
    if _is_greeting(low):
        return "GREETING"
    if _is_topic_list(low):
        return "TOPIC_LIST"
    return None
