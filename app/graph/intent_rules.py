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

Intent = Literal["GREETING", "AMBIGUOUS", "OFF_SCOPE", "TOPIC_LIST", "MALICIOUS"]

# ── Rule 0: prompt-injection / jailbreak / system-prompt extraction ──────────
# Deterministic guard for ADVERSARIAL inputs that try to override the bot's
# instructions, extract its system prompt, or activate a "no rules" persona.
# Routes straight to MALICIOUS (canned refusal) BEFORE the LLM classifier.
#
# WHY deterministic: an injection attempt ("abaikan instruksimu", "kamu
# sekarang DAN mode") must be caught BEFORE any softer rule or the LLM path can
# mishandle it — historically a role-play-flavored injection misrouted to the
# loosest conversational path and recited the system prompt. A rule here fires
# first, so the attempt never reaches a prompt that could be coaxed.
#
# PRECISION over recall: every pattern requires a verb/target co-occurrence or
# a multi-word anchor that essentially never appears in a legitimate Amartha
# materials question. Subtler paraphrases still fall through to the LLM.
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
    re.compile(
        r"\b(abaikan|lupakan|hiraukan|acuhkan)\b.{0,30}\b(instruksi|aturan|perintah|arahan|prompt)\b",
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
# Regex-based to catch common Indonesian typo elongations:
# "halooo", "haiiii", "heiii", "heyy", "helloooo", etc.
# Pattern anchored at start of string, case-insensitive applied at call site.
_GREETING_PREFIX_RE = re.compile(
    r"^("
    r"halo+|hai+|hei+|hi+|hey+|hello+"   # salutations with elongation
    r"|pagi|siang|sore|malam"              # time-of-day greetings
    r"|good\s+(morning|afternoon|evening)" # English time greetings
    r"|selamat"                            # formal ID opener
    r"|permisi|excuse\s+me"                # polite openers
    r"|test"                               # test messages
    r")[\s,\.!?]*",                        # optional trailing punct/space
    re.IGNORECASE,
)

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
# Saves ~1500 LLM input tokens per qualifying query by short-circuiting the
# LLM pre-processor call entirely. The handler then injects the topic list.
_TOPIC_LIST_PHRASES = (
    # ID
    "ada topik apa", "topik apa aja", "topik apa saja",
    "ada materi apa", "materi apa aja", "materi apa saja",
    "ada course apa", "course apa aja", "course apa saja",
    "list topik", "list materi", "list course", "list pelatihan",
    "topik yang tersedia", "materi yang tersedia",
    "course yang tersedia", "pelatihan yang tersedia",
    "topik apa yang", "materi apa yang", "course apa yang",
    # EN
    "what topics", "what courses", "what materials", "what training",
    "available topics", "available courses", "available materials",
    "list of topics", "list of courses", "list of materials",
    "topics available", "courses available", "materials available",
    "what do you have", "what can i learn", "what can you teach",
)

# Regex backstop for TOPIC_LIST phrasings the fixed-substring list above misses:
# the `-nya` suffix ("topiknya apa aja", "materinya apa saja") and the reversed
# word order ("apa aja topiknya"). These are the exact forms that were falling
# through to KNOWLEDGE — where only `final_top_k` chunks are retrieved, spanning
# a SUBSET of courses, so the model listed 3 topics instead of all 5. Routing
# them to TOPIC_LIST instead pulls the complete, ground-truth course list from
# Postgres. Anchored to a topic-meta marker on BOTH sides so content questions
# ("produk apa aja", "prinsip apa aja") do NOT match and stay KNOWLEDGE.
_TOPIC_MARKER = r"(?:topik|tema|materi|course|kursus|pelatihan|modul|pembelajaran)"
_TOPIC_LIST_RE = re.compile(
    # marker(nya) … apa aja/saja      "topiknya apa aja", "materi apa saja"
    rf"\b{_TOPIC_MARKER}(?:nya)?\b[^.?!\n]{{0,6}}\bapa\s*(?:aja|saja|yg ada|yang ada)\b"
    # apa aja/saja … marker(nya)      "apa aja topiknya"
    rf"|\bapa\s*(?:aja|saja)\b[^.?!\n]{{0,6}}\b{_TOPIC_MARKER}(?:nya)?\b"
    # marker(nya) + apa at END        "topiknya apa", "materinya apa sih"
    rf"|\b{_TOPIC_MARKER}(?:nya)?\s+apa\b(?:\s+(?:sih|dong|donk|ya|yaa|jir|woi|kak|ka|min))?\s*\??\s*$"
    # belajar/pelajari + apa aja/saja "bisa belajar apa aja", "mau belajar apa saja"
    rf"|\b(?:belajar|pelajari|dipelajari)\b[^.?!\n]{{0,6}}\bapa\s*(?:aja|saja)\b",
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
# former should short-circuit to GREETING; the latter must reach the LLM so the
# actual question gets answered.
_GREETING_WORDS = {
    "halo", "hai", "hei", "hi", "hey", "hello", "pagi", "siang", "sore", "malam",
    "selamat", "datang", "test", "good", "morning", "afternoon", "evening",
    "permisi", "excuse", "me",
    "assalamualaikum", "assalam", "waalaikumsalam", "waalaykum",
}
# Conversational fluff that may trail a greeting without making it a question.
_GREETING_FLUFF = {
    "ya", "yaa", "dong", "donk", "kak", "ka", "min", "bang", "gan", "ges",
    "gaes", "nih", "ni", "aja", "deh", "sih", "kok", "aku", "saya", "gw", "gue",
    "wr", "wb", "semua", "semuanya", "teman", "temen", "guys", "all",
}


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
    """Match any off-scope keyword. Whole-word match for both single and
    multi-word keywords to avoid catching substrings of legitimate terms."""
    padded = " " + low + " "
    for kw in _OFF_SCOPE_KEYWORDS:
        if " " in kw:
            # Multi-word: pad with spaces so "berita terkini" doesn't match
            # "soal berita terkininya" (the trailing "nya" would be caught
            # by the unpadded `kw in low` check).
            if f" {kw} " in padded:
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
    and fall through to LLM); identity before off-scope (so 'kamu siapa'
    isn't accidentally caught by a future keyword); greeting last (the
    most permissive prefix match).
    """
    if not text:
        return "AMBIGUOUS"
    low = text.lower().strip()

    # Rule 0: adversarial injection/jailbreak — checked FIRST, on original-case
    # text (the DAN rule needs ALL-CAPS). Hard-routes to the canned MALICIOUS
    # refusal before any softer rule or the LLM classifier can mis-handle it.
    if _is_injection(text):
        return "MALICIOUS"

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
    if _is_topic_list(low):
        return "TOPIC_LIST"
    return None
