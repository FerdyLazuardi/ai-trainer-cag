"""Shared user-language detection used by intent handlers.

Single source of truth so the topic-list handler, low-relevance handler,
and any future canned-response handler use the same heuristic. Scans
HumanMessages across the full conversation history — short queries like
"producct amartha" don't carry enough markers on their own to flip the
flag, but if ANY prior user turn was Indonesian we should stay Indonesian.
"""
from __future__ import annotations

from typing import Iterable


# Pronouns, question words, particles, slang, common verbs/connectors.
# Covers both formal Indonesian and casual variants ("gw", "lu", "dong", "deh").
_ID_MARKERS: tuple[str, ...] = (
    # Pronouns
    "kamu", "saya", "aku", "kita", "anda", "gw", "gue", "lu", "lo",
    # Question words
    "apa", "apakah", "siapa", "gimana", "bagaimana", "kenapa", "mengapa",
    "dimana", "di mana", "kapan", "berapa",
    # Verbs / connectors / particles
    "jelasin", "jelaskan", "ceritain", "ceritakan", "yang", "ini", "itu",
    "adalah", "untuk", "dengan", "sebagai", "punya", "buat", "bikin",
    "kerja", "udah", "sudah", "belum", "lagi", "mau", "pengen", "ingin",
    "tolong", "coba", "boleh", "harus", "bisa", "dapetin", "cara",
    # Particles / slang
    "dong", "sih", "nih", "deh", "banget", "aja", "tuh", "yaa", "kak",
    "bro", "bedanya", "ngapain",
    # Topic-list flavour markers
    "topik", "materi", "kursus", "ada apa", "fungsi", "buat apa",
    "semuanya", "semua",
)

# Common ID prefixes that wouldn't match space-padded membership but are
# unambiguously Indonesian when they lead the message.
_ID_PREFIXES: tuple[str, ...] = ("apa itu", "gimana", "kenapa", "kok ")


def is_indonesian(text: str) -> bool:
    """True if a single text snippet looks Indonesian."""
    if not text:
        return False
    low = text.lower()
    padded = " " + low + " "
    if any(f" {w} " in padded for w in _ID_MARKERS):
        return True
    return low.startswith(_ID_PREFIXES)


def history_is_indonesian(messages: Iterable) -> bool:
    """True if ANY HumanMessage in the conversation history looks Indonesian.

    Defaults to False (English) so the caller decides the fallback. Only
    inspects HumanMessage objects — assistant turns are skipped because the
    bot's language is downstream of the user's, not a separate signal.
    """
    from langchain_core.messages import HumanMessage  # local import to avoid cycle

    for m in messages or ():
        if not isinstance(m, HumanMessage):
            continue
        content = getattr(m, "content", None)
        text = content if isinstance(content, str) else ""
        if is_indonesian(text):
            return True
    return False
