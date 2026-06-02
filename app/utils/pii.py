"""
PII redaction for telemetry.

Before query/answer text lands in Phoenix spans (or any other persisted
telemetry), we run a regex pass over it to mask structured PII
patterns. Names and free-form addresses are NOT covered (would need a
NER model) — this targets the patterns that are both common in
mood-support / financial / health conversations AND reliably
detectable without false positives.

Detected patterns (all replaced with the same-length mask character):
  - email addresses
  - Indonesian phone numbers (+62..., 08xx, 62xx)
  - 16-digit NIK (Indonesian national ID)
  - 15-16 digit NPWP (Indonesian tax ID, with or without dots/dashes)
  - 13-19 digit credit card numbers (with optional spaces/dashes)

Usage:
    from app.utils.pii import redact_pii
    safe = redact_pii("email saya ferdy@x.com dan NIK 1234567890123456")
    # -> "email saya f████@x.com dan NIK ████████████████"
"""
from __future__ import annotations

import re


# Each pattern's group(0) (the full match) is what gets masked.
# Order matters: longer / more specific patterns first so a phone
# number containing a credit-card-like substring still matches as
# a phone.
_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Email — RFC 5322 simplified; covers the practical 99% of inputs.
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    # 13-19 digit credit card with optional spaces/dashes (Luhn-agnostic
    # — we'd rather false-positive on a long number than leak a card).
    re.compile(r"\b(?:\d[ \-]?){13,19}\b"),
    # Indonesian NIK: exactly 16 digits (with optional spaces every 4-6).
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b"),
    # Indonesian NPWP: 15-16 digits with dots/dashes (e.g. 01.234.567.8-901.000).
    re.compile(r"\b\d{2}\.\d{3}\.\d{3}\.\d{1}[-\.]\d{3}\.\d{3}\b"),
    re.compile(r"\b\d{15,16}\b"),
    # Indonesian phone: +62, 62, or 08 prefix, 8-13 more digits with
    # optional spaces/dashes.
    re.compile(r"\b(?:\+?62|0)8[\d \-]{7,13}\b"),
)


def _mask(match: re.Match[str]) -> str:
    """Replace each char of the match with `*`.

    Using `*` (an ASCII printable) instead of a Unicode block char so
    the redaction survives any terminal/log pipeline (cp1252, ASCII
    email gateways, etc.) and is still visually distinct from any
    alphanumeric. Length is preserved → downstream string-formatting
    assumptions (CSV columns, log layout) still hold.
    """
    return "*" * len(match.group(0))


def redact_pii(text: str | None) -> str | None:
    """Return a PII-redacted copy of `text`. None passes through unchanged."""
    if not text:
        return text
    out = text
    for pat in _PII_PATTERNS:
        out = pat.sub(_mask, out)
    return out


def redact_io(io: dict | None) -> dict | None:
    """Redact PII in the `query`, `resolved_query`, and `answer` fields of
    a span I/O dict. Other fields pass through unchanged so trace
    metadata (intent, scores, conversation_id) is preserved.
    """
    if not io:
        return io
    out = dict(io)
    for key in ("query", "resolved_query", "answer"):
        if key in out and isinstance(out[key], str):
            out[key] = redact_pii(out[key])
    return out
