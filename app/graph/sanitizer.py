"""Output sanitization, prompt-leak guard, and Amartha glossary expansion."""
import re

# Strips leaked instruction blocks from the LLM response. Some models
# (Gemini Flash Lite especially) occasionally echo the literal contents of
# <retrieved_context> / <user_history> / etc. as part of their output —
# leading to giant <h1>-rendered context dumps in the UI. We catch that
# server-side as a defensive net even after prompt-level guards.
_LEAK_BLOCK_RE = re.compile(
    r"<(retrieved_context|user_history|previous_context|user_preferences|user_context|response_shape|conversation_signals|capabilities|mode|output_contract|role|rules|how_to_talk|length|grounding|disambiguate|no_context|when_to_ask_vs_answer|how_to_ask|during_the_loop|wrap_up|scope|available_topics)>"
    r".*?"
    r"</\1>\s*",
    re.DOTALL | re.IGNORECASE,
)
_LEAK_OPEN_TAG_RE = re.compile(
    r"</?(retrieved_context|user_history|previous_context|user_preferences|user_context|response_shape|conversation_signals|capabilities|mode|output_contract|role|rules|how_to_talk|length|grounding|disambiguate|no_context|when_to_ask_vs_answer|how_to_ask|during_the_loop|wrap_up|scope|available_topics)>",
    re.IGNORECASE,
)
_OFFSCOPE_RE = re.compile(r"\[OFFSCOPE\]", re.IGNORECASE)
_OFFSCOPE_PARTIAL_RE = re.compile(
    r"\[(?:O(?:F(?:F(?:S(?:C(?:O(?:P(?:E\]?)?)?)?)?)?)?)?)?$",
    re.IGNORECASE,
)
_COURSE_NUM_RE = re.compile(r"\bCourse\s+\d+(?:\s*:\s*|\s+)?", re.IGNORECASE)
# Citation header from context formatter — "[N] Course: <name> (ID:<id>)".
# Distinctive pattern; never appears in legitimate prose.
_LEAK_CITATION_HEAD_RE = re.compile(
    r"^\s*(?:[>\-*]\s*)?(?:\d+[.)]\s*)?(?:\[\d+\]\s*)?Course:\s*[^\n]*",
    re.MULTILINE | re.IGNORECASE,
)
_META_CONTEXT_LINE_RE = re.compile(
    r"^\s*>?\s*\*\*\[Meta-(?:Context|Konteks)\]\*\*[^\n]*",
    re.MULTILINE | re.IGNORECASE,
)
# ATX markdown headings — "# Foo", "## Bar".
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+", re.MULTILINE)
# Inline source citations like "[[1]]" or "[[1]][[2]]".
_INLINE_CITE_RE = re.compile(r"\[\[\d+\]\]")
# Layer-4 leak: lines that look like LITERAL prompt directives.
_DIRECTIVE_LINE_RE = re.compile(
    r"^[ \t]*(?:"
    r"Default\s*:\s*SHORT|"
    r"Go LONGER and more structured|"
    r"EXCEPTION\s*[—–-]|"
    r"NEVER\s+(?:echo|pull|use|close|start|emit|start|open)|"
    r"ALWAYS\s+(?:open|close|preserve|use|emit|start)|"
    r"Open with the answer|"
    r"End with substance|"
    r"No hedging|"
    r"Use complete sentences|"
    r"Use bullets for lists|"
    r"Mirror the user's language|"
    r"If <context> is absent|"
    r"When the context (?:IS|is) relevant|"
    r"When the user asks about (?:a SET|the set)|"
    r"CRITICAL\s*[—–-]|"
    r"Talk like a senior|"
    r"Answer factual (?:lookups|questions)|"
    r"Format examples \(Indonesian\)|"
    r"STYLE\s*[—–-]|"
    r"MENTOR MINDSET|"
    r"In COACHING mode|"
    r"FRUSTRATION OVERRIDE|"
    r"COACHING CONDUCT|"
    r"First check RELEVANCE|"
    r"When the context IS relevant|"
    r"(?:The )?[Uu]ser asked what topics|"
    r"List ONLY the topics|"
    r"Runs before answering|"
    r"Check if the turn is UNDERSPECIFIED|"
    r"Ask ONE short clarifying question|"
    r"Irrelevant with the user question|"
    r"\(\d\)\s+A (?:broad|bare|reference|BARE)"
    r")"
    r"[^\n]*",
    re.MULTILINE | re.IGNORECASE,
)

# Meta-conversation recall questions ("udah bahas apa aja", "yang kita bahas", etc.)
_META_CONVO_RE = re.compile(
    r"(?:udah|sudah|udh|tadi|barusan|kita|kami)\b[^.?!\n]{0,30}"
    r"(?:bahas|dibahas|ngomong|omongin|diskusi|obrol)"
    r"|(?:yang|apa)\b[^.?!\n]{0,20}(?:tadi|barusan|kita|kami|sebelumnya)\s+(?:di)?(?:bahas|omongin|diskusi)"
    r"|itu aja[^.?!\n]{0,25}(?:bahas|omongin)"
    r"|what (?:did|have|were) we (?:discuss|talk|cover|go over|chat)",
    re.IGNORECASE,
)

_AMARTHA_GLOSSARY = {
    "BM": "Business Manager",
    "BP": "Business Partner",
    "PAR": "Portfolio at Risk",
    "OS": "Outstanding",
    "BTC": "Back to Current",
    "DPD": "Days Past Due",
    "NPL": "Non-Performing Loan",
    "RR": "Repayment Rate",
    "PJ": "Penanggung Jawab",
}

_GLOSSARY_PATTERN = r"\b(" + "|".join(_AMARTHA_GLOSSARY.keys()) + r")\b"
_GLOSSARY_RE = re.compile(_GLOSSARY_PATTERN, flags=re.IGNORECASE)


def _apply_glossary(text: str) -> str:
    """Replaces Amartha acronyms with their full terms using exact word boundaries."""
    if not text:
        return text
    return _GLOSSARY_RE.sub(lambda m: _AMARTHA_GLOSSARY[m.group(0).upper()], text)


_BOLD_EM_DASH_RE = re.compile(r"\*\*\s*—\s*")
_EM_DASH_NORM_RE = re.compile(r"\s*—\s*")
_CLOSING_REF_RE = re.compile(
    r"Untuk\s+detail\s+[^.!?]+silakan\s+cek\s+langsung\s+di\s+modul\s+Business\s+Process(?:[^.!?]*Amarthapedia)?\.?",
    re.IGNORECASE,
)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


def _normalize_dashes(text: str) -> str:
    if "—" in text:
        text = _BOLD_EM_DASH_RE.sub("**: ", text)
        text = _EM_DASH_NORM_RE.sub(", ", text)
    if "–" in text:
        text = text.replace("–", "-")
    return text


def _sanitize_answer(text: str) -> str:
    """Strip any leaked instruction-block content / tags from an LLM reply."""
    if not text:
        return text
    cleaned = _CLOSING_REF_RE.sub(
        "Kamu bisa pelajari lebih lanjut di Amarthapedia atau bertanya langsung denganku.",
        text,
    )
    cleaned = _LEAK_BLOCK_RE.sub("", cleaned)
    cleaned = _LEAK_OPEN_TAG_RE.sub("", cleaned)
    cleaned = _INLINE_CITE_RE.sub("", cleaned)
    cleaned = _META_CONTEXT_LINE_RE.sub("", cleaned)
    cleaned = _DIRECTIVE_LINE_RE.sub("", cleaned)
    cleaned = _OFFSCOPE_RE.sub("", cleaned)
    cleaned = _COURSE_NUM_RE.sub("", cleaned)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)

    matches = list(_LEAK_CITATION_HEAD_RE.finditer(cleaned))
    if matches:
        last_end = matches[-1].end()
        tail = cleaned[last_end:].strip()
        paragraphs = [p.strip() for p in _PARA_SPLIT_RE.split(tail) if p.strip()]
        if len(paragraphs) >= 2:
            cleaned = "\n\n".join(paragraphs[1:])
        elif paragraphs:
            only = paragraphs[0]
            if len(only) > 80 and not only.startswith(("-", "*", "•")):
                cleaned = only
            else:
                cleaned = "Maaf, ada kendala merangkum jawaban. Coba tanya ulang ya."
        else:
            cleaned = "Maaf, ada kendala merangkum jawaban. Coba tanya ulang ya."

    cleaned = _normalize_dashes(cleaned.lstrip())
    if text.strip() and not cleaned.strip():
        if _OFFSCOPE_RE.search(text):
            return cleaned
        return "Maaf, ada kendala merangkum jawaban. Coba tanya ulang ya."
    return cleaned


class StreamLeakGuard:
    """Stream-time leak detector. Buffers the generated reply only when a leak
    signature is detected, otherwise passes clean tokens through immediately.
    """

    _LEAK_PATTERNS = (
        _LEAK_CITATION_HEAD_RE,
        _LEAK_OPEN_TAG_RE,
        _INLINE_CITE_RE,
        _DIRECTIVE_LINE_RE,
        _OFFSCOPE_PARTIAL_RE,
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._mode = "passthrough"

    def feed(self, token: str) -> str:
        """Push a streamed token. Returns the safe text to emit (may be "")."""
        self._buffer += token
        buf = self._buffer
        if self._mode == "buffered":
            if any(p.search(buf) for p in self._LEAK_PATTERNS):
                return ""
        else:
            has_leak = (
                ("<" in buf and _LEAK_OPEN_TAG_RE.search(buf))
                or ("[" in buf and (_INLINE_CITE_RE.search(buf) or _OFFSCOPE_PARTIAL_RE.search(buf)))
                or (":" in buf and _LEAK_CITATION_HEAD_RE.search(buf))
                or _DIRECTIVE_LINE_RE.search(buf)
            )
            if has_leak:
                self._mode = "buffered"
                return ""

        self._mode = "passthrough"
        self._buffer = ""
        return buf

    def flush(self) -> str:
        """Called at end-of-stream. Returns sanitized trailing text."""
        if self._mode == "buffered":
            cleaned = _sanitize_answer(self._buffer)
            self._buffer = ""
            return cleaned
        out = self._buffer
        self._buffer = ""
        return out

    @property
    def leak_detected(self) -> bool:
        return self._mode == "buffered"
