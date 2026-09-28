"""Section Drilldown helpers (deterministic regex + fuzzy token matching)."""
from pathlib import Path
import re

_SECTION_NAME_STOPWORDS = frozenset({
    "ada", "apa", "aja", "saja", "di", "dari", "ke", "yang", "itu",
    "ini", "tadi", "tuh", "nih", "kan", "ya", "ga", "gak", "nggak",
    "kok", "sih", "dong", "kak", "bang", "mas", "mbak", "bu", "pak",
    "tolong", "mau", "ingin", "bisa", "dapat", "lihat", "tampil",
    "list", "daftar", "show", "tampilkan", "lihatin",
    "materi", "materinya", "dokumen", "dokumennya", "judul", "judulnya",
    "file", "filenya", "topik", "topiknya", "topic", "section",
    "course", "kursus", "pelajaran", "ajar", "nya", "aja",
})

try:
    import yaml as _yaml_drilldown
    _DRILLDOWN_PATTERNS_PATH = Path(__file__).parent / "intent_patterns.yaml"
    _DRILLDOWN_PATTERNS = _yaml_drilldown.safe_load(
        _DRILLDOWN_PATTERNS_PATH.read_text(encoding="utf-8")
    ) or {}
    _SECTION_DRILLDOWN_PHRASES = tuple(_DRILLDOWN_PATTERNS.get("section_drilldown_phrases", []))
except Exception:
    _SECTION_DRILLDOWN_PHRASES = (
        "ada apa aja", "ada apa", "apa aja", "apa saja", "apa isinya",
        "isinya apa", "di dalamnya apa", "dalamnya apa", "materinya apa",
        "materi apa", "dokumennya apa", "judulnya apa", "list materi",
        "list dokumen", "list judul", "daftar materi",
        "tolong lihat", "lihat materi", "tampilkan materi",
        "tampilkan dokumen",
    )

_ORDINAL_TO_INT = {
    "1": 1, "satu": 1, "pertama": 1, "kesatu": 1, "a": 1,
    "2": 2, "dua": 2, "kedua": 2, "kedu": 2, "b": 2,
    "3": 3, "tiga": 3, "ketiga": 3, "c": 3,
    "4": 4, "empat": 4, "keempat": 4, "d": 4,
    "5": 5, "lima": 5, "kelima": 5, "e": 5,
    "6": 6, "enam": 6, "keenam": 6, "f": 6,
    "7": 7, "tujuh": 7, "ketujuh": 7, "g": 7,
    "8": 8, "delapan": 8, "kedelapan": 8, "h": 8,
}


def _normalize_section_tokens(name: str) -> list[str]:
    """Lowercase + strip punctuation + remove stopwords. Returns significant tokens."""
    s = re.sub(r"[^\w\s]", " ", (name or "").lower())
    return [t for t in s.split() if t and t not in _SECTION_NAME_STOPWORDS and len(t) > 1]


def _levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein edit distance. O(len(a)*len(b)). For short tokens only."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i, bc in enumerate(b, 1):
        cur = [i]
        for j, ac in enumerate(a, 1):
            cur.append(min(
                cur[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ac != bc),
            ))
        prev = cur
    return prev[-1]


def _fuzzy_token_match(qt: str, st: str) -> bool:
    """Token match with edit-distance fallback for cross-language stem variants."""
    if not qt or not st:
        return False
    if qt == st:
        return True
    ratio = min(len(qt), len(st)) / max(len(qt), len(st))
    if ratio < 0.55:
        return False
    d = _levenshtein(qt, st)
    max_edits = max(2, int(max(len(qt), len(st)) * 0.30))
    return d <= max_edits


def _score_query_against_section(query: str, section_name: str) -> float:
    """Score how well `query` matches `section_name`. 0.0 = no match, 1.0 = perfect."""
    q_toks = _normalize_section_tokens(query)
    s_toks = _normalize_section_tokens(section_name)
    if not q_toks or not s_toks:
        return 0.0
    overlap = 0
    for qt in q_toks:
        for st in s_toks:
            if qt in st or st in qt:
                overlap += 1
                break
            if len(qt) >= 4 and len(st) >= 4 and qt[:4] == st[:4]:
                overlap += 1
                break
            if _fuzzy_token_match(qt, st):
                overlap += 1
                break
    token_score = overlap / max(1, len(s_toks))
    q_full = " ".join(q_toks)
    s_full = " ".join(s_toks)
    if q_full and s_full and (q_full in s_full or s_full in q_full):
        return 1.0
    return min(1.0, token_score)


def _detect_section_from_query(query: str, section_map: dict[str, list[str]]) -> str | None:
    """Match query -> canonical section name via token containment."""
    if not section_map:
        return None
    best_section, best_score = None, 0.0
    for section in section_map.keys():
        score = _score_query_against_section(query, section)
        if score > best_score:
            best_score, best_section = score, section
    return best_section if best_score >= 0.30 else None


def _flatten_message_content(content) -> str:
    """LangChain message content can be str OR list[{type:text}]. Flatten to str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict):
                txt = blk.get("text") or blk.get("content") or ""
                if txt:
                    parts.append(str(txt))
            elif isinstance(blk, str):
                parts.append(blk)
        return " ".join(parts)
    return str(content) if content else ""


def _has_topic_list_marker(content: str) -> bool:
    """Detect if a previous AI message was a TOPIC_LIST response."""
    low = (content or "").lower()
    markers = (
        "berikut topik", "topik-topik", "daftar topik", "berikut daftar",
        "ini dia topik", "topik yang tersedia", "berikut beberapa topik",
        "kamu bisa belajar", "kamu bisa pelajari", "materi yang tersedia",
        "available topics", "topics available",
    )
    return any(m in low for m in markers)


def _extract_sections_from_topic_list(content: str) -> list[str]:
    """Parse a TOPIC_LIST AI response to recover the section list."""
    if not content:
        return []
    text = content

    numbered = re.findall(
        r"(?:^|\n)\s*(?:\d+|[A-Ha-h])[\.\)]\s+([^\n]{2,80})", text
    )
    if numbered:
        cleaned = []
        for s in numbered:
            s = s.strip().rstrip(",;.")
            s = re.sub(r"^[\*_\-`]+|[\*_\-`]+$", "", s).strip()
            if 2 <= len(s) <= 80:
                cleaned.append(s)
        if cleaned:
            return cleaned

    bullets = re.findall(r"(?:^|\n)\s*[-*•·]\s+([^\n]{2,80})", text)
    if bullets:
        cleaned = []
        for s in bullets:
            s = s.strip().rstrip(",;.")
            s = re.sub(r"^[\*_\-`]+|[\*_\-`]+$", "", s).strip()
            if 2 <= len(s) <= 80:
                cleaned.append(s)
        if cleaned:
            return cleaned

    bolds = re.findall(r"\*\*([^*\n]{2,60})\*\*", text)
    if bolds:
        cleaned = [s.strip().rstrip(",;.") for s in bolds if 2 <= len(s.strip()) <= 60]
        if cleaned:
            return cleaned

    return []


def _resolve_section_ordinal(query: str, sections: list[str]) -> str | None:
    """Resolve 'yang kedua', 'topik B', 'nomor 3' against a section list."""
    q = (query or "").lower().strip()
    if not q or not sections:
        return None
    m = re.search(
        r"(?:yang|topi[ck]|no(?:mor)?|pilihan?)\s*"
        r"(?:ke-?|nomor\s*)?\s*"
        r"(satu|dua|tiga|empat|lima|enam|tujuh|delapan|"
        r"pertama|kedua|ketiga|keempat|kelima|keenam|ketujuh|kedelapan|"
        r"[1-8]|[a-h])\b",
        q,
    )
    if m:
        word = m.group(1).lower()
        idx = _ORDINAL_TO_INT.get(word)
        if idx and 1 <= idx <= len(sections):
            return sections[idx - 1]
    if re.fullmatch(
        r"(?:yang\s+(?:itu|tadi|barusan|sebelumnya|maksud|disebut|dibahas))+|"
        r"(?:yang)|(?:itu)|(?:tadi)|(?:yang\s+aja)|(?:pilih\s+itu)",
        q.strip(),
    ):
        if len(sections) == 1:
            return sections[0]
        return None
    return None


def _extract_topic_list_from_history(messages: list) -> list[str]:
    """Walk messages backwards, find last AI TOPIC_LIST response, return section list."""
    if not messages:
        return []
    for m in reversed(messages[:-1]):
        role = getattr(m, "type", None) or getattr(m, "role", "")
        if role and role not in ("ai", "assistant"):
            continue
        content = _flatten_message_content(getattr(m, "content", ""))
        if not _has_topic_list_marker(content):
            continue
        sections = _extract_sections_from_topic_list(content)
        if sections:
            return sections
    return []


def _resolve_drilldown_section(
    query: str,
    messages: list,
    section_map: dict[str, list[str]],
) -> tuple[str | None, str | None]:
    """Resolve drilldown query -> canonical section name."""
    if not query or not section_map:
        return None, None

    sections_in_history = _extract_topic_list_from_history(messages or [])
    if not sections_in_history:
        return None, None

    ordinal = _resolve_section_ordinal(query, sections_in_history)
    if ordinal:
        for sec in section_map.keys():
            if _score_query_against_section(ordinal, sec) >= 0.50:
                return sec, "history_ordinal"

    best, best_score = None, 0.0
    for sec in sections_in_history:
        score = _score_query_against_section(query, sec)
        if score > best_score:
            best_score, best = score, sec
    if best and best_score >= 0.50:
        for sec in section_map.keys():
            if _score_query_against_section(best, sec) >= 0.50:
                return sec, "history"

    direct = _detect_section_from_query(query, section_map)
    if direct:
        return direct, "query"

    return None, None


def _is_section_drilldown_shape(query: str) -> bool:
    """Quick shape check: does the query LOOK like 'what's inside topic X'?"""
    if not query or len(query) > 150:
        return False
    low = query.lower().strip()
    return any(p in low for p in _SECTION_DRILLDOWN_PHRASES)
