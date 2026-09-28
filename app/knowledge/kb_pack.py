import hashlib
import html
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.knowledge.moodle_markdown import MoodleMarkdownFile


@dataclass(frozen=True)
class KBPack:
    text: str
    kb_hash: str


_DOC_TAG_RE = re.compile(
    r'<doc\s+[^>]*section="([^"]*)"[^>]*file="([^"]*)"',
    re.IGNORECASE,
)
_ROLES_COMMENT_RE = re.compile(r"<!--\s*roles:\s*([^\-]+?)\s*-->", re.IGNORECASE)
_ROLES_FM_RE = re.compile(r"^roles:\s*\[?([^\]\n]+)\]?", re.MULTILINE)
_H123_LINE_RE = re.compile(r"^(?:AI:\s*|User:\s*)?#{1,3}\s+(.+)$", re.MULTILINE)
_BOLD_TITLE_RE = re.compile(r"\*\*([^*]{3,50})\*\*")
_H2_LINE_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_EMPH_RE = re.compile(r"[*_`~]+")


def _extract_roles(content: str) -> list[str]:
    # Match: <!-- roles: HO, HMB, AM, RM -->
    match = _ROLES_COMMENT_RE.search(content)
    if match:
        return [r.strip().upper() for r in match.group(1).split(",")]
    
    # Check frontmatter:
    # roles: [BP, BM]
    # roles: BP, BM
    match_fm = _ROLES_FM_RE.search(content)
    if match_fm:
        return [r.strip().upper() for r in match_fm.group(1).split(",")]
        
    return ["ALL"]


class KBService:
    """Encapsulated Knowledge Base service for pack assembly and section/topic parsing."""

    @staticmethod
    def assemble_pack(docs: Sequence[MoodleMarkdownFile]) -> KBPack:
        return assemble_kb_pack(docs)

    @staticmethod
    def extract_sections(kb_text: str) -> dict[str, list[str]]:
        return extract_kb_sections(kb_text)

    @staticmethod
    def extract_topics(kb_text: str) -> list[str]:
        return extract_kb_topics(kb_text)

    @staticmethod
    def extract_h2_headings(kb_text: str) -> list[str]:
        return extract_kb_h2_headings(kb_text)



def assemble_kb_pack(docs: Sequence[MoodleMarkdownFile]) -> KBPack:
    ordered = sorted(docs, key=lambda d: (d.course_id, d.section_id, d.filename))
    canonical = [
        {
            "course_id": d.course_id,
            "section_id": d.section_id,
            "section_name": d.section_name,
            "filename": d.filename,
            "content": _normalize_text(d.content),
            "roles": _extract_roles(d.content),
        }
        for d in ordered
    ]
    kb_hash = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    lines = [f'<knowledge_base version="sha256:{kb_hash}">', "<kb_index>"]
    for i, doc in enumerate(canonical, 1):
        lines.extend(
            [
                f"- [DOC-{i:03d}] Course: {doc['course_id']}",
                f"  Section: {doc['section_name']}",
                f"  File: {doc['filename']}",
            ]
        )
    lines.extend(["</kb_index>", ""])

    for i, doc in enumerate(canonical, 1):
        roles_str = ",".join(doc["roles"])
        lines.extend(
            [
                (
                    f'<doc id="DOC-{i:03d}" course="{doc["course_id"]}" '
                    f'section="{html.escape(str(doc["section_name"]), quote=True)}" '
                    f'file="{html.escape(str(doc["filename"]), quote=True)}" '
                    f'roles="{roles_str}">'
                ),
                str(doc["content"]),
                "</doc>",
            ]
        )
    lines.append("</knowledge_base>")
    return KBPack(text="\n".join(lines), kb_hash=kb_hash)


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_kb_sections(kb_text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    for section_raw, file_raw in _DOC_TAG_RE.findall(kb_text or ""):
        section = html.unescape(section_raw).strip()
        item = Path(html.unescape(file_raw).strip()).stem.strip()
        if not section or not item:
            continue
        sections.setdefault(section, [])
        if item not in sections[section]:
            sections[section].append(item)
    return {section: sections[section] for section in sorted(sections)}


def extract_kb_topics(kb_text: str) -> list[str]:
    return list(extract_kb_sections(kb_text).keys())


def extract_kb_filenames(kb_text: str) -> list[str]:
    """Extract all individual document filenames (without extension) from active KB."""
    filenames: list[str] = []
    for _section, file_raw in _DOC_TAG_RE.findall(kb_text or ""):
        stem = Path(html.unescape(file_raw).strip()).stem.strip()
        if stem and stem not in filenames:
            filenames.append(stem)
    return sorted(filenames)


def extract_h2_headings(markdown_text: str) -> list[str]:
    """Extract headings (#, ##, ###) or bold titles from markdown content."""
    if not markdown_text:
        return []
    headings = _H123_LINE_RE.findall(markdown_text)
    bolds = _BOLD_TITLE_RE.findall(markdown_text)
    candidates = headings + bolds
    cleaned = []
    ignore_words = {"user", "ai", "note", "catatan", "perhatian", "penting", "sumber", "ringkasan"}
    for c in candidates:
        clean = _HTML_TAG_RE.sub("", c).strip().strip(":#*-_")
        if clean and len(clean) >= 3 and clean.lower() not in ignore_words and clean not in cleaned:
            cleaned.append(clean)
    return cleaned[:10]


def extract_kb_h2_headings(kb_text: str) -> list[str]:
    """Extract all unique level-2 (## ) markdown headings from active KB text."""
    if not kb_text:
        return []
    raw_h2 = _H2_LINE_RE.findall(kb_text)
    ignore_words = {"note", "catatan", "perhatian", "penting", "sumber", "ringkasan"}
    headings: list[str] = []
    for h in raw_h2:
        clean = _HTML_TAG_RE.sub("", h)
        clean = _MD_EMPH_RE.sub("", clean).strip().strip(":#*-_")
        if clean and len(clean) >= 3 and clean.lower() not in ignore_words and clean not in headings:
            headings.append(clean)
    return headings



