"""Course catalog search and query detection utility.

Provides:
- detect_course_query: extracts subject keywords if user asks for a course/class link.
- search_courses: searches PostgreSQL course_catalog via pg_trgm similarity or ILIKE fallback.
"""

from __future__ import annotations

import re
from typing import Any
from loguru import logger
from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import CourseCatalog

# Patterns detecting explicit request for course/class link:
# "minta link course ...", "link kelas ... dong", "ada link pelatihan ...", "url modul ..."
_COURSE_LINK_PREFIX_PATTERNS = [
    re.compile(
        r"(?:minta|bagi|spill|cari|tolong|ada|kasih|kirim|bisa\s+minta|bisa\s+kasih|dimana)?\s*(?:link|tautan|url)\s+(?:buat\s+|untuk\s+)?(?:course|kelas|modul|pelatihan|materi|training|belajar)?\s*(?:tentang|buat|untuk|soal)?\s*(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:mau\s+ikut|mau\s+daftar|ikut)?\s*(?:course|kelas|modul|pelatihan|training|materi)\s+(?:tentang|buat|untuk|soal)?\s*(.+?)\s*(?:ada\s+link|linknya\s+apa|linknya\s+dimana|linknya\s+mana|minta\s+link|link)?\s*[?.]*$",
        re.IGNORECASE,
    ),
]


# Follow-up link requests referring to previous conversation turn ("link nya", "minta linknya", "mana linknya")
_FOLLOW_UP_LINK_PATTERN = re.compile(
    r"^(?:minta\s+|bagi\s+|spill\s+|mana\s+|ada\s+|kirim\s+)?(?:link|tautan|url)(?:nya|\s+nya|\s+dong|\s+min|\s+va|\s+plis)?\s*(?:apa|dong|deh|ya|kah|kan|ada|nggak|gak|ga)?\s*[?.]*$",
    re.IGNORECASE,
)


def detect_course_query(message: str, previous_query: str | None = None) -> str | None:
    """Detect if the user is asking for a course/class link and extract search keyword.
    
    Supports:
    1. Explicit link query in same turn: "minta link course collaborate to influence"
    2. Contextual follow-up turn: "Link nya" / "minta linknya" with previous_query from chat history.
    """
    raw = message.strip()
    if not raw:
        return None

    # Strip conversational noise
    clean = re.sub(r"^(?:ava|hai|halo|pagi|siang|sore|malam)\s*[,!.]?\s*", "", raw, flags=re.IGNORECASE).strip()

    # Case A: Follow-up turn asking for link of previous topic ("Link nya", "minta linknya")
    if previous_query and _FOLLOW_UP_LINK_PATTERN.match(clean):
        prev_clean = re.sub(r"^(?:tolong\s+)?(?:jelaskan(?:\s+tentang)?|apa\s+itu|apa\s+maksud(?:\s+dari)?|tentang)\s+", "", previous_query.strip(), flags=re.IGNORECASE).strip()
        prev_clean = re.sub(r"[?!.,]+$", "", prev_clean).strip()
        if len(prev_clean) >= 2:
            return prev_clean

    # Case B: Explicit mention in current turn
    has_link_marker = any(w in clean.lower() for w in ("link", "tautan", "url"))
    has_course_marker = any(w in clean.lower() for w in ("course", "kelas", "modul", "pelatihan", "training", "belajar", "materi"))

    if not (has_link_marker and has_course_marker):
        return None

    for pattern in _COURSE_LINK_PREFIX_PATTERNS:
        m = pattern.search(clean)
        if m:
            candidate = m.group(1).strip()
            # Remove trailing punctuation, questions, and conversational fluff
            for _ in range(2):
                candidate = re.sub(r"[?!.,]+$", "", candidate).strip()
                candidate = re.sub(r"\b(linknya\s+dimana|linknya\s+mana|linknya\s+apa|linknya|tautannya|urlnya)\b.*$", "", candidate, flags=re.IGNORECASE).strip()
                candidate = re.sub(r"\b(dong|deh|ya|kah|kan|min|ava|nya|plis|please|ada|nggak|ngga|gak|ga|mana|dimana|apa)\b$", "", candidate, flags=re.IGNORECASE).strip()
            candidate = re.sub(r"[?!.,]+$", "", candidate).strip()
            if len(candidate) >= 2:
                return candidate

    return None


async def search_courses(
    session: AsyncSession,
    query: str,
    limit: int = 3,
    min_similarity: float = 0.15,
) -> list[dict[str, Any]]:
    """Search active courses in PostgreSQL using pg_trgm similarity with ILIKE fallback."""
    q = query.strip()
    if not q:
        return []

    # 1. Primary: PostgreSQL pg_trgm similarity
    try:
        sql = text("""
            SELECT id, fullname, category, url,
                   GREATEST(similarity(fullname, :query), word_similarity(:query, fullname)) as sim
            FROM course_catalog
            WHERE status = 'Aktif'
              AND (similarity(fullname, :query) >= :min_sim OR word_similarity(:query, fullname) >= 0.25)
            ORDER BY sim DESC
            LIMIT :limit
        """)
        res = await session.execute(sql, {"query": q, "min_sim": min_similarity, "limit": limit})
        rows = res.fetchall()
        if rows:
            return [
                {
                    "id": r[0],
                    "fullname": r[1],
                    "category": r[2],
                    "url": r[3],
                    "score": float(r[4]),
                }
                for r in rows
            ]
    except Exception as exc:
        # Fallback if pg_trgm is not available (e.g. SQLite tests or extension error)
        logger.debug(f"pg_trgm search fallback due to: {exc}")

    # 2. Fallback: Word token ILIKE
    words = [w for w in re.split(r"\s+", q) if len(w) >= 3]
    if not words:
        words = [q]

    stmt = select(CourseCatalog).where(CourseCatalog.status == "Aktif")
    conditions = []
    for w in words[:4]:
        conditions.append(CourseCatalog.fullname.ilike(f"%{w}%"))
        conditions.append(CourseCatalog.category.ilike(f"%{w}%"))

    if conditions:
        stmt = stmt.where(or_(*conditions)).limit(limit)
        res = await session.execute(stmt)
        courses = res.scalars().all()
        return [
            {
                "id": c.id,
                "fullname": c.fullname,
                "category": c.category,
                "url": c.url,
                "score": 0.5,
            }
            for c in courses
        ]

    return []
