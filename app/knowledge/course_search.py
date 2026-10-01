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
        r"(?:minta|bagi|spill|cari|tolong|ada)?\s*(?:link|tautan|url)\s+(?:course|kelas|modul|pelatihan|materi|training)?\s*(?:tentang|buat|untuk|soal)?\s*(.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:course|kelas|modul|pelatihan|training)\s+(?:tentang|buat|untuk|soal)?\s*(.+?)\s*(?:ada\s+link|linknya\s+apa|minta\s+link|link)?\s*[?.]*$",
        re.IGNORECASE,
    ),
]


def detect_course_query(message: str) -> str | None:
    """Detect if the user is asking for a course/class link and extract search keyword."""
    raw = message.strip()
    if not raw:
        return None

    # Strip conversational noise
    clean = re.sub(r"^(?:ava|hai|halo|pagi|siang|sore|malam)\s*[,!.]?\s*", "", raw, flags=re.IGNORECASE).strip()

    # Must contain anchor words indicating a link/course lookup
    has_link_marker = any(w in clean.lower() for w in ("link", "tautan", "url"))
    has_course_marker = any(w in clean.lower() for w in ("course", "kelas", "modul", "pelatihan", "training"))

    if not (has_link_marker and has_course_marker):
        return None

    for pattern in _COURSE_LINK_PREFIX_PATTERNS:
        m = pattern.search(clean)
        if m:
            candidate = m.group(1).strip()
            # Remove trailing punctuation and conversational fluff
            for _ in range(2):
                candidate = re.sub(r"[?!.,]+$", "", candidate).strip()
                candidate = re.sub(r"\b(dong|deh|ya|kah|kan|min|ava|nya|plis|please|ada|nggak|ngga|gak|ga)\b$", "", candidate, flags=re.IGNORECASE).strip()
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
            SELECT id, fullname, category, url, similarity(fullname, :query) as sim
            FROM course_catalog
            WHERE status = 'Aktif' AND similarity(fullname, :query) >= :min_sim
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
