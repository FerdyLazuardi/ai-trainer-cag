"""
Long-term memory service — persists user context across sessions.

Flow per request:
  Sesi baru → load()   → inject ke system prompt via user_profile state
  Setiap N turns → update() via background_tasks → merge ke PostgreSQL
"""
from datetime import datetime

from loguru import logger
from sqlalchemy import select

from app.api.user_utils import is_real_user
from app.database.models import UserMemory
from app.database.postgres import AsyncSessionLocal

MAX_TOPICS = 20


class LongTermMemoryService:

    async def load(self, user_id: str) -> dict:
        """
        Load user memory dari PostgreSQL.
        Return: {"summary": str, "topics": list[str]}
        Dipanggil sekali di awal sesi baru.
        """
        if not is_real_user(user_id=user_id, role="moodle_user"):
            return {"summary": "", "topics": []}
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(UserMemory).where(UserMemory.user_id == user_id)
                )
                row = result.scalar_one_or_none()
                if not row:
                    return {"summary": "", "topics": []}
                return {
                    "summary": row.summary or "",
                    "topics": row.topics or [],
                }
        except Exception as exc:
            logger.warning("Failed to load LTM", user_id=user_id, error=str(exc))
            return {"summary": "", "topics": []}

    async def update(
        self,
        user_id: str,
        session_summary: str,
        new_topics: list[str],
        llm=None,
    ) -> None:
        """
        Merge session summary ke PostgreSQL.
        Dipanggil di background setelah N turns.
        """
        if not is_real_user(user_id=user_id, role="moodle_user"):
            return
        if not session_summary:
            return
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(UserMemory).where(UserMemory.user_id == user_id)
                )
                row = result.scalar_one_or_none()

                if row:
                    merged_summary = await self._merge_summaries(
                        old=row.summary or "",
                        new=session_summary,
                        llm=llm,
                    )
                    merged_topics = list(
                        set((row.topics or []) + new_topics)
                    )[:MAX_TOPICS]
                    row.summary = merged_summary
                    row.topics = merged_topics
                    row.last_active = datetime.utcnow()
                else:
                    row = UserMemory(
                        user_id=user_id,
                        summary=session_summary[:2000],
                        topics=new_topics[:MAX_TOPICS],
                    )
                    session.add(row)

                await session.commit()
                logger.info("LTM updated", user_id=user_id)
        except Exception as exc:
            logger.warning("Failed to update LTM", user_id=user_id, error=str(exc))

    async def _merge_summaries(self, old: str, new: str, llm=None) -> str:
        """Merge dua summary. LLM jika tersedia, fallback ke new saja."""
        if not old:
            return new
        if not llm:
            return new
        try:
            from langchain_core.messages import HumanMessage
            prompt = (
                "Gabungkan dua ringkasan berikut menjadi SATU ringkasan 3-4 kalimat. "
                "Pertahankan informasi penting dari keduanya. Gunakan bahasa yang sama.\n\n"
                f"Ringkasan lama:\n{old}\n\nRingkasan baru:\n{new}\n\nRingkasan gabungan:"
            )
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception:
            return new


long_term_memory = LongTermMemoryService()
