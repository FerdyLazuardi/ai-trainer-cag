from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import KnowledgeBasePack


async def get_active_kb_pack(
    session: AsyncSession,
    *,
    source: str = "moodle",
) -> KnowledgeBasePack | None:
    result = await session.execute(
        select(KnowledgeBasePack)
        .where(KnowledgeBasePack.source == source, KnowledgeBasePack.is_active.is_(True))
        .order_by(KnowledgeBasePack.created_at.desc(), KnowledgeBasePack.id.desc())
        .limit(1)
    )
    return result.scalars().first()


async def save_active_kb_pack(
    session: AsyncSession,
    *,
    source: str,
    kb_hash: str,
    content: str,
    token_count: int | None = None,
) -> KnowledgeBasePack:
    active = await get_active_kb_pack(session, source=source)
    if active and active.kb_hash == kb_hash:
        return active

    await session.execute(
        update(KnowledgeBasePack)
        .where(KnowledgeBasePack.source == source, KnowledgeBasePack.is_active.is_(True))
        .values(is_active=False)
    )
    pack = KnowledgeBasePack(
        source=source,
        kb_hash=kb_hash,
        content=content,
        token_count=token_count,
        is_active=True,
    )
    session.add(pack)
    await session.flush()
    return pack
