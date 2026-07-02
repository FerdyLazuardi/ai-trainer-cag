from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge.kb_pack import assemble_kb_pack
from app.knowledge.moodle_markdown import pull_moodle_markdown
from app.knowledge.store import get_active_kb_pack, save_active_kb_pack
from app.utils.cache import flush_cache
from app.utils.token_counter import count_tokens


async def sync_moodle_kb_pack(
    *,
    session: AsyncSession,
    course_id: int | None = None,
    target_sections: list[str] | None = None,
) -> dict[str, Any]:
    from app.config.settings import get_settings

    settings = get_settings()
    cid = course_id or settings.cag_moodle_course_id
    docs = await pull_moodle_markdown([cid], target_sections=target_sections)
    pack = assemble_kb_pack(docs)
    active = await get_active_kb_pack(session, source=settings.cag_kb_source)
    unchanged = bool(active and active.kb_hash == pack.kb_hash)
    saved = await save_active_kb_pack(
        session,
        source=settings.cag_kb_source,
        kb_hash=pack.kb_hash,
        content=pack.text,
        token_count=count_tokens(pack.text),
    )
    if not unchanged:
        await flush_cache()
        from app.graph.pipeline import clear_cag_kb_cache
        clear_cag_kb_cache()
    return {
        "status": "unchanged" if unchanged else "updated",
        "pack_id": saved.id,
        "kb_hash": saved.kb_hash,
        "files": len(docs),
        "token_count": saved.token_count,
    }
