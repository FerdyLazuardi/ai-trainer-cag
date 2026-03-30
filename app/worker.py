import asyncio
from typing import Any
from arq.connections import RedisSettings
from loguru import logger

from app.config.settings import get_settings
from app.database.postgres import AsyncSessionLocal
from app.ingestion.moodle_sync import sync_moodle_knowledge_base

settings = get_settings()

async def sync_moodle_task(ctx: dict, course_id: int | None, target_sections: list[str] | None, force_reingest: bool) -> dict[str, Any]:
    """arq task to run the moodle sync."""
    logger.info(f"Starting background Moodle sync task via arq for course_id={course_id}", force_reingest=force_reingest)
    try:
        async with AsyncSessionLocal() as session:
            summary = await sync_moodle_knowledge_base(
                session=session,
                course_id=course_id,
                target_sections=target_sections,
                force_reingest=force_reingest
            )
            # Commit any changes made by the sync
            await session.commit()
            
            logger.info(f"arq Moodle sync completed: {summary}")
            return summary
    except Exception as exc:
        logger.error(f"arq Moodle sync failed: {exc}")
        raise

async def dummy_task(ctx: dict, name: str) -> str:
    """A dummy task to verify the worker is running."""
    logger.info(f"Running dummy task for {name}")
    await asyncio.sleep(1)
    return f"Hello, {name}! Task completed."

async def startup(ctx: dict):
    """Initialize resources for the worker."""
    logger.info("Worker starting up...")

async def shutdown(ctx: dict):
    """Cleanup resources for the worker."""
    logger.info("Worker shutting down...")

async def sync_ltm_task(ctx: dict, conversation_id: str, user_id: str) -> dict[str, Any]:
    """arq task to run LTM summarization after 30 mins of AFK."""
    from app.database.redis_client import get_redis_client
    import time
    
    redis = get_redis_client()
    # Check if the user has been active recently
    last_active = await redis.get(f"rag:last_active:{conversation_id}")
    if last_active:
        time_since_active = time.time() - float(last_active)
        # If they were active within the last 30 minutes (minus 1 min buffer), abort
        if time_since_active < (30 * 60) - 60:
            logger.info("User still active, aborting LTM sync task", conversation_id=conversation_id)
            return {"status": "aborted", "reason": "still_active"}
            
    # AFK for 30 minutes confirmed! Let's summarize
    from app.agents.memory import get_or_summarize_history
    from app.llm.client import get_cheap_llm
    from app.agents.long_term_memory import long_term_memory
    
    cheap_llm = get_cheap_llm()
    summary, recent_history = await get_or_summarize_history(
        conversation_id=conversation_id,
        llm=cheap_llm,
        max_fresh_turns=10, 
    )

    if not summary and not recent_history:
        return {"status": "skipped", "reason": "no_history"}

    session_summary_for_ltm = summary if summary else " ".join([m["content"] for m in recent_history])[:1000]

    await long_term_memory.update(
        user_id=user_id,
        session_summary=session_summary_for_ltm,
        new_topics=[],
        llm=cheap_llm,
    )
    
    # Clean up the last active key
    await redis.delete(f"rag:last_active:{conversation_id}")
    logger.info("AFK LTM Sync Complete", conversation_id=conversation_id)
    return {"status": "synced"}

class WorkerSettings:
    """arq worker configuration."""
    redis_settings = RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        database=settings.redis_db,
        password=settings.redis_password if settings.redis_password else None,
    )
    functions = [sync_moodle_task, dummy_task, sync_ltm_task]
    on_startup = startup
    on_shutdown = shutdown
