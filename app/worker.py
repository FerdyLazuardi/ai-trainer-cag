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

class WorkerSettings:
    """arq worker configuration."""
    redis_settings = RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        database=settings.redis_db,
        password=settings.redis_password if settings.redis_password else None,
    )
    functions = [sync_moodle_task, dummy_task]
    on_startup = startup
    on_shutdown = shutdown
