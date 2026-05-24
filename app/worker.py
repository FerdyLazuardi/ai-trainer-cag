import asyncio
import os
from typing import Any
from arq.connections import RedisSettings
from loguru import logger

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.database.postgres import AsyncSessionLocal
from app.ingestion.moodle_sync import sync_moodle_knowledge_base
from app.ingestion.portfolio_sync import sync_portfolio_knowledge_base
from app.observability import set_langfuse_client, get_langfuse_client

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


async def sync_portfolio_task(ctx: dict, force_reingest: bool = False) -> dict[str, Any]:
    """arq task to scrape ferdy-fadhil-lazuardi.my.id + CV into Personal_Portfolio."""
    logger.info(f"Starting Askfer portfolio sync via arq", force_reingest=force_reingest)
    try:
        async with AsyncSessionLocal() as session:
            summary = await sync_portfolio_knowledge_base(
                session=session,
                force_reingest=force_reingest,
            )
            await session.commit()
            logger.info(f"arq portfolio sync completed: {summary}")
            return summary
    except Exception as exc:
        logger.error(f"arq portfolio sync failed: {exc}")
        raise

async def _profile_watcher_task():
    """Watch `data/personal/profile.md` and auto-refresh on save.

    Spawned once at worker startup. Cancelled at shutdown. Debounced 1s so
    rapid editor saves collapse into a single refresh.
    """
    import os
    try:
        from watchfiles import awatch
    except ImportError:
        logger.warning("watchfiles not available — profile auto-refresh disabled")
        return

    profile_dir = "data/personal"
    target = "profile.md"
    os.makedirs(profile_dir, exist_ok=True)
    logger.info("Profile watcher started", path=profile_dir)

    try:
        # Polling is required when running inside a container with a bind mount
        # from a Windows/macOS host — inotify events aren't forwarded across
        # those filesystem boundaries. step=1000ms keeps reaction time ~1s.
        async for changes in awatch(
            profile_dir,
            debounce=1000,
            force_polling=True,
            poll_delay_ms=1000,
        ):
            # Filter: only react when profile.md itself changed.
            hit = any(
                p.replace("\\", "/").endswith(f"/{target}") or
                p.replace("\\", "/").endswith(target)
                for _, p in changes
            )
            if not hit:
                continue

            try:
                async with AsyncSessionLocal() as session:
                    from app.ingestion.portfolio_sync import refresh_profile_only
                    result = await refresh_profile_only(session)
                    await session.commit()
                logger.info(f"Profile auto-refreshed: {result}")
            except Exception as e:
                logger.warning(f"Profile auto-refresh failed: {e}")
    except asyncio.CancelledError:
        logger.info("Profile watcher cancelled (shutdown)")
        raise
    except Exception as e:
        logger.error(f"Profile watcher crashed: {e}")


async def startup(ctx: dict):
    """Initialize resources for the worker."""
    setup_logging(debug=settings.app_debug)
    logger.info("Worker starting up...")

    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
            os.environ["LANGFUSE_HOST"] = settings.langfuse_host

            from langfuse import Langfuse
            lf = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
                debug=settings.app_debug,
            )
            set_langfuse_client(lf)
            logger.info("Worker Langfuse v4 initialized")
        except Exception as e:
            logger.warning(f"Worker Langfuse init failed: {e}")
    else:
        logger.warning("Worker Langfuse keys not set — tracing disabled")

    try:
        from app.config.embedding_config import ensure_llamaindex_configured
        ensure_llamaindex_configured()
        logger.info("Worker embedding model pre-warmed")
    except Exception as e:
        logger.warning(f"Worker embedding pre-warm failed: {e}")

    # Spawn profile.md auto-refresh watcher (fire-and-forget).
    ctx["profile_watcher"] = asyncio.create_task(_profile_watcher_task())


async def shutdown(ctx: dict):
    """Cleanup resources for the worker."""
    logger.info("Worker shutting down...")

    # Cancel profile watcher first so its async generator unwinds cleanly
    # before we tear down the Langfuse client / event loop.
    task = ctx.get("profile_watcher")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Profile watcher shutdown error: {e}")

    lf = get_langfuse_client()
    if lf:
        try:
            lf.flush()
        except Exception as e:
            logger.warning(f"Worker Langfuse flush failed: {e}")

async def sync_ltm_task(ctx: dict, conversation_id: str, user_id: str) -> dict[str, Any]:
    """
    arq background task: persist a new LTM episode to Qdrant after 10-second AFK.

    Flow:
        1. Guard: check Redis dedup key — skip if another task already ran.
        2. Guard: check last_active timestamp — abort if user is still active.
        3. Get session summary from STM (get_or_summarize_history).
        4. Extract topics via cheap LLM.
        5. Upsert new episode to Qdrant via QdrantLTMService.update().
        6. Clean up Redis keys.
    """
    from app.database.redis_client import get_redis_client
    import time

    redis = get_redis_client()

    # ── Guard 1: Deduplication ────────────────────────────────────────────────
    # Only the FIRST task to acquire this key proceeds; subsequent ones abort.
    dedup_key = f"rag:ltm:syncing:{conversation_id}"
    acquired = await redis.set(dedup_key, "1", nx=True, ex=3600)   # 1-hour TTL
    if not acquired:
        logger.info("LTM sync: skipped (another task already running)", conversation_id=conversation_id)
        return {"status": "skipped", "reason": "dedup_lock"}

    # ── Guard 2: Activity check ───────────────────────────────────────────────
    last_active = await redis.get(f"rag:last_active:{conversation_id}")
    if last_active:
        time_since_active = time.time() - float(last_active)
        if time_since_active < 5:   # Quick test: check if user was active in last 5 seconds
            await redis.delete(dedup_key)          # release lock so retry can happen
            logger.info(
                "LTM sync: user still active, deferring task",
                conversation_id=conversation_id,
                seconds_since_active=round(time_since_active),
            )
            from arq import Retry
            raise Retry(defer=10)

    # ── Step 3: Get session summary ───────────────────────────────────────────
    from app.agents.memory import get_or_summarize_history
    from app.llm.client import get_cheap_llm
    from app.agents.long_term_memory_qdrant import qdrant_ltm

    cheap_llm = get_cheap_llm()
    summary, recent_history = await get_or_summarize_history(
        conversation_id=conversation_id,
        llm=cheap_llm,
        max_fresh_turns=10,
    )

    if not summary and not recent_history:
        await redis.delete(dedup_key)
        await redis.delete(f"rag:ltm:scheduled:{conversation_id}")
        return {"status": "skipped", "reason": "no_history"}

    # Generate a definitive session summary for LTM and extract preferences via structured output
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel, Field

    class LTMSummaryResult(BaseModel):
        summary: str = Field(
            description=(
                "All distinct topics discussed in the session. Max 30 words. "
                "Telegraphic style — drop articles (a/an/the) for token efficiency, "
                "since this is internal stored memory not shown to users. "
                "Example: 'User asked about Amartha products and Client Protection rules.'"
            )
        )
        role: str | None = Field(
            default=None,
            description="User's job/role if mentioned (e.g., 'Loan Officer'), else null."
        )
        preferred_tone: str | None = Field(
            default=None,
            description="Tone the user explicitly requested (e.g., 'formal', 'casual'), else null."
        )
        formatting_pref: str | None = Field(
            default=None,
            description="Output format the user explicitly requested (e.g., 'bullet points'), else null."
        )
        custom_instructions: str | None = Field(
            default=None,
            description="Any other specific instructions the user gave, else null."
        )

    raw_tail = "\n".join([f"{'User' if m.get('role') == 'user' else 'AI'}: {m.get('content', '')[:300]}" for m in recent_history])

    prompt = (
        "Analyze the following conversation and produce a structured session summary "
        "plus any user preferences inferred from it.\n\n"
        f"Previous Context:\n{summary}\n\n"
        f"Recent Conversation:\n{raw_tail}"
    )

    session_summary = ""
    prefs_data = None
    try:
        from langfuse.langchain import CallbackHandler
        lf_handler = CallbackHandler()
        structured = cheap_llm.with_structured_output(LTMSummaryResult)
        result = await structured.ainvoke(
            [HumanMessage(content=prompt)],
            config={"callbacks": [lf_handler], "run_name": "a-pedi-ltm-sync-summarize"}
        )
        session_summary = result.summary or ""
        prefs_data = {
            "role": result.role,
            "preferred_tone": result.preferred_tone,
            "formatting_pref": result.formatting_pref,
            "custom_instructions": result.custom_instructions,
        }
    except Exception as exc:
        logger.warning(f"LTM sync: structured summarization failed, falling back to raw concat: {exc}")
        session_summary = summary if summary else raw_tail[:1000]

    # Save preferences to PostgreSQL if any were detected
    if prefs_data and any(prefs_data.values()):
        try:
            from app.database.postgres import AsyncSessionLocal
            from app.database.models import UserProfile
            
            async with AsyncSessionLocal() as session:
                user_profile = await session.get(UserProfile, user_id)
                if not user_profile:
                    user_profile = UserProfile(user_id=user_id)
                    session.add(user_profile)
                
                if prefs_data.get("role"):
                    user_profile.role = prefs_data["role"]
                if prefs_data.get("preferred_tone"):
                    user_profile.preferred_tone = prefs_data["preferred_tone"]
                if prefs_data.get("formatting_pref"):
                    user_profile.formatting_pref = prefs_data["formatting_pref"]
                if prefs_data.get("custom_instructions"):
                    user_profile.custom_instructions = prefs_data["custom_instructions"]
                    
                await session.commit()
                logger.info("LTM sync: User preferences updated in PostgreSQL", user_id=user_id)
        except Exception as exc:
            logger.error(f"LTM sync: Failed to save preferences to PostgreSQL: {exc}")

    # ── Step 4+5: Extract course names + upsert to Qdrant ───────────────────────────
    # Course name extraction is done inside qdrant_ltm.update() when new_course_names=[]
    await qdrant_ltm.update(
        user_id=user_id,
        session_summary=session_summary,
        new_course_names=[],          # let _extract_course_names() handle this via llm=
        session_id=conversation_id,
        llm=cheap_llm,
    )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    await redis.delete(f"rag:last_active:{conversation_id}")
    await redis.delete(dedup_key)
    await redis.delete(f"rag:ltm:scheduled:{conversation_id}")

    logger.info("LTM sync: episode persisted to Qdrant (AFK)", conversation_id=conversation_id)
    return {"status": "synced"}

class WorkerSettings:
    """arq worker configuration."""
    redis_settings = RedisSettings(
        host=settings.redis_host,
        port=settings.redis_port,
        database=settings.redis_db,
        password=settings.redis_password if settings.redis_password else None,
    )
    functions = [sync_moodle_task, dummy_task, sync_ltm_task, sync_portfolio_task]
    on_startup = startup
    on_shutdown = shutdown
