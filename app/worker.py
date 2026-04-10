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

    # Generate a definitive session summary for LTM and extract preferences
    from langchain_core.messages import HumanMessage
    import json
    raw_tail = "\n".join([f"{'User' if m.get('role') == 'user' else 'AI'}: {m.get('content', '')[:300]}" for m in recent_history])
    
    prompt = (
        "Analisis percakapan berikut dan berikan output dalam format JSON strict dengan struktur berikut:\n"
        "{\n"
        '  "summary": "Ringkasan 1-2 kalimat fokus pada topik/fakta yang dibahas",\n'
        '  "preferences": {\n'
        '    "role": "Jabatan/profesi user jika disebutkan (misal: Loan Officer), null jika tidak ada",\n'
        '    "preferred_tone": "Gaya bahasa yang diminta (misal: formal, santai, dsb), null jika tidak ada",\n'
        '    "formatting_pref": "Format jawaban yang diminta (misal: bullet points, paragraf pendek, dsb), null jika tidak ada",\n'
        '    "custom_instructions": "Instruksi spesifik lainnya, null jika tidak ada"\n'
        "  }\n"
        "}\n\n"
        f"Konteks Sebelumnya:\n{summary}\n\n"
        f"Percakapan Terbaru:\n{raw_tail}\n\n"
        "Output JSON:"
    )
    
    session_summary = ""
    prefs_data = None
    try:
        resp = await cheap_llm.ainvoke([HumanMessage(content=prompt)])
        content = resp.content.strip()
        # Remove markdown code blocks if any
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        parsed = json.loads(content)
        session_summary = parsed.get("summary", "")
        prefs_data = parsed.get("preferences", {})
    except Exception as exc:
        logger.warning(f"LTM sync: LLM summarization/extraction failed, falling back to raw concat: {exc}")
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
    functions = [sync_moodle_task, dummy_task, sync_ltm_task]
    on_startup = startup
    on_shutdown = shutdown
