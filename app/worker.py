import asyncio
import zoneinfo
from contextlib import asynccontextmanager
from datetime import timedelta, timezone
from typing import Any

from loguru import logger
from streaq import Worker

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.database.postgres import AsyncSessionLocal
from app.knowledge.sync import sync_moodle_kb_pack

settings = get_settings()

try:
    JakartaTz: Any = zoneinfo.ZoneInfo("Asia/Jakarta")
except Exception:
    JakartaTz = timezone(timedelta(hours=7), name="WIB")


@asynccontextmanager
async def _worker_lifespan():
    await startup({})
    try:
        yield
    finally:
        await shutdown({})


worker = Worker(
    redis_url=settings.streaq_redis_url,
    concurrency=2,
    priorities=["low", "high"],
    lifespan=_worker_lifespan,
    tz=JakartaTz,
    signing_secret=settings.streaq_signing_secret or None,
)


@worker.task(max_tries=1, timeout=60)
async def ingest_text_task(text: str, title: str, source: str, metadata: dict) -> dict[str, Any]:
    return {"status": "disabled", "reason": "raw_rag_ingestion_disabled_in_cag"}


@worker.task(max_tries=3, timeout=600)
async def sync_moodle_task(
    course_id: int | None,
    target_sections: list[str] | None,
    force_reingest: bool,
) -> dict[str, Any]:
    async with AsyncSessionLocal() as session:
        summary = await sync_moodle_kb_pack(
            session=session,
            course_id=course_id,
            target_sections=target_sections,
        )
        await session.commit()
        logger.info(f"CAG Moodle KB sync completed: {summary}")
        return summary


@worker.task(max_tries=1, timeout=900)
async def sync_spreadsheet_task() -> dict[str, Any]:
    """Paginated GAS spreadsheet sync (10k users, ~4MB/page).

    Runs in the worker so the API never holds a request past the Cloudflare
    120s proxy timeout. Timeout 900s covers ~10 user pages + N branch pages
    plus staging upserts.
    """
    from app.knowledge.sync_spreadsheet import sync_kpi_from_spreadsheet
    import json
    from datetime import datetime

    async with AsyncSessionLocal() as session:
        result = await sync_kpi_from_spreadsheet(session)
        logger.info(f"Spreadsheet sync completed: {result}")

        # Update Redis schedule state with last run details
        try:
            from app.database.redis_client import get_redis_client
            redis = get_redis_client()
            now_iso = datetime.now(JakartaTz).isoformat()
            raw = await redis.get("cag:spreadsheet:schedule")
            sched = json.loads(raw) if raw else {
                "enabled": False,
                "schedule_type": "daily",
                "hour": 2,
                "minute": 0,
                "day_of_week": 1,
            }
            sched["last_run_at"] = now_iso
            sched["last_status"] = result.get("status", "unknown")
            sched["last_result"] = result
            await redis.set("cag:spreadsheet:schedule", json.dumps(sched))
        except Exception as exc:
            logger.warning(f"Failed to record spreadsheet sync status to Redis: {exc}")

        return result


@worker.task
async def dummy_task(name: str) -> str:
    logger.info(f"Running dummy task for {name}")
    await asyncio.sleep(1)
    return f"Hello, {name}! Task completed."


@worker.task(max_tries=2, timeout=60)
async def summarize_refresh_task(conversation_id: str) -> dict[str, Any]:
    from app.agents.conversation_state import _SUMMARY_REFRESH_PREFIX, get_or_summarize_history
    from app.database.redis_client import get_redis_client
    from app.llm.client import get_cheap_llm

    redis = get_redis_client()
    try:
        from app.llm.client import get_stm_llm
        await get_or_summarize_history(
            redis,
            conversation_id,
            llm=get_stm_llm(),
            max_fresh_turns=settings.max_fresh_turns,
            persist=True,
        )
        return {"status": "refreshed", "conversation_id": conversation_id}
    finally:
        await redis.delete(f"{_SUMMARY_REFRESH_PREFIX}{conversation_id}")


@worker.task(max_tries=2, timeout=90)
async def sync_ltm_task(conversation_id: str, user_id: str) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage
    from app.agents.conversation_state import get_history_and_summary
    from app.database.models import UserLTMMemory
    from app.database.postgres import AsyncSessionLocal
    from app.database.redis_client import get_redis_client
    from app.knowledge.kb_pack import extract_h2_headings
    from app.llm.client import get_cheap_llm

    redis = await get_redis_client()
    history, stm_summary = await get_history_and_summary(redis, conversation_id)

    if not history and not stm_summary:
        return {"status": "skipped", "reason": "empty_session"}

    dialogue_str = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'AI'}: {m.get('content', '')}"
        for m in history
    )
    h2_headings = extract_h2_headings(dialogue_str)

    session_parts = []
    if stm_summary:
        session_parts.append(f"[EARLIER TURNS SUMMARY]:\n{stm_summary}")
    if dialogue_str:
        session_parts.append(f"[RECENT DIALOGUE TURNS]:\n{dialogue_str}")
    combined_session_summary = "\n\n".join(session_parts) if session_parts else "No session activity."

    async with AsyncSessionLocal() as session:
        memory = await session.get(UserLTMMemory, user_id)
        if not memory:
            memory = UserLTMMemory(user_id=user_id)
            session.add(memory)

        old_learning_summary = memory.learning_summary or "Belum ada riwayat profil belajar sebelumnya."

        from app.llm.prompts import LTM_LEARNING_SUMMARY_PROMPT
        prompt = LTM_LEARNING_SUMMARY_PROMPT.format(
            old_learning_summary=old_learning_summary,
            session_summary=combined_session_summary,
        )

        from app.llm.client import get_ltm_llm
        llm = get_ltm_llm()
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        raw_output = resp.content.strip()

        new_learning_summary = raw_output
        import json as _json
        import re as _re
        try:
            clean_json = raw_output
            if clean_json.startswith("```"):
                clean_json = _re.sub(r"^```(?:json)?\n?|\n?```$", "", clean_json, flags=_re.IGNORECASE).strip()
            parsed = _json.loads(clean_json)
            if isinstance(parsed, dict) and "learning_summary" in parsed:
                new_learning_summary = str(parsed["learning_summary"])
        except Exception:
            pass

        memory.learning_summary = new_learning_summary
        await session.commit()

    return {
        "status": "synced",
        "user_id": user_id,
        "conversation_id": conversation_id,
    }


async def prune_ltm_cron_task() -> dict[str, Any]:
    return {"status": "skipped", "reason": "ltm_disabled_in_cag"}


async def prune_agent_logs_cron_task() -> dict[str, Any]:
    from sqlalchemy import text

    days = settings.agent_log_retention_days
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text("DELETE FROM agent_logs WHERE created_at < NOW() - make_interval(days => :days)"),
            {"days": days},
        )
        deleted = int(getattr(result, "rowcount", 0) or 0)
        await session.commit()
    return {"status": "pruned", "deleted_rows": deleted, "retention_days": days}


async def startup(ctx: dict):
    setup_logging(debug=settings.app_debug)
    logger.info("CAG worker starting")


async def shutdown(ctx: dict):
    logger.info("CAG worker shutting down")


@worker.cron("0 2 * * *", timeout=60)
async def _run_ltm_prune():
    await prune_ltm_cron_task()


@worker.cron("0 4 * * *", timeout=300)
async def _run_agent_logs_prune():
    await prune_agent_logs_cron_task()


@worker.cron("0 */6 * * *", timeout=600)
async def _run_sync_moodle():
    # Sync Moodle KB pack every 6 hours
    await sync_moodle_task(course_id=None, target_sections=None, force_reingest=False)


@worker.cron("0 * * * *", timeout=900)
async def _run_scheduled_spreadsheet_sync():
    """Check hourly if spreadsheet auto-sync is scheduled and due in WIB."""
    try:
        from app.database.redis_client import get_redis_client
        import json
        from datetime import datetime

        redis = get_redis_client()
        schedule_raw = await redis.get("cag:spreadsheet:schedule")
        if not schedule_raw:
            return

        sched = json.loads(schedule_raw)
        if not sched.get("enabled"):
            return

        now = datetime.now(JakartaTz)
        target_hour = int(sched.get("hour", 2))
        if now.hour != target_hour:
            return

        sched_type = sched.get("schedule_type", "daily")
        if sched_type == "weekly":
            target_dow = int(sched.get("day_of_week", 1))  # 0=Monday, 6=Sunday
            if now.weekday() != target_dow:
                return

        # Anti-double-run: check if already run within the last 6 hours
        last_run = sched.get("last_run_at")
        if last_run:
            last_dt = datetime.fromisoformat(last_run)
            if (now - last_dt).total_seconds() < 21600:
                return

        logger.info(f"Triggering scheduled spreadsheet sync at hour {now.hour} WIB")
        await sync_spreadsheet_task()
    except Exception as exc:
        logger.error(f"Error in scheduled spreadsheet sync check: {exc}")



async def _eval_turn_task_fn(**kwargs) -> dict[str, Any]:
    from app.eval.tasks import eval_turn_task as _run_eval_turn_task

    return await _run_eval_turn_task(**kwargs)


eval_turn_task = worker.task(
    _eval_turn_task_fn,
    name="eval_turn_task",
    max_tries=2,
    timeout=120,
)  # type: ignore[call-overload]
