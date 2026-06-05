import asyncio
from contextlib import asynccontextmanager
from typing import Any

from loguru import logger
from streaq import StreaqRetry, Worker

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.database.postgres import AsyncSessionLocal, prune_agent_logs
from app.eval.tasks import eval_turn_task as _eval_turn_task_fn
from app.ingestion.moodle_sync import sync_moodle_knowledge_base
from app.ingestion.pipeline import ingest_document
from app.ingestion.portfolio_sync import sync_portfolio_knowledge_base
# Phoenix removed Jun 2026 — observability is a no-op shim now. The previous
# `from app.observability import setup_phoenix, flush as flush_traces,
#  is_observability_enabled` import is gone since setup_phoenix is gone too.

settings = get_settings()

# Streaq worker configuration.
# arq → streaq migration (Jun 2026): streaq uses coredis (separate from
# redis-py used by the app's 50+ direct Redis call sites), so this
# doesn't constrain our redis-py 8 upgrade path.


@asynccontextmanager
async def _worker_lifespan():
    """Bridge the arq-style startup/shutdown hooks to streaq's lifespan.

    streaq's Worker takes a single async context manager (lifespan)
    instead of arq's separate on_startup/on_shutdown callbacks. The
    original startup(ctx) / shutdown(ctx) bodies don't use ctx, so we
    pass an empty dict to preserve the existing function signatures
    unchanged.
    """
    await startup({})
    try:
        yield
    finally:
        await shutdown({})


worker = Worker(
    redis_url=settings.redis_url,
    concurrency=2,  # was arq's max_jobs=2
    lifespan=_worker_lifespan,
)


@worker.task
async def ingest_text_task(text: str, title: str, source: str, metadata: dict) -> dict[str, Any]:
    """streaq task: ingest a single document off the API request path.

    The text payload is bounded at 200,000 chars by IngestRequest validation,
    so the worker's embedding + Qdrant upsert runs without OOM risk. The
    task opens its own AsyncSession (the API process no longer holds one
    for the duration of the embedding pipeline).
    """
    logger.info("streaq ingest_text_task starting", title=title, source=source, text_len=len(text))
    try:
        async with AsyncSessionLocal() as session:
            result = await ingest_document(
                text=text,
                session=session,
                metadata=metadata,
                title=title,
                source=source,
            )
            await session.commit()
        logger.info(
            "streaq ingest_text_task complete",
            document_id=result.document_id,
            chunks=result.chunks_count,
            tokens=result.total_tokens,
        )
        return {
            "document_id": result.document_id,
            "chunks_count": result.chunks_count,
            "total_tokens": result.total_tokens,
        }
    except Exception as exc:
        logger.error("streaq ingest_text_task failed", error=str(exc))
        raise


@worker.task
async def sync_moodle_task(course_id: int | None, target_sections: list[str] | None, force_reingest: bool) -> dict[str, Any]:
    """streaq task to run the moodle sync."""
    logger.info(f"Starting background Moodle sync task via streaq for course_id={course_id}", force_reingest=force_reingest)
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
            
            logger.info(f"streaq Moodle sync completed: {summary}")
            return summary
    except Exception as exc:
        logger.error(f"streaq Moodle sync failed: {exc}")
        raise

@worker.task
async def dummy_task(name: str) -> str:
    """A dummy task to verify the worker is running."""
    logger.info(f"Running dummy task for {name}")
    await asyncio.sleep(1)
    return f"Hello, {name}! Task completed."


@worker.task
async def sync_portfolio_task(force_reingest: bool = False) -> dict[str, Any]:
    """streaq task to scrape ferdy-fadhil-lazuardi.my.id + CV into Personal_Portfolio."""
    logger.info("Starting Askfer portfolio sync via streaq", force_reingest=force_reingest)
    try:
        async with AsyncSessionLocal() as session:
            summary = await sync_portfolio_knowledge_base(
                session=session,
                force_reingest=force_reingest,
            )
            await session.commit()
            logger.info(f"streaq portfolio sync completed: {summary}")
            return summary
    except Exception as exc:
        logger.error(f"streaq portfolio sync failed: {exc}")
        raise

async def _profile_watcher_task():
    """Watch `data/personal/profile.md` and auto-refresh on save.

    NOT auto-spawned since v3.1 (see `startup` docstring). Kept here
    as a manual helper if the operator wants to wire it back: spawn
    it from a one-off script or re-add the asyncio.create_task call
    in `startup` after flipping INGEST_AUTO_ENABLED-style env.

    The function body is dead code at runtime (no caller). Keep it
    as reference for the watchfiles pattern + debounce + filter
    logic; do NOT call it from production startup.

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

    # Skip file watching in production — no one edits profile.md there,
    # and the 1000ms poll loop burns CPU on every worker heartbeat.
    if settings.app_env == "production":
        logger.info("Profile watcher disabled in production (app_env=production)")
        return

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
    """Initialize resources for the worker.

    NOTE on ingestion: as of v3.1 all chunk ingestion is MANUAL only.
    The previous `_profile_watcher_task` file-watcher (which auto-
    refreshed Personal_Portfolio when `data/personal/profile.md`
    changed) has been removed. Ingestion paths now:

      - Moodle KB     → POST /api/v1/admin/moodle-sync    (force_reingest flag)
      - Askfer port.  → POST /api/v1/askfer/sync           (force_reingest flag)
      - One-off text  → POST /api/v1/ingest/text
      - CLI (portfolio) → `uv run python -m app.ingestion.portfolio_sync --force`

    Rationale: auto-ingest on every worker boot (or every profile.md
    save) caused the 2× duplicate chunks seen in the b14 KB audit
    (lifespan + cron both fired). With manual-only, the operator
    decides exactly when a re-ingest runs and the hash-based dedup
    in `moodle_sync._ingest_markdown` is the only path.
    """
    setup_logging(debug=settings.app_debug)
    logger.info("Worker starting up...")

    try:
        from app.config.embedding_config import ensure_llamaindex_configured
        ensure_llamaindex_configured()
        logger.info("Worker embedding model pre-warmed")
    except Exception as e:
        logger.warning(f"Worker embedding pre-warm failed: {e}")


async def shutdown(ctx: dict):
    """Cleanup resources for the worker."""
    logger.info("Worker shutting down...")

@worker.task
async def sync_ltm_task(conversation_id: str, user_id: str) -> dict[str, Any]:
    """
    streaq background task: persist a new LTM episode to Qdrant after 10-second AFK.

    Flow:
        1. Guard: check Redis dedup key — skip if another task already ran.
        2. Guard: check last_active timestamp — abort if user is still active.
        3. Get session summary from STM (get_or_summarize_history).
        4. Extract topics via cheap LLM.
        5. Upsert new episode to Qdrant via QdrantLTMService.update().
        6. Clean up Redis keys.
    """
    from app.database.redis_client import get_redis_client
    from app.agents.conversation_state import (
        acquire_ltm_lock, release_ltm_lock, get_last_active,
        clear_conversation, get_courses,
    )
    import time

    redis = get_redis_client()

    # ── Guard 1: Deduplication ────────────────────────────────────────────────
    # Only the FIRST task to acquire this lock proceeds; subsequent ones abort.
    # The lock lives at `rag:ltm:syncing:{id}` (STRING, kept separate from the
    # conversation HASH so it can outlive a HASH expiry while a slow worker
    # is still finishing). 5-min TTL — task completes in <30s; shorter TTL
    # means a crashed task unblocks within 5min instead of 1h.
    if not await acquire_ltm_lock(redis, conversation_id):
        logger.info("LTM sync: skipped (another task already running)", conversation_id=conversation_id)
        return {"status": "skipped", "reason": "dedup_lock"}

    # ── Guard 2: Activity check ───────────────────────────────────────────────
    # If the user has interacted within `ltm_afk_threshold_seconds`, they're
    # not really AFK yet — defer the job until the full quiet window has
    # elapsed since their last activity. We schedule the next attempt for
    # exactly the remaining gap so we don't poll multiple times.
    last_active = await get_last_active(redis, conversation_id)
    if last_active is not None:
        time_since_active = time.time() - last_active
        afk_threshold = settings.ltm_afk_threshold_seconds
        if time_since_active < afk_threshold:
            await release_ltm_lock(redis, conversation_id)  # release lock so retry can happen
            remaining = max(60, int(afk_threshold - time_since_active))
            logger.info(
                "LTM sync: user still active, deferring task",
                conversation_id=conversation_id,
                seconds_since_active=round(time_since_active),
                retry_in_s=remaining,
            )
            raise StreaqRetry(delay=remaining)

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
        await release_ltm_lock(redis, conversation_id)
        await clear_conversation(redis, conversation_id)
        return {"status": "skipped", "reason": "no_history"}

    # Generate a definitive session summary for LTM and extract preferences via structured output
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel, Field

    class LTMSummaryResult(BaseModel):
        summary: str = Field(
            description=(
                "All distinct topics discussed in the session. Max 15 words. "
                "Telegraphic style — drop articles (a/an/the) for token efficiency, "
                "since this is internal stored memory not shown to users. "
                "Example: 'User asked about Amartha products and Client Protection rules.'"
            )
        )
        unanswered_questions: list[str] = Field(
            default_factory=list,
            description=(
                "Questions the user asked that the AI did NOT answer (gave 'belum "
                "menemukan info', refused, or never addressed). Helps continue "
                "future sessions. Max 3 entries. Empty list if everything was answered."
            ),
        )
        role: str | None = Field(
            default=None,
            description=(
                "User's specific Amartha job role ONLY if they explicitly stated it as "
                "their identity (e.g., 'aku Loan Officer', 'I work as a Field Officer', "
                "'gw BP', 'aku Business Partner'). "
                "Must be a concrete Amartha role: BP/Business Partner, FO/Field Officer, "
                "BM/Branch Manager, HO/Head Office, Loan Officer, etc. "
                "REJECT generic words like 'User', 'Karyawan', 'Pegawai', 'Staff', 'Employee'. "
                "Do NOT infer from context. Else null."
            )
        )
        preferred_tone: str | None = Field(
            default=None,
            description=(
                "User's STANDING tone preference. Set ONLY if the user used an "
                "explicit always-keyword: 'selalu', 'always', 'mulai sekarang', "
                "'from now on', 'going forward', 'setiap kali'. "
                "REJECT one-off requests like 'jelasin pakai bahasa awam', "
                "'tolong diringkas', 'jawab singkat aja' — those apply to ONE turn. "
                "REJECT inferences from how the user happens to phrase a question. "
                "Else null."
            )
        )
        formatting_pref: str | None = Field(
            default=None,
            description=(
                "User's STANDING formatting preference. Set ONLY if the user used "
                "an explicit always-keyword (selalu/always/mulai sekarang/from now on). "
                "Examples that QUALIFY: 'selalu jawab dalam bullet point', "
                "'always use a table', 'mulai sekarang kasih jawaban singkat'. "
                "REJECT one-off requests like 'tolong diringkas', 'pakai bullet dong'. "
                "Else null."
            )
        )
        custom_instructions: str | None = Field(
            default=None,
            description=(
                "Persistent rules for ALL future answers. Set ONLY if the user "
                "explicitly used an always-keyword AND it's a durable rule. "
                "Examples that QUALIFY: 'selalu cite sumber halaman', "
                "'jangan pakai jargon teknis dari sekarang'. "
                "REJECT a stylistic preference inferred from one terse message. "
                "REJECT instructions on ONE specific topic ('jangan jawab pertanyaan "
                "tentang X'). Else null."
            )
        )

    raw_tail = "\n".join([f"{'User' if m.get('role') == 'user' else 'AI'}: {m.get('content', '')[:300]}" for m in recent_history])

    prompt = (
        "Analyze the following conversation and produce a structured session summary "
        "plus any user preferences inferred from it.\n\n"
        "CRITICAL RULES for preferences (role, preferred_tone, formatting_pref, "
        "custom_instructions):\n"
        "- Set a value ONLY when the user EXPLICITLY stated a STANDING rule using "
        "always-keywords: selalu / always / mulai sekarang / from now on / setiap kali / "
        "going forward.\n"
        "- One-off requests stay null. Examples of REJECT (must be null): "
        "'jelasin pakai bahasa awam', 'tolong diringkas', 'jawab singkat dong', "
        "'pakai bullet point', 'in English please'.\n"
        "- For role: must be a concrete Amartha role name (BP, FO, BM, Loan Officer, "
        "Branch Manager, etc.) explicitly self-identified by the user. REJECT generic "
        "labels like 'User', 'Karyawan', 'Pegawai', 'Staff'.\n"
        "- Do NOT infer preferences from how the AI happened to respond, only from "
        "what the user EXPLICITLY asked for going forward.\n"
        "- When in doubt → null. False-positives are worse than false-negatives "
        "because preferences persist across all future sessions.\n\n"
        f"Previous Context:\n{summary}\n\n"
        f"Recent Conversation:\n{raw_tail}"
    )

    session_summary = ""
    prefs_data = None
    unanswered_questions: list[str] = []
    try:
        structured = cheap_llm.with_structured_output(LTMSummaryResult)
        result = await structured.ainvoke(
            [HumanMessage(content=prompt)],
            config={"run_name": "a-pedi-ltm-sync-summarize"}
        )
        session_summary = result.summary or ""
        unanswered_questions = (result.unanswered_questions or [])[:3]
        prefs_data = {
            "role": result.role,
            "preferred_tone": result.preferred_tone,
            "formatting_pref": result.formatting_pref,
            "custom_instructions": result.custom_instructions,
        }
    except Exception as exc:
        logger.warning(f"LTM sync: structured summarization failed, falling back to raw concat: {exc}")
        session_summary = summary if summary else raw_tail[:1000]

    # Course names: read directly from the `courses` field of the conversation
    # HASH, populated by the chat handler from `chunk.metadata.course_name` (KB
    # ground truth). NEVER ask the LLM — it hallucinates names like "Amartha
    # products" that aren't in the KB.
    try:
        course_names = (await get_courses(redis, conversation_id))[:3]
    except Exception as exc:
        logger.warning(f"LTM sync: failed to read course_names from HASH: {exc}")
        course_names = []

    # Save preferences to PostgreSQL if any were detected
    if prefs_data and any(prefs_data.values()):
        try:
            from app.database.models import UserProfile
            from datetime import datetime, timezone

            async with AsyncSessionLocal() as session:
                user_profile = await session.get(UserProfile, user_id)
                if not user_profile:
                    user_profile = UserProfile(user_id=user_id)
                    session.add(user_profile)

                # Write any non-null prefs. For fields that were explicitly
                # set to None by the LLM (e.g. user said "stop using formal tone"),
                # we check if the key is present in prefs_data — present+None means
                # explicit clear; missing key means no signal (leave unchanged).
                if "role" in prefs_data:
                    user_profile.role = prefs_data["role"]
                if "preferred_tone" in prefs_data:
                    user_profile.preferred_tone = prefs_data["preferred_tone"]
                if "formatting_pref" in prefs_data:
                    user_profile.formatting_pref = prefs_data["formatting_pref"]
                if "custom_instructions" in prefs_data:
                    user_profile.custom_instructions = prefs_data["custom_instructions"]

                # Touch updated_at so the stale-prefs filter knows when this was last confirmed.
                user_profile.updated_at = datetime.now(timezone.utc)

                await session.commit()
                logger.info("LTM sync: User preferences updated in PostgreSQL", user_id=user_id)
        except Exception as exc:
            logger.error(f"LTM sync: Failed to save preferences to PostgreSQL: {exc}")

    # ── Step 4+5: Upsert to Qdrant with course_names + unanswered_questions inline ───
    # course_names already extracted by the structured output above — no second LLM hop.
    await qdrant_ltm.update(
        user_id=user_id,
        session_summary=session_summary,
        new_course_names=course_names,
        unanswered_questions=unanswered_questions,
        session_id=conversation_id,
        llm=cheap_llm,
    )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    # One DEL on the HASH drops all 5 ephemeral fields (history, summary,
    # last_active, scheduled, courses) atomically. The owner field is left
    # intact — it has its own 7d TTL and a real user reclaiming the same
    # conversation_id should not have their ownership erased by a worker
    # that just synced their LTM.
    await clear_conversation(redis, conversation_id)
    await release_ltm_lock(redis, conversation_id)

    logger.info("LTM sync: episode persisted to Qdrant (AFK)", conversation_id=conversation_id)
    return {"status": "synced"}

async def prune_ltm_cron_task() -> dict[str, Any]:
    """Cron task: run daily to prune LTM vectors older than 60 days across all users."""
    logger.info("Starting global LTM pruning cron job")
    try:
        from app.agents.long_term_memory_qdrant import qdrant_ltm
        deleted = await qdrant_ltm.prune_global_inactive_episodes(days_old=60.0)
        return {"status": "pruned", "deleted_episodes": deleted}
    except Exception as exc:
        logger.error(f"Global LTM pruning failed: {exc}")
        raise


async def prune_agent_logs_cron_task() -> dict[str, Any]:
    """Cron task: daily sweep that deletes agent_logs rows older than 30 days.

    Prevents unbounded growth — at 600 DAU × 8 q/day × ~5KB JSONB rows = ~24MB/day,
    which would OOM the 768MB Postgres cgroup in 60-90 days. Retention is set to
    30 days (1.5x the longest eval feedback cycle) which keeps the Streamlit
    admin dashboard's "last 100" view (admin.py:21) comfortable while bounding RAM.
    """
    logger.info("Starting agent_logs retention sweep")
    try:
        deleted = await prune_agent_logs(retention_days=30)
        return {"status": "pruned", "deleted_rows": deleted}
    except Exception as exc:
        logger.error(f"agent_logs retention sweep failed: {exc}")
        raise


# ── Cron registration ───────────────────────────────────────────────────────
# arq used cron(fn, hour=2, minute=0). streaq's cron takes a crontab string
# and decorates a no-arg async function. Both daily jobs share the 2 AM slot
# to avoid two separate cold-starts. agent_logs hits Postgres; LTM hits
# Qdrant; both are idempotent and small enough to run sequentially.
# timeout=600 preserves arq's global job_timeout=600.
@worker.cron("0 2 * * *", timeout=600)
async def _run_ltm_prune():
    await prune_ltm_cron_task()


@worker.cron("30 2 * * *", timeout=600)
async def _run_agent_logs_prune():
    await prune_agent_logs_cron_task()


# ── Task registration for the eval task (defined in app/eval/tasks.py) ─────
# worker.task(fn) returns an AsyncRegisteredTask that is both callable
# (runs the function directly) and exposes .enqueue() for the queueing
# path. chat.py uses the .enqueue() form.
eval_turn_task = worker.task(_eval_turn_task_fn)
