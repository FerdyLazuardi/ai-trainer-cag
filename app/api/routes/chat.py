import asyncio
import json
import random
import re
import time
import uuid
from typing import Any, Optional
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, status
from fastapi.responses import StreamingResponse
from loguru import logger

# Moved inline imports to file-level
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from app.agents import conversation_state as _cs
from app.api.schemas import ChatRequest, ChatResponse, SourceReference
from app.api.concurrency import acquire_pipeline_slot, acquire_pipeline_slot_or_503
from app.database.postgres import AsyncSessionLocal
from app.database.models import UserLTMMemory
from app.utils.cache import get_cached_response, set_cached_response
from app.config.settings import get_settings
from app.utils.logger_batch import batch_logger
from app.api.auth import get_current_user, User
from app.api.user_utils import is_real_user
from app.database.redis_client import get_redis_client
from app.llm.cag_client import (
    OpenRouterUsage,
    extract_openrouter_usage,
    fetch_openrouter_generation_usage,
)

router = APIRouter()
settings = get_settings()

_META_CONVO_RE = re.compile(
    r"(?:udah|sudah|udh|tadi|barusan|kita|kami)\b[^.?!\n]{0,30}"
    r"(?:bahas|dibahas|ngomong|omongin|diskusi|obrol)"
    r"|(?:yang|apa)\b[^.?!\n]{0,20}(?:di)?(?:bahas|omongin|diskusi)"
    r"|itu aja[^.?!\n]{0,25}(?:bahas|omongin)"
    r"|what (?:did|have|were) we (?:discuss|talk|cover|go over|chat)"
    r"|(?:yg|yang)\s+(?:mana|tadi|yg\s+tadi|sebelumnya|sebelum|yg\s+sebelumnya)\b"
    r"|(?:yg|yang)\s+(?:mana|tadi|sebelumnya)\s*[?.!\s]*$"
    r"|(?:gimana|gmana|gmn|how)\s+(?:caranya|carany)(?:\s+(?:ya|yaa|dong|donk|sih))?\s*[?.!\s]*$"
    r"|(?:terus|trus|lanjut|next)\s+(?:gimana|gmn|apa|apanya)\b",
    re.IGNORECASE,
)


def get_cag_graph():
    from app.graph.pipeline import get_cag_graph as _get_cag_graph
    return _get_cag_graph()


def get_rag_graph():
    return get_cag_graph()


def get_cheap_llm():
    from app.llm.client import get_cheap_llm as _get_cheap_llm
    return _get_cheap_llm()


async def _complete_openrouter_usage(usage: OpenRouterUsage) -> OpenRouterUsage:
    if usage.cost or not usage.generation_id:
        return usage
    try:
        stats = await fetch_openrouter_generation_usage(usage.generation_id)
    except Exception:
        return usage
    if not stats.cost:
        return usage
    return OpenRouterUsage(
        prompt_tokens=stats.prompt_tokens or usage.prompt_tokens,
        cached_tokens=stats.cached_tokens or usage.cached_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        completion_tokens=stats.completion_tokens or usage.completion_tokens,
        provider=stats.provider or usage.provider,
        cost=stats.cost,
        generation_id=stats.generation_id or usage.generation_id,
    )


# ── Cache privacy helpers (C1) ────────────────────────────────────────────────
def compute_was_personalized(
    *,
    ltm_profile: Optional[dict],
    user_pref_dict: Optional[dict],
    recent_history: Optional[list],
    summary: Optional[str],
    user_context: Optional[dict] = None,
) -> bool:
    """True when user-specific content was injected into the LLM prompt.

    The query cache is keyed by {namespace}:{course_id|global}:{sha256(query)}
    — NOT by user_id. Any answer shaped by user-specific input must be cached
    under a user-scoped namespace (see `cache_namespace_for`) or it leaks to
    other users asking the same question.

    Personalizing inputs:
      - LTM episodes (user's past queries/preferences)
      - UserProfile prefs (custom tone/formatting)
      - User context (Moodle profile like name, dept)
      - Conversation history / summary (C1): `recent_history` and `summary` are
        injected into `messages` / `conversation_summary` and DO shape
        multi-turn answers (pronoun binding, follow-ups referencing earlier
        turns). A multi-turn answer can embed details from this user's earlier
        private turns, so it must not land in the global namespace. The earlier
        assumption that "summary is not personal" was exactly the leak.

    A first-turn-ever query (no history, summary, LTM, or prefs) returns False
    and stays globally cacheable, preserving the cold-KB cache-hit rate.
    """
def compute_was_personalized(
    *,
    ltm_profile: Optional[dict],
    recent_history: list,
    summary: str,
    user_context: Optional[dict],
) -> bool:
    """Return True if prompt contains user-specific payload."""
    _ltm = ltm_profile or {}
    has_ltm = bool((_ltm.get("learning_summary") or "").strip())
    has_history = bool(recent_history) or bool((summary or "").strip())
    ctx = user_context or {}
    has_user_ctx = bool(ctx) and any(
        bool(str(v).strip()) for v in ctx.values() if v
    )
    return has_ltm or has_history or has_user_ctx


def cache_namespace_for(*, was_personalized: bool, user_id) -> str:
    """Resolve the cache namespace. User-scoped when personalized, else global.

    Single source of truth for both the chat and chat-stream write paths so the
    privacy guarantee can't drift between them.
    """
    return f"rag_user_{user_id}" if was_personalized else "rag"

# Module-level set to keep references to background stream tasks alive
# (prevents GC from cancelling them mid-flight after the SSE generator returns).
_stream_bg_tasks: set[asyncio.Task] = set()

# ── STM facade ──────────────────────────────────────────────────────────────
# Thin wrappers over app.agents.conversation_state (Redis 8 HASH schema).
# These adapt the dependency-injected `redis`-first API to the call signatures
# used throughout this module, injecting the singleton redis client so the
# many call sites below stay unchanged. get_redis_client() is @lru_cache'd —
# each call returns the same client instance, no per-call overhead.
async def append_to_history(
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    max_turns: int = 10,
) -> int:
    return await _cs.append_to_history(
        get_redis_client(),
        conversation_id,
        user_message,
        assistant_message,
        max_turns=max_turns,
    )


async def get_conversation_history(conversation_id: str) -> list[dict]:
    return await _cs.get_history(get_redis_client(), conversation_id)


async def get_seen_chunk_ids(conversation_id: str) -> set[str]:
    return await _cs.get_seen_chunk_ids(get_redis_client(), conversation_id)


async def add_seen_chunk_ids(conversation_id: str, retrieved_context: list) -> None:
    chunk_ids = [
        str(c.get("chunk_id"))
        for c in retrieved_context or []
        if c.get("chunk_id")
    ]
    await _cs.add_seen_chunk_ids(get_redis_client(), conversation_id, chunk_ids)


async def clear_conversation_history(conversation_id: str) -> None:
    await _cs.clear_conversation(get_redis_client(), conversation_id)


async def bump_topic_streak(conversation_id: str, topic: str) -> Optional[str]:
    return await _cs.bump_topic_streak(
        get_redis_client(),
        conversation_id,
        topic,
        threshold=settings.coaching_streak_threshold,
    )


async def resolve_numeric_query(query: str, conversation_id: str) -> str:
    return await _cs.resolve_numeric_query(
        get_redis_client(), query, conversation_id
    )


# Bare affirmation / continuation tokens — "iya", "boleh", "oke lanjut", etc.
# A message made up ONLY of these has NO standalone meaning: its answer depends
# entirely on what the AI just offered. Such turns must NEVER touch the semantic
# cache (read OR write). WHY: the cache keys on the embedding of the RAW query.
# A bare "iya boleh" embeds to a near-constant vector, so (a) writing an answer
# under it POISONS the cache — every future "iya"/"boleh" then matches at
# score ~1.0 and gets served that one stale answer regardless of context; and
# (b) reading skips the graph, so the affirmation-to-offer rewrite (which lives
# in the pre-processor and resolves "iya boleh" → the offered topic using
# history) never runs. Skipping cache forces these through the graph every time.
_AFFIRMATION_TOKENS = frozenset({
    "iya", "ya", "yaa", "iyaa", "yoi", "yup", "yep", "yes", "yess",
    "boleh", "bole", "oleh", "oke", "okay", "ok", "oce", "sip", "siap",
    "mau", "lanjut", "lanjutkan", "lanjutin", "terus", "next", "gas", "gaskeun",
    "sure", "yuk", "ayo", "ayok", "dong", "deh", "aja", "nih", "kuy",
    "go", "ahead", "please", "tolong", "monggo", "silakan", "silahkan",
})


def _is_bare_affirmation(query: str) -> bool:
    """True if the message is ONLY affirmation/continuation tokens (≤4 words).

    "iya boleh" / "boleh" / "oke lanjut" → True (un-cacheable, context-dependent).
    "iya mana prinsipnya" → False (carries the substantive token "prinsipnya").
    """
    toks = re.findall(r"[a-zA-Z]+", query.lower())
    if not toks or len(toks) > 4:
        return False
    return all(t in _AFFIRMATION_TOKENS for t in toks)


async def get_or_summarize_history(
    conversation_id: str, llm, max_fresh_turns: int = settings.max_fresh_turns, *, persist: bool = True
) -> tuple[str, list[dict]]:
    return await _cs.get_or_summarize_history(
        get_redis_client(),
        conversation_id,
        llm,
        max_fresh_turns=max_fresh_turns,
        persist=persist,
    )


async def _schedule_summary_refresh(conv_id: str) -> None:
    """Fire-and-forget out-of-band STM summary refresh (C7). Enqueues a streaq
    task IFF the conversation overflowed the fresh window; deduped via NX lock.
    Wrapped so the hot path can `create_task` it without awaiting."""
    try:
        await _cs.schedule_summary_refresh(
            get_redis_client(), conv_id, max_fresh_turns=settings.max_fresh_turns
        )
    except Exception:
        pass


async def _schedule_afk_ltm_sync(conv_id: str, u_id: str):
    """Delegate to the HASH-based scheduler. last_active + scheduled are now
    HASH fields on `rag:conv:{id}` (written via HSETEX), and the worker's AFK
    guard reads the SAME fields — closing the split-key bug where chat.py wrote
    `rag:last_active:{id}`/`rag:ltm:scheduled:{id}` STRING keys the worker never
    read. schedule_afk_sync handles set_last_active + dedup + enqueue + set_scheduled."""
    await _cs.schedule_afk_sync(get_redis_client(), conv_id, u_id)


async def _track_session_courses(conv_id: str, retrieved_context: list) -> None:
    """Capture distinct course_names from retrieved_context into the conversation
    HASH `courses` field (via conversation_state.add_courses).

    LTM sync worker reads this via get_courses() instead of asking the LLM to
    extract course names — the LLM hallucinates names that don't exist in the KB
    ("Amartha products"), but `chunk.metadata.course_name` is ground truth from
    the Moodle ingestion. Stored as a HASH field with the LTM scheduling-window
    TTL so it's still readable when the AFK worker fires.
    """
    if not retrieved_context:
        return
    names: set[str] = set()
    for c in retrieved_context:
        name = (c.get("course_name") or "").strip()
        if name and name not in ("?", "Unknown"):
            names.add(name)
    if not names:
        return
    try:
        await _cs.add_courses(get_redis_client(), conv_id, names)
    except Exception as e:
        logger.warning(f"Failed to track session courses: {e}")

DEV_BYPASS_USER_ID = "dev_user_123"


# Intents whose answers don't go through the CAG generator — no faithfulness
# signal worth measuring. GREETING/AMBIGUOUS are canned, MALICIOUS is a refusal,
# TOPIC_LIST reads from Postgres metadata, COACHING often returns a Socratic
# guiding question (not a standard answer to grade for faithfulness),
# OFF_SCOPE is a canned redirect with no retrieval.
_EVAL_SKIP_INTENTS = {"GREETING", "AMBIGUOUS", "MALICIOUS", "TOPIC_LIST", "COACHING", "OFF_SCOPE"}


def _should_eval_turn(
    *,
    intent: str | None,
    intent_scores: dict | None,
    max_dense_score: float | None,
    answer: str | None,
    is_low_relevance: bool = False,
) -> bool:
    """Sampling decision for post-hoc evaluation.

    Evaluates when ANY of:
    - Random draw under `eval_sample_rate` (baseline drift signal).
    - Empathy axis ≥ threshold (vent / resign — high-stakes turn).
    - Top dense cosine below threshold (suspected retrieval miss / potential halu).

    Skips canned-response intents and empty answers regardless.

    H6: Skips low-relevance turns entirely. When both gate signals miss, the
    generator emits a canned NOT-FOUND refusal that makes no factual claim, so
    the judge auto-scores it 1.0 (faithful-by-vacuity) — grading these inflates
    the mean faithfulness and hides real regressions. The `dense<below` force
    branch is therefore meaningful only for sparse-rescued turns (low dense but
    a real, grounded answer was produced), which is exactly what remains once
    low-relevance is filtered out here.
    """
    if not settings.eval_enabled:
        return False
    if not answer or not answer.strip():
        return False
    if intent in _EVAL_SKIP_INTENTS:
        return False
    if is_low_relevance:
        return False

    scores = intent_scores or {}
    empathy = float(scores.get("needs_empathy") or 0.0)
    if empathy >= settings.eval_always_if_empathy_above:
        return True

    if (
        max_dense_score is not None
        and max_dense_score < settings.eval_always_if_dense_below
    ):
        return True

    return random.random() < settings.eval_sample_rate


async def _enqueue_eval(
    *,
    turn_id: Optional[str],
    query: str,
    answer: str,
    retrieved_context: list,
    intent: Optional[str],
    intent_scores: Optional[dict],
) -> None:
    """Push the turn onto the streaq queue for async LLM-as-judge evaluation.

    Fire-and-forget — failures are logged and swallowed so eval scheduling
    never affects user-facing flow.
    """
    # turn_id correlates the async faithfulness score back to the agent_logs row.
    if not turn_id:
        return
    try:
        from app.worker import eval_turn_task

        await eval_turn_task.enqueue(
            query=query,
            answer=answer,
            retrieved_context=retrieved_context or [],
            intent=intent,
            intent_scores=intent_scores or {},
            turn_id=turn_id,
        ).start(priority="high")
    except Exception as e:
        logger.warning(f"Failed to enqueue eval task: {e}")


def _quality_log_fields(
    intent: Optional[str],
    intent_scores: Optional[dict],
    max_dense_score: Optional[float],
    gate_score: Optional[dict] = None,
) -> dict:
    """Build the durable quality-signal columns for an agent_logs row.

    Persists intent + retrieval signal + semantic-gate trace to Postgres so
    monitoring works without any external tracing backend. ``gate_score`` is
    the dataclass-as-dict from ``classify_semantic_with_scores`` (or None
    for cache hits / non-chat paths that skip the pre-processor).
    """
    scores = intent_scores or {}

    def _f(key):
        v = scores.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    def _gf(key, caster=float):
        if not gate_score:
            return None
        v = gate_score.get(key)
        return caster(v) if v is not None else None

    return {
        "intent": intent,
        "needs_lookup": _f("needs_lookup"),
        "needs_reasoning": _f("needs_reasoning"),
        "needs_empathy": _f("needs_empathy"),
        "max_dense_score": float(max_dense_score) if isinstance(max_dense_score, (int, float)) else None,
        # Semantic-gate trace (HIT/MISS/SKIP + the raw cosine numbers that
        # drove the decision). The dashboard reads these directly to plot
        # per-intent distributions and detect drift.
        "gate_decision": _gf("decision", str),
        "gate_intent": _gf("best_intent", str),
        "gate_best_cosine": _gf("best_cosine"),
        "gate_second_cosine": _gf("second_cosine"),
        "gate_margin": _gf("margin"),
    }


def _serialize_gate_score(gs) -> Optional[dict]:
    """Convert a GateScore dataclass (or None / dict) to a plain dict for
    batch_logger pickling. Returns None when the gate didn't run (cache
    hit, or pre-processor skipped it)."""
    if gs is None:
        return None
    if isinstance(gs, dict):
        return gs
    return {
        "decision": getattr(gs, "decision", None),
        "committed": getattr(gs, "committed", None),
        "best_intent": getattr(gs, "best_intent", None),
        "best_cosine": getattr(gs, "best_cosine", None),
        "second_intent": getattr(gs, "second_intent", None),
        "second_cosine": getattr(gs, "second_cosine", None),
        "margin": getattr(gs, "margin", None),
    }


async def _verify_conversation_ownership(conversation_id: str, current_user: User):
    """Ensure the user owns this conversation before accessing history.

    B1: ownership is the `owner` field of the conversation HASH `rag:conv:{id}`
    itself — no separate STRING, written with NO per-field TTL. Owner and
    history therefore share one Redis key and are evicted TOGETHER under
    volatile-lru (eviction is atomic per key, never field-by-field). This closes
    the prior fail-open window where an LRU-evicted `rag:conv_owner:{id}` STRING
    left a surviving history HASH re-claimable by a different user.

    Migration (dual-read): conversations created before B1 still carry their
    owner in the legacy `rag:conv_owner:{id}` STRING with no `owner` HASH field.
    When the field is absent we fall back to the legacy STRING; if present we
    adopt it into the HASH and honor it, so an active pre-B1 conversation can't
    be re-claimed during cutover. The legacy fallback (and the old keys) can be
    dropped one full `conversation_ttl_seconds` cycle after deploy.

    Dev-bypass migration: a conversation owned by the dev bypass user is handed
    over to the first real authenticated user seamlessly.
    """
    redis = get_redis_client()
    conv_key = _cs._conv_key(conversation_id)
    ttl = settings.conversation_ttl_seconds

    stored_owner = await redis.hget(conv_key, "owner")

    # ── Migration dual-read: fall back to the legacy STRING if no HASH field ──
    if not stored_owner:
        legacy_owner = await redis.get(f"rag:conv_owner:{conversation_id}")
        if legacy_owner:
            # Adopt the legacy claim into the HASH (idempotent under a race —
            # concurrent adopters write the same value). key-level EXPIRE keeps
            # the HASH evictable/bounded.
            async with redis.pipeline(transaction=True) as pipe:
                pipe.hset(conv_key, "owner", legacy_owner)
                pipe.expire(conv_key, ttl)
                await pipe.execute()
            stored_owner = legacy_owner

    logger.info(
        "Checking ownership",
        conversation_id=conversation_id,
        current_user_id=current_user.user_id,
        stored_owner=stored_owner,
    )

    if stored_owner:
        # Allow real user to reclaim a conversation previously owned by dev bypass
        if stored_owner == DEV_BYPASS_USER_ID and current_user.user_id != DEV_BYPASS_USER_ID:
            logger.info(
                "Migrating conversation ownership from dev_user to real user",
                conversation_id=conversation_id,
                new_owner=current_user.user_id,
            )
            await redis.hset(conv_key, "owner", current_user.user_id)
        elif stored_owner != current_user.user_id:
            logger.error(
                "Ownership mismatch 403",
                conversation_id=conversation_id,
                current_user_id=current_user.user_id,
                stored_owner=stored_owner,
            )
            raise HTTPException(status_code=403, detail="Not authorized to access this conversation")
    else:
        # No owner anywhere — claim atomically. HSETNX can't double-claim a new
        # conversation_id under concurrent first-turn requests; EXPIRE in the
        # same pipeline guarantees a key-level TTL so a claim-then-crash can't
        # leave an unevictable, unbounded HASH.
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hsetnx(conv_key, "owner", current_user.user_id)
            pipe.expire(conv_key, ttl)
            results = await pipe.execute()
        if not results[0]:
            # Another request claimed it between our HGET and HSETNX — re-read to verify
            actual_owner = await redis.hget(conv_key, "owner")
            if actual_owner and actual_owner != current_user.user_id:
                logger.error(
                    "Ownership race lost — another user claimed this conversation_id",
                    conversation_id=conversation_id,
                    current_user_id=current_user.user_id,
                    actual_owner=actual_owner,
                )
                raise HTTPException(status_code=403, detail="Not authorized to access this conversation")
        logger.info(
            "Claiming conversation ownership",
            conversation_id=conversation_id,
            new_owner=current_user.user_id,
        )
async def _prepare_cag_context(
    request: ChatRequest,
    current_user: User,
    conversation_id: str,
    resolved_query: str,
) -> dict:
    """Shared context preparation for both /chat and /chat/stream.

    Builds the graph input and checks exact-match cache before the CAG model
    runs. The query cache is Redis exact-match only and does NOT use embeddings.
    """
    from app.graph.intent_rules import classify as _tier1_classify
    import time as _time

    _t0 = _time.perf_counter()

    # Tier-1 pre-check: skip embedding entirely for greetings/fillers.
    _tier1_intent = _tier1_classify(resolved_query)
    _skip_embedding = _tier1_intent in ("GREETING", "AMBIGUOUS")

    # NOTE: embed+rewrite moved BELOW history fetch + cache check so cache hits
    # skip both (~1s+ savings on cache hits, and embed/rewrite overlap on miss).
    query_embedding = None
    # synthesis/empathy. Cache stores KNOWLEDGE-shaped answers — feeding them
    # to opinion/vent queries causes shape mismatch (e.g. user asks "menurut
    # kamu mana paling kritis" but cache returns a flat list).
    # Cheap regex check; full intent classification still happens in graph.
    #
    # TOPIC_LIST is skipped too: it's answered from the live Postgres course
    # list and is never written to the cache (excluded from the write gate), so
    # a lookup would always miss — skipping it just saves the round-trip and
    # keeps the answer bound to live Postgres data rather than a stale snapshot.
    import re
    _OPINION_REGEX = re.compile(
        r"\b(menurut|menurutmu|pendapat|opini|kasih saran|sarankan|advice|"
        r"what (?:do you|would you) think|"
        r"capek?|cape lah|lelah|males|stress|bingung|pusing|frustrasi|nyerah|curhat|"
        r"gimana kalau|kalau aku|what if|bantuin mikir|help me think|"
        r"mana yang|mana yg|paling penting|paling kritis|paling baik|"
        r"role[\s-]?play|anggap kamu)\b",
        re.IGNORECASE,
    )
    skip_cache = (
        _skip_embedding
        or _tier1_intent == "TOPIC_LIST"
        or request.coaching_mode
        or _is_bare_affirmation(resolved_query)
        or bool(_OPINION_REGEX.search(resolved_query))
        or bool(_META_CONVO_RE.search(resolved_query))
    )
    if skip_cache:
        logger.debug(
            "Cache lookup skipped — greeting/filler, coaching mode, "
            "opinion/synthesis, or meta-conversation recall pattern",
            query=resolved_query[:60],
        )

    # Build message history + LTM + UserProfile in parallel.
    # These three I/O calls are independent (Redis, Postgres) — fan
    # them out instead of awaiting serially. Only LTM and UserProfile depend
    # on `ltm_eligible`, so they're guarded with stub coroutines that early-
    # return for non-real users (dev bypass, anonymous, etc.).
    #
    # Tradeoff: on a brand-new session of a real user we'll fetch LTM +
    # UserProfile that get discarded (since `is_brand_new_session` is only
    # known AFTER history resolves). That's 1 LTM lookup + 1 Postgres
    # SELECT on first-turn-ever — acceptable to save 100-300ms on every
    # subsequent regular turn.
    user_id = current_user.user_id
    ltm_eligible = is_real_user(user_id=user_id, role=current_user.role)

    # Query KPI and Branch data from the database
    kpi_data = None
    branch_data = None
    try:
        async with AsyncSessionLocal() as db_session:
            from sqlalchemy import select
            from app.database.models import UserKPIData, BranchData

            # Fetch User KPI
            kpi_stmt = select(UserKPIData).where(UserKPIData.username == current_user.username)
            kpi_data = (await db_session.execute(kpi_stmt)).scalars().first()

            # Fetch Branch Data (with normalized fallback match)
            point_val = str(current_user.point or "").strip()
            if point_val:
                branch_stmt = select(BranchData).where(BranchData.point == point_val)
                branch_data = (await db_session.execute(branch_stmt)).scalars().first()
                if not branch_data:
                    norm_point = point_val.lower().replace(" ", "")
                    all_branches_stmt = select(BranchData)
                    all_branches = (await db_session.execute(all_branches_stmt)).scalars().all()
                    for b in all_branches:
                        if (b.point or "").lower().replace(" ", "") == norm_point:
                            branch_data = b
                            break
    except Exception as exc:
        logger.warning(f"Failed to load spreadsheet data from database: {exc}")

    # Live Moodle profile (firstname + custom fields) from the JWT. Lets the
    # generate node greet by name and tailor answers to the user's dept/role.
    # Per-user, so it never goes in the shared cache key (cache_namespace_for
    # already splits on user_id).
    user_context = {
        "name": current_user.username,
        "dept": current_user.dept,
        "location": current_user.location,
        "position": current_user.position,
        "grade": current_user.grade,
        "point": current_user.point,
        "gender": current_user.gender,
        "area": current_user.area,
        "regional": current_user.regional,
    }

    if kpi_data and kpi_data.full_name:
        user_context["name"] = kpi_data.full_name

    # Keys to exclude from raw dynamic injection because they duplicate standard profile fields or internal metadata
    REDUNDANT_KEYS = {
        "full_name", "Jabatan", "Role", "nama_cabang", "username", 
        "user_id", "nik", "user_name", "updated_at", "periode", "periode_kpi"
    }

    # Special handling for Management Trainee role adaptation from spreadsheet
    pos_upper = str(user_context.get("position") or "").strip().upper()
    grade_upper = str(user_context.get("grade") or "").strip().upper()
    is_mt = "MANAGEMENT TRAINEE" in pos_upper or pos_upper == "MT" or "MT" in grade_upper

    if kpi_data and isinstance(kpi_data.data, dict):
        sp_role = str(kpi_data.data.get("Role") or kpi_data.data.get("Jabatan") or "").strip()
        if is_mt and sp_role:
            user_context["position"] = f"Management Trainee ({sp_role.upper()})"
            user_context["role"] = sp_role.upper()

        for k, v in kpi_data.data.items():
            if k not in user_context and k not in REDUNDANT_KEYS and v is not None:
                user_context[k] = v

    # Inject branch-specific spreadsheet data dynamically (role-tailored & metadata filtered)
    if branch_data and isinstance(branch_data.data, dict):
        user_role = str(user_context.get("role") or "").strip().upper()
        if not user_role and user_context.get("position"):
            pos_str = str(user_context["position"]).upper()
            if "BM" in pos_str or "MANAGER" in pos_str:
                user_role = "BM"
            elif "BP" in pos_str or "PARTNER" in pos_str:
                user_role = "BP"

        for k, v in branch_data.data.items():
            k_clean = str(k).strip()
            if k_clean in REDUNDANT_KEYS or v is None:
                continue
            if k_clean.lower().endswith("updated_at") or k_clean.lower().endswith("updated at"):
                continue

            # Tailor role-specific branch aggregates (e.g. "Point BP - ..." vs "Point BM - ...")
            if user_role and ("Point BP - " in k_clean or "Point BM - " in k_clean):
                opposite_role = "BM" if user_role == "BP" else "BP"
                if f"Point {opposite_role} - " in k_clean:
                    continue  # Skip opposite role's branch aggregate

            if k_clean not in user_context:
                user_context[k_clean] = v


    # For Tier-1 intents (GREETING, AMBIGUOUS), skip all expensive I/O:
    # no history summarization (avoids LLM call), no LTM, no UserProfile.
    # These intents have hardcoded responses that don't use any of this context.
    if _skip_embedding:
        return {
            "cached": None,
            "initial_state": {
                "messages": [HumanMessage(content=resolved_query)],
                "conversation_id": conversation_id,
                "conversation_summary": "",
                "user_profile": {"summary": "", "course_names": []},
                "user_preferences": None,
                "user_context": user_context,
            },
            "query_embedding": None,
            "was_personalized": False,
            "skip_cache": skip_cache,
        }

    # History fetch runs FIRST so cache hits can short-circuit before embed.
    # On miss it also gives the rewrite input (langchain messages) we fan out
    # alongside embed+LTM+profile below — rewrite no longer blocks the graph's
    # critical path (it ran sequentially there before).
    logger.debug(f"[TIMING] pre-history: {_time.perf_counter()-_t0:.2f}s")
    _t_hist = _time.perf_counter()
    summary, recent_history = await get_or_summarize_history(
        conversation_id=conversation_id,
        llm=get_cheap_llm(),
        max_fresh_turns=settings.max_fresh_turns,
        persist=False,  # C7: no LLM, no write on the hot path
    )
    # C7: kick the out-of-band summary refresh (fire-and-forget). Only enqueues
    # if the conversation overflowed the fresh window; NX-deduped.
    asyncio.create_task(_schedule_summary_refresh(conversation_id))

    seen_chunk_ids = await get_seen_chunk_ids(conversation_id)
    logger.debug(f"[TIMING] history: {_time.perf_counter()-_t_hist:.2f}s")
    if (recent_history or summary) and len(resolved_query.split()) <= 3:
        skip_cache = True
        logger.debug("Cache lookup skipped - short follow-up query (context-dependent)")

    cached = None
    if not skip_cache:
        _t_cache_start = _time.perf_counter()

        private_ns = f"rag_user_{current_user.user_id}"
        global_ns = "rag"

        private_cached, global_cached = await asyncio.gather(
            get_cached_response(
                resolved_query,
                course_id=request.course_id,
                cache_namespace=private_ns,
            ),
            get_cached_response(
                resolved_query,
                course_id=request.course_id,
                cache_namespace=global_ns,
            )
        )
        cached = private_cached or global_cached
        logger.debug(f"[TIMING] get_cached_response (private+global): {_time.perf_counter()-_t_cache_start:.2f}s")
    if cached:
        return {
            "cached": cached,
            "query_embedding": None,
            "initial_state": {},
            "was_personalized": False,
            "skip_cache": skip_cache,
        }

    # Build langchain messages for the CAG graph. Retrieval-era rewrite and
    # embedding are deliberately skipped: generate_node receives the full KB.
    messages: list[BaseMessage] = []
    for turn in recent_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=resolved_query))
    is_brand_new_session = not recent_history and not summary

    _retrieval_query = resolved_query
    _rewritten_queries = None
    query_embedding = None
    ltm_profile = {"learning_summary": ""}

    user_profile_obj = None
    if ltm_eligible:
        async with AsyncSessionLocal() as session:
            user_profile_obj = await session.get(UserLTMMemory, user_id)

    if ltm_eligible and user_profile_obj is not None:
        ltm_profile = {
            "learning_summary": user_profile_obj.learning_summary or "",
        }

    initial_state = {
        "messages": messages,
        "conversation_id": conversation_id,
        "conversation_summary": summary,
        "user_profile": ltm_profile,
        "user_preferences": None,
        "user_context": user_context,
        # Legacy embedding fields are intentionally empty in CAG mode.
        "query_embedding": query_embedding,
        "query_embedding_text": _retrieval_query if query_embedding is not None else None,
        # Pre-computed rewrite (parallel with embed above) so _pre_processor
        # skips its own rewrite LLM call. None → pre_processor falls back to its
        # own rewrite (graph invoked without chat.py pre-compute, e.g. tests).
        "rewritten_queries": _rewritten_queries,
        "retrieval_query": _retrieval_query,
        "seen_chunk_ids": list(seen_chunk_ids),
        # Socratic coaching toggle (opt-in via UI). When True, _pre_processor
        # promotes a real question to COACHING → SOCRATIC_PROMPT.
        "coaching_mode": request.coaching_mode,
    }

    # ── Cross-user cache-leak guard (FIX 1 + C1) ──────────────────────────────
    # Privacy decision extracted to `compute_was_personalized` (pure + unit
    # tested). When any user-specific content (LTM, prefs, OR conversation
    # history/summary) shaped the answer, it must be cached user-scoped so it
    # can't leak to another user asking the same question.
    was_personalized = compute_was_personalized(
        ltm_profile=ltm_profile,
        recent_history=recent_history,
        summary=summary,
        user_context=user_context,
    )

    return {
        "cached": None,
        "initial_state": initial_state,
        "query_embedding": query_embedding,
        "was_personalized": was_personalized,
        "skip_cache": skip_cache,
    }


_prepare_rag_context = _prepare_cag_context

def _extract_sources(retrieved_context: list) -> list:
    sources = []
    if retrieved_context:
        for c in retrieved_context:
            if c.get("source") and c.get("source") != "Unknown":
                sources.append({
                    "chunk_id": c.get("chunk_id") or str(uuid.uuid4()),
                    "document_id": c.get("document_id") or "Unknown",
                    "source": c.get("source"),
                    "title": c.get("course_name") or c.get("title") or "Unknown",
                    "chunk_index": c.get("chunk_index") or 0,
                    "score": c.get("score") or 0.0,
                })
    return sources

def _auto_detect_course_id(retrieved_context: list, request_course_id: Optional[int]) -> Optional[int]:
    effective_course_id = request_course_id
    if effective_course_id in (None, 0) and retrieved_context:
        cids = [c.get("course_id") for c in retrieved_context if c.get("course_id") not in (None, "", 0)]
        if cids:
            try:
                effective_course_id = int(Counter(cids).most_common(1)[0][0])
            except (ValueError, TypeError):
                pass
    return effective_course_id


@router.post("/chat", response_model=ChatResponse, summary="Ask a question using the CAG pipeline")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    return await _run_chat(request, background_tasks, current_user)


@router.get("/chat/ban-status", summary="Get current chat ban status")
async def chat_ban_status(
    current_user: User = Depends(get_current_user),
) -> dict:
    return {"ban_remaining_seconds": await _get_ban_ttl(current_user.user_id)}


async def _get_ban_ttl(user_id: str) -> int:
    """Return remaining ban seconds for user. 0 means not banned."""
    try:
        redis_client = get_redis_client()
        ttl = await redis_client.ttl(f"ava:user:{user_id}:banned")
        # ttl == -2: key doesn't exist (not banned)
        # ttl == -1: key exists but no expiry (treat as not banned, shouldn't happen)
        return max(ttl, 0) if ttl > 0 else 0
    except Exception as exc:
        logger.warning(f"Failed to check user ban TTL: {exc}")
        return 0


async def _handle_off_scope_violation(user_id: str) -> tuple[int, bool]:
    """Increment violation count for the user. Returns (new_count, just_banned)."""
    try:
        redis_client = get_redis_client()
        from app.config.settings import get_settings
        settings = get_settings()
        key = f"ava:user:{user_id}:off_scope_violations"
        new_count = await redis_client.incr(key)
        await redis_client.expire(key, settings.off_scope_violation_window_seconds)

        just_banned = False
        if new_count >= settings.max_off_scope_violations:
            await redis_client.setex(
                f"ava:user:{user_id}:banned",
                settings.off_scope_ban_duration_seconds,
                "1"
            )
            just_banned = True
        return new_count, just_banned
    except Exception as exc:
        logger.warning(f"Failed to handle off-scope violation: {exc}")
        return 0, False


async def _run_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User,
) -> ChatResponse:
    start_time = time.perf_counter()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    await _verify_conversation_ownership(conversation_id, current_user)

    ban_ttl = await _get_ban_ttl(current_user.user_id)
    if ban_ttl > 0:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return ChatResponse(
            answer="Maaf, Ai Trainer dinonaktifkan sementara karena terlalu banyak menanyakan hal di luar topik.",
            sources=[],
            conversation_id=conversation_id,
            resolved_query=None,
            cached=False,
            latency_ms=round(latency_ms, 2),
            ban_remaining_seconds=ban_ttl,
        )

    resolved_query = await resolve_numeric_query(request.query, conversation_id)
    from app.graph.pipeline import _apply_glossary
    resolved_query = _apply_glossary(resolved_query)
    # Cache key MUST hash the SAME string the READ path hashes. READ (chat.py
    # _prepare_cag_context) passes `resolved_query` to get_cached_response, and
    # _query_hash normalizes via .strip().lower(). resolved_query is glossary +
    # numeric-resolved — deterministic, NO LLM — so it's safe to key on. Hashing
    # raw request.query here wrote to a key the read never produced → 0 hits
    # whenever glossary/numeric mutated the query.
    _raw_query_for_cache = resolved_query
    logger.info("Chat request received", query=request.query[:80], resolved_query=resolved_query[:80] if resolved_query != request.query else None, conversation_id=conversation_id)

    context = await _prepare_cag_context(request, current_user, conversation_id, resolved_query)
    cached = context.get("cached")

    if cached:
        latency_ms = (time.perf_counter() - start_time) * 1000

        background_tasks.add_task(
            batch_logger.add_log,
            {
                "conversation_id": conversation_id,
                "query": request.query,
                "rewritten_query": resolved_query,
                "answer": cached["answer"],
                "chunks_retrieved": 0,
                "latency_ms": round(latency_ms, 2),
                "cache_hit": True,
            }
        )
        await append_to_history(conversation_id=conversation_id, user_message=request.query, assistant_message=cached["answer"])

        # FIX 2: a cache hit still counts as session activity. Refresh
        # last_active + (re)schedule the AFK LTM sync so an all-cache-hit
        # session still persists LTM and the AFK guard doesn't misfire.
        # `_schedule_afk_ltm_sync` updates `rag:last_active` internally, so
        # this covers both the last_active refresh and the job enqueue. We do
        # NOT re-inject memory — the cached answer is already final.
        background_tasks.add_task(_schedule_afk_ltm_sync, conversation_id, current_user.user_id)

        return ChatResponse(
            answer=cached["answer"],
            sources=[SourceReference(**s) for s in cached["sources"]],
            conversation_id=conversation_id,
            resolved_query=resolved_query if resolved_query != request.query else None,
            cached=True,
            latency_ms=round(latency_ms, 2),
        )

    # Cache miss: acquire concurrency slot
    _sem = await acquire_pipeline_slot()
    try:
        return await _execute_chat_flow(
            request=request,
            background_tasks=background_tasks,
            current_user=current_user,
            context=context,
            resolved_query=resolved_query,
            start_time=start_time,
            conversation_id=conversation_id,
        )
    finally:
        _sem.release()


async def _process_off_scope_status(
    off_scope_detected: bool,
    user_id: str,
    settings: Any,
    answer: str,
) -> tuple[str, int]:
    off_scope_ban_ttl: int = 0
    if off_scope_detected:
        new_count, _ = await _handle_off_scope_violation(user_id)
        if new_count == settings.warning_off_scope_violations:
            warning_text = "\n\n**Peringatan**: Harap tanyakan hal seputar materi Amartha. Jika Anda terus bertanya di luar topik sekali lagi, Ai Trainer akan dinonaktifkan sementara."
            answer += warning_text
        elif new_count >= settings.max_off_scope_violations:
            off_scope_ban_ttl = await _get_ban_ttl(user_id)
    return answer, off_scope_ban_ttl


async def _extract_message_token_usage(final_message: Any) -> dict[str, Any]:
    usage_data = {
        "llm_tokens_used": 0,
        "or_prompt_tokens": 0,
        "or_completion_tokens": 0,
        "or_cached_tokens": 0,
        "or_provider": None,
        "or_cost": 0.0,
        "or_generation_id": None,
    }
    if hasattr(final_message, "response_metadata"):
        rm = final_message.response_metadata or {}
        tu = rm.get("token_usage", {})
        usage = await _complete_openrouter_usage(extract_openrouter_usage(final_message))
        usage_data["llm_tokens_used"] = tu.get("total_tokens", usage.prompt_tokens + usage.completion_tokens)
        usage_data["or_prompt_tokens"] = usage.prompt_tokens
        usage_data["or_completion_tokens"] = usage.completion_tokens
        usage_data["or_cached_tokens"] = usage.cached_tokens
        usage_data["or_provider"] = usage.provider
        usage_data["or_cost"] = usage.cost
        usage_data["or_generation_id"] = usage.generation_id
    return usage_data


async def _execute_chat_flow(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User,
    context: dict,
    resolved_query: str,
    start_time: float,
    conversation_id: str,
) -> ChatResponse:
    from app.config.settings import get_settings
    settings = get_settings()

    initial_state = context["initial_state"]
    was_personalized = context.get("was_personalized", False)
    cag_graph = get_cag_graph()

    result = None
    answer = None
    latency_ms = 0.0

    try:
        result = await asyncio.wait_for(
            cag_graph.ainvoke(
                initial_state,
                config={"run_name": "ava-chat"},
            ),
            timeout=settings.pipeline_total_timeout_s,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        final_message = result["messages"][-1]
        from app.graph.pipeline import _sanitize_answer
        raw_answer = final_message.content if hasattr(final_message, "content") else str(final_message)
        sanitized = _sanitize_answer(raw_answer)

        off_scope_detected = result.get("off_scope_detected", False)
        answer, off_scope_ban_ttl = await _process_off_scope_status(
            off_scope_detected, current_user.user_id, settings, sanitized
        )

        usage_data = await _extract_message_token_usage(final_message)
        llm_tokens_used = usage_data["llm_tokens_used"]
        or_prompt_tokens = usage_data["or_prompt_tokens"]
        or_completion_tokens = usage_data["or_completion_tokens"]
        or_cached_tokens = usage_data["or_cached_tokens"]
        or_provider = usage_data["or_provider"]
        or_cost = usage_data["or_cost"]
        or_generation_id = usage_data["or_generation_id"]


    except asyncio.TimeoutError as exc:
        # An upstream black-hole pinned this slot to the wall-clock ceiling.
        # Fail loud with 504 (NOT 500) so the client knows it's a timeout and
        # the slot is freed by the outer `finally` instead of held for minutes.
        logger.error(
            "Chat pipeline timed out",
            timeout_s=settings.pipeline_total_timeout_s,
            query=request.query[:60],
            conversation_id=conversation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The assistant took too long to respond. Please try again.",
        ) from exc
    except Exception as exc:
        logger.error("CAG pipeline error", error=str(exc), query=request.query[:60])
        raise HTTPException(status_code=500, detail="CAG pipeline failed") from exc

    rewritten_query = result.get("rewritten_query") or resolved_query
    resolved_query = rewritten_query
    intent = result.get("intent", "KNOWLEDGE")
    # Correlation key tying this agent_logs row to its async faithfulness score.
    turn_id = str(uuid.uuid4())
    
    actual_chunks = 0
    max_chunk_score = None
    retrieved_context = result.get("retrieved_context") or []
    
    if retrieved_context:
        real_chunks = [c for c in retrieved_context if c.get("source") not in (None, "", "None")]
        actual_chunks = len(real_chunks)
        dense_scores = [c.get("dense_score") for c in retrieved_context if isinstance(c.get("dense_score"), (int, float))]
        if dense_scores:
            max_chunk_score = max(dense_scores)

    sources = _extract_sources(retrieved_context)

    is_low_relevance = False
    
    if (
        intent not in ("GREETING", "AMBIGUOUS", "MALICIOUS", "TOPIC_LIST", "COACHING", "OFF_SCOPE")
        and not is_low_relevance
        and not _is_bare_affirmation(request.query)
        and not context.get("skip_cache", False)
    ):
        ns = cache_namespace_for(was_personalized=was_personalized, user_id=current_user.user_id)
        background_tasks.add_task(
            set_cached_response,
            query=resolved_query,
            answer=answer,
            sources=sources,
            course_id=request.course_id,
            cache_namespace=ns,
        )
    # ponytail: persist history synchronously before the response returns,
    # not as a BackgroundTask. BackgroundTasks run AFTER the response is sent,
    # so the next turn's rewrite would read Redis before this commits → stale
    # history → wrong pronoun/ordinal resolution (same bug as the stream path).
    try:
        await append_to_history(conversation_id=conversation_id, user_message=request.query, assistant_message=answer)
    except Exception as hist_err:
        logger.warning(f"append_to_history (non-stream) failed: {hist_err}")
    try:
        await add_seen_chunk_ids(conversation_id, retrieved_context)
    except Exception as seen_err:
        logger.debug(f"seen chunk tracking skipped: {seen_err}")
    background_tasks.add_task(
        batch_logger.add_log,
        {
            "turn_id": turn_id,
            "endpoint": "chat",
            "conversation_id": conversation_id,
            "query": request.query,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "chunks_retrieved": actual_chunks,
            "latency_ms": round(latency_ms, 2),
            "llm_tokens_used": llm_tokens_used,
            "or_prompt_tokens": or_prompt_tokens,
            "or_cached_tokens": or_cached_tokens,
            "or_completion_tokens": or_completion_tokens,
            "or_provider": or_provider,
            "or_generation_id": or_generation_id,
            "or_cost": or_cost,
            "cache_hit": False,
            "retrieved_context": retrieved_context,
            **_quality_log_fields(
                intent,
                result.get("intent_scores"),
                max_chunk_score,
                _serialize_gate_score(result.get("gate_score")),
            ),
        }
    )
    
    logger.info(
        "Chat response sent",
        query=request.query[:60],
        latency_ms=round(latency_ms, 2),
        chunks_retrieved=actual_chunks,
        max_chunk_score=max_chunk_score,
    )
    
    background_tasks.add_task(_schedule_afk_ltm_sync, conversation_id, current_user.user_id)
    background_tasks.add_task(_track_session_courses, conversation_id, retrieved_context)

    if _should_eval_turn(
        intent=intent,
        intent_scores=result.get("intent_scores"),
        max_dense_score=max_chunk_score,
        answer=answer,
        is_low_relevance=is_low_relevance,
    ):
        background_tasks.add_task(
            _enqueue_eval,
            turn_id=turn_id,
            query=resolved_query,
            answer=answer,
            retrieved_context=retrieved_context,
            intent=intent,
            intent_scores=result.get("intent_scores"),
        )

    return ChatResponse(
        answer=answer,
        sources=[SourceReference(**s) for s in sources],
        conversation_id=conversation_id,
        resolved_query=resolved_query if resolved_query != request.query else None,
        cached=False,
        latency_ms=round(latency_ms, 2),
        ban_remaining_seconds=off_scope_ban_ttl if off_scope_ban_ttl > 0 else None,
    )


@router.get("/chat/history/{conversation_id}", summary="Get chat history for a session")
async def get_history(
    conversation_id: str,
    current_user: Optional[User] = Depends(get_current_user),
) -> list[dict]:
    if current_user:
        await _verify_conversation_ownership(conversation_id, current_user)
    return await get_conversation_history(conversation_id)


@router.delete("/chat/history/{conversation_id}", summary="Clear chat history for a session")
async def delete_history(
    conversation_id: str,
    current_user: Optional[User] = Depends(get_current_user),
):
    if current_user:
        await _verify_conversation_ownership(conversation_id, current_user)
    await clear_conversation_history(conversation_id)
    return {"status": "success", "message": "Conversation history cleared"}


@router.get("/user/onboarding", summary="Has the current user seen the onboarding tour?")
async def get_onboarding_status(
    current_user: User = Depends(get_current_user),
) -> dict:
    """Return {completed: bool} for the authenticated user.

    Drives the first-run tour: the frontend only auto-starts the tour when
    completed=false. Stored in user_profiles.onboarding_completed_at so the
    "seen it" state follows the user across browsers/devices (localStorage is
    per-device). The dev-bypass user is treated as never-completed so local
    iteration always sees the tour.
    """
    if not is_real_user(current_user.user_id, current_user.role):
        return {"completed": False}
    async with AsyncSessionLocal() as session:
        profile = await session.get(UserLTMMemory, current_user.user_id)
    completed = bool(profile and profile.onboarding_completed_at is not None)
    return {"completed": completed}


@router.post("/user/onboarding/complete", summary="Mark the onboarding tour as seen")
async def complete_onboarding(
    current_user: User = Depends(get_current_user),
) -> dict:
    """Stamp user_ltm_memories.onboarding_completed_at = now() for this user.

    Idempotent: re-calling just refreshes the timestamp. Creates the
    UserLTMMemory row if it doesn't exist yet (first-ever interaction). No-op for
    the dev-bypass user so we don't pollute the table with a synthetic row.
    """
    if not is_real_user(current_user.user_id, current_user.role):
        return {"status": "skipped"}
    from datetime import datetime, timezone
    async with AsyncSessionLocal() as session:
        profile = await session.get(UserLTMMemory, current_user.user_id)
        if not profile:
            profile = UserLTMMemory(user_id=current_user.user_id)
            session.add(profile)
        profile.onboarding_completed_at = datetime.now(timezone.utc)
        await session.commit()
    return {"status": "success"}


@router.get("/chat/topics", summary="List available KB topics (instant, no LLM)")
async def list_topics(
    current_user: Optional[User] = Depends(get_current_user),
) -> dict:
    """Return the ground-truth topic list straight from Postgres.

    Powers the 'Topik' welcome chip: rendered client-side instantly, bypassing
    the chat pipeline's LLM generate step (~2s). Reuses _load_course_names,
    which is TTL-cached, so this is a ~ms in-memory return on the hot path.
    """
    from app.graph.pipeline import _load_course_names
    try:
        topics = await _load_course_names()
    except Exception as exc:
        logger.warning(f"/chat/topics load failed: {exc}")
        topics = []
    return {"topics": topics}


@router.get("/chat/sections", summary="List Moodle sections and their items (instant, no LLM)")
async def list_sections(
    current_user: Optional[User] = Depends(get_current_user),
) -> dict:
    """Return {section_name: [item, ...]} straight from Postgres.

    Powers the UI topic-list button: the chat widget renders an accordion of
    sections; expanding one shows its items; clicking an item sends a normal
    "jelaskan tentang <item>" query. Deterministic (no LLM, no NLP section
    parsing). Reuses _load_section_map (TTL-cached) so this is a ~ms return.
    """
    from app.graph.pipeline import _load_section_map, resolve_user_role
    try:
        user_ctx = {
            "location": current_user.location if current_user else "",
            "grade": current_user.grade if current_user else "",
        }
        resolved_role = resolve_user_role(user_ctx)
        sections = await _load_section_map(resolved_role)
    except Exception as exc:
        logger.warning(f"/chat/sections load failed: {exc}")
        sections = {}
    return {"sections": sections}


@router.post("/chat/sync_memory/{conversation_id}", summary="Sync chat history to Long-Term Memory")
async def sync_memory(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
):
    # ponytail: no-op. LTM sync is driven by the AFK worker (worker.py
    # sync_ltm_task), not this route. Kept as a 200 so any legacy Moodle
    # plugin POSTing here doesn't error; body says "ignored" so it's not
    # mistaken for a real sync. Delete the route once no client calls it.
    return {"status": "ignored", "reason": "handled_by_afk_worker_in_background"}


@router.post("/chat/stream", summary="Stream a CAG response via Server-Sent Events")
async def chat_stream(
    request: ChatRequest,
    req: Request,
    current_user: User = Depends(get_current_user),
):
    # Pipeline semaphore — held for the ENTIRE SSE stream, not just the
    # function body. The release callable is captured by both inner
    # generators (cache-hit and CAG stream) and invoked in their `finally`
    # so the permit returns when the stream ends or the client disconnects.
    # Raises HTTP 503 with Retry-After: 5 on saturation.
    sem_release = await acquire_pipeline_slot_or_503()
    # Everything between the permit acquire and the StreamingResponse return
    # runs BEFORE either generator's `finally` exists, so a raise here (403
    # ownership mismatch, or an embedding/Postgres/summary-LLM blip inside
    # _prepare_cag_context) would otherwise leak the permit permanently —
    # after 12 such raises the whole /chat/stream surface 503s until restart.
    # Release on any pre-stream failure; sem_release is idempotent so the
    # generator's own `finally` stays correct on the success path.
    try:
        start_time = time.perf_counter()
        conversation_id = request.conversation_id or str(uuid.uuid4())
        await _verify_conversation_ownership(conversation_id, current_user)

        stream_ban_ttl = await _get_ban_ttl(current_user.user_id)
        if stream_ban_ttl > 0:
            sem_release()
            _stream_ban_ttl = stream_ban_ttl
            async def banned_generator():
                yield f"event: banned\ndata: {json.dumps({'ban_remaining_seconds': _stream_ban_ttl, 'conversation_id': conversation_id})}\n\n"
                yield f"event: done\ndata: {json.dumps({'sources': [], 'conversation_id': conversation_id, 'cached': False, 'latency_ms': 0.0})}\n\n"
            return StreamingResponse(banned_generator(), media_type="text/event-stream")

        resolved_query = await resolve_numeric_query(request.query, conversation_id)
        from app.graph.pipeline import _apply_glossary
        resolved_query = _apply_glossary(resolved_query)
        # Cache key MUST hash the SAME string the READ path hashes (mirror non-
        # stream chat() at line 877). Used by set_cached_response in _stream_cag_body.
        _raw_query_for_cache = resolved_query
        logger.info("Stream request received", query=request.query[:80], resolved_query=resolved_query[:80] if resolved_query != request.query else None, conversation_id=conversation_id)

        context = await _prepare_cag_context(request, current_user, conversation_id, resolved_query)
    except BaseException:
        sem_release()
        raise
    cached = context.get("cached")
    was_personalized = context.get("was_personalized", False)

    if cached:
        async def _stream_cached():
            try:
                latency_ms = (time.perf_counter() - start_time) * 1000
                if resolved_query != request.query:
                    yield f"event: resolved\ndata: {json.dumps({'resolved_query': resolved_query})}\n\n"

                # Release semaphore BEFORE fake playback — cache hits don't need
                # a pipeline slot. Holding it for ~4s of word-by-word streaming
                # wastes a concurrent slot for zero compute work.
                sem_release()

                words = cached["answer"].split(" ")
                chunk_size = 4
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    if i > 0:
                        chunk = " " + chunk
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                    await asyncio.sleep(0.02)

                sources_list = list(cached.get("sources", []))
                yield f"event: done\ndata: {json.dumps({'sources': sources_list, 'conversation_id': conversation_id, 'cached': True, 'latency_ms': round(latency_ms, 2)})}\n\n"

                await append_to_history(conversation_id=conversation_id, user_message=request.query, assistant_message=cached["answer"])

                # FIX 2: a cache hit is still session activity. Refresh last_active
                # + (re)schedule the AFK LTM sync (the helper updates
                # `rag:last_active` internally) so an all-cache-hit session still
                # persists LTM and the AFK guard doesn't misfire. No memory is
                # re-injected — the cached answer is already final.
                try:
                    await _schedule_afk_ltm_sync(conversation_id, current_user.user_id)
                except Exception as e:
                    logger.warning(f"Cache-hit AFK LTM schedule failed: {e}")
            finally:
                pass  # semaphore already released before playback started

        return StreamingResponse(_stream_cached(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # NOTE: the Redis greeting-cache short-circuit and the Tier-1 fast-path
    # (which bypassed the graph for GREETING/AMBIGUOUS with canned replies) were
    # removed when the pipeline was collapsed to a single conversational LLM.
    # They were the source of the "yang benerlah → identity intro" misroute:
    # edge phrasings hit the regex identity check and got a canned self-intro.
    # All turns now flow through the conversational graph, which handles
    # greetings/identity/meta-turns naturally in one prompt. The Redis exact
    # cache (checked above) still short-circuits byte-identical re-asks.

    initial_state = context["initial_state"]

    async def _stream_cag_raw():
        nonlocal resolved_query
        from app.graph.pipeline import (
            StreamLeakGuard,
            _handle_malicious,
            _pre_processor,
            _sanitize_answer,
            stream_openrouter_generate,
        )

        full_answer = ""
        retrieved_context: list = []
        intent = "KNOWLEDGE"
        stream_intent_scores: dict = {}
        stream_gate_score = None
        stream_total_tokens: int | None = None
        stream_prompt_tokens = 0
        stream_completion_tokens = 0
        stream_cached_tokens = 0
        stream_provider = None
        stream_generation_id = None
        stream_cost = 0.0
        turn_id = str(uuid.uuid4())
        stream_off_scope_detected = False
        leak_guard = StreamLeakGuard()
        token_count = 0
        stream_max_score = None
        is_low_relevance_stream = False
        _logged = False
        restart_buffer = ""

        def _dedupe_stream_text(text: str) -> str:
            nonlocal full_answer, restart_buffer
            if not text:
                return ""
            if restart_buffer:
                restart_buffer += text
                if full_answer.startswith(restart_buffer):
                    return ""
                if restart_buffer.startswith(full_answer):
                    emit = restart_buffer[len(full_answer):]
                    full_answer += emit
                    restart_buffer = ""
                    return emit
                emit = restart_buffer
                full_answer += emit
                restart_buffer = ""
                return emit
            if full_answer and full_answer.startswith(text):
                restart_buffer = text
                return ""
            full_answer += text
            return text

        async def _emit_log() -> None:
            nonlocal _logged, stream_prompt_tokens, stream_cached_tokens
            nonlocal stream_completion_tokens, stream_provider, stream_cost
            nonlocal stream_generation_id, stream_total_tokens
            if _logged:
                return
            _logged = True
            try:
                usage = await _complete_openrouter_usage(OpenRouterUsage(
                    prompt_tokens=stream_prompt_tokens,
                    cached_tokens=stream_cached_tokens,
                    completion_tokens=stream_completion_tokens,
                    provider=stream_provider,
                    cost=stream_cost,
                    generation_id=stream_generation_id,
                ))
                stream_prompt_tokens = usage.prompt_tokens
                stream_cached_tokens = usage.cached_tokens
                stream_completion_tokens = usage.completion_tokens
                stream_provider = usage.provider
                stream_cost = usage.cost
                stream_generation_id = usage.generation_id
                if stream_total_tokens is None and (usage.prompt_tokens or usage.completion_tokens):
                    stream_total_tokens = usage.prompt_tokens + usage.completion_tokens

                await batch_logger.add_log({
                    "turn_id": turn_id,
                    "endpoint": "chat-stream",
                    "conversation_id": conversation_id,
                    "query": request.query,
                    "rewritten_query": resolved_query,
                    "answer": full_answer,
                    "chunks_retrieved": len(retrieved_context),
                    "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                    "llm_tokens_used": stream_total_tokens if stream_total_tokens else token_count,
                    "or_prompt_tokens": stream_prompt_tokens,
                    "or_cached_tokens": stream_cached_tokens,
                    "or_completion_tokens": stream_completion_tokens,
                    "or_provider": stream_provider,
                    "or_generation_id": stream_generation_id,
                    "or_cost": stream_cost,
                    "cache_hit": False,
                    "retrieved_context": retrieved_context,
                    **_quality_log_fields(
                        intent,
                        stream_intent_scores,
                        stream_max_score,
                        _serialize_gate_score(stream_gate_score),
                    ),
                })
            except Exception as log_err:
                logger.warning(f"Stream add_log failed: {log_err}")

        try:
            config: RunnableConfig = {"run_name": "ava-chat-stream"}
            if resolved_query != request.query:
                yield f"event: resolved\ndata: {json.dumps({'resolved_query': resolved_query})}\n\n"

            pre = await _pre_processor(initial_state, config)
            if isinstance(pre, dict):
                initial_state.update(pre)
                intent = pre.get("intent") or intent
                stream_intent_scores = pre.get("intent_scores") or {}
                stream_gate_score = pre.get("gate_score")
                stream_off_scope_detected = intent == "OFF_SCOPE"
                new_rewrite = pre.get("rewritten_query")
                if new_rewrite and new_rewrite != resolved_query:
                    resolved_query = new_rewrite
                    yield f"event: resolved\ndata: {json.dumps({'resolved_query': resolved_query})}\n\n"

            if intent == "MALICIOUS":
                out = await _handle_malicious(initial_state, config)
                msgs = out.get("messages") if isinstance(out, dict) else None
                content = getattr(msgs[-1], "content", "") if msgs else ""
                if content:
                    full_answer += content
                    yield f"data: {json.dumps({'token': content})}\n\n"
            else:
                async for ev in stream_openrouter_generate(initial_state, config=config):
                    ev_type = ev.get("type")
                    if ev_type == "ping":
                        yield ": ping\n\n"
                        continue
                    if ev_type == "usage":
                        usage = ev.get("usage")
                        if usage:
                            stream_prompt_tokens = usage.prompt_tokens
                            stream_cached_tokens = usage.cached_tokens
                            stream_completion_tokens = usage.completion_tokens
                            stream_provider = usage.provider
                            stream_cost = usage.cost
                            stream_generation_id = usage.generation_id
                            stream_total_tokens = usage.prompt_tokens + usage.completion_tokens
                        continue
                    if ev_type != "token":
                        continue
                    emit = _dedupe_stream_text(ev.get("text") or "")
                    if not emit:
                        continue
                    token_count += 1
                    safe = leak_guard.feed(emit)
                    if safe:
                        safe = re.sub(r"[ \t]*[—–][ \t]*", ", ", safe)
                        yield f"data: {json.dumps({'token': safe})}\n\n"
                    if token_count % 10 == 0 and await req.is_disconnected():
                        logger.info("Client disconnected mid-stream", conversation_id=conversation_id, tokens=token_count)
                        await _emit_log()
                        return

            tail = leak_guard.flush()
            if tail:
                yield f"data: {json.dumps({'token': tail})}\n\n"
            if leak_guard.leak_detected:
                full_answer = tail

            cleaned_answer = _sanitize_answer(full_answer)
            if cleaned_answer != full_answer:
                full_answer = cleaned_answer
            if "[OFFSCOPE]" in full_answer.upper():
                stream_off_scope_detected = True

            if not full_answer.strip():
                logger.warning(
                    "Raw OpenRouter stream produced empty answer; retrying graph ainvoke",
                    conversation_id=conversation_id,
                )
                try:
                    fb_result = await asyncio.wait_for(
                        get_cag_graph().ainvoke(initial_state, config={"run_name": "ava-chat-stream-fb"}),
                        timeout=settings.pipeline_total_timeout_s,
                    )
                    fb_msgs = fb_result.get("messages") or []
                    if fb_result.get("off_scope_detected"):
                        stream_off_scope_detected = True
                    if fb_msgs:
                        fb_msg = fb_msgs[-1]
                        fb_content = getattr(fb_msg, "content", None) or ""
                        if isinstance(fb_content, str) and fb_content.strip():
                            full_answer = _sanitize_answer(fb_content)
                            usage = extract_openrouter_usage(fb_msg)
                            stream_prompt_tokens = usage.prompt_tokens
                            stream_cached_tokens = usage.cached_tokens
                            stream_completion_tokens = usage.completion_tokens
                            stream_provider = usage.provider
                            stream_cost = usage.cost
                            stream_generation_id = usage.generation_id
                            if usage.prompt_tokens or usage.completion_tokens:
                                stream_total_tokens = usage.prompt_tokens + usage.completion_tokens
                            retrieved_context = fb_result.get("retrieved_context") or retrieved_context
                            sources = _extract_sources(retrieved_context)
                            yield f"data: {json.dumps({'token': full_answer})}\n\n"
                except Exception as fb_exc:
                    logger.warning(f"Fallback ainvoke failed after empty raw stream: {type(fb_exc).__name__}: {fb_exc}")

            if not full_answer.strip():
                yield f"event: error\ndata: {json.dumps({'error': 'empty response, please retry'})}\n\n"
                await _emit_log()
                return

            latency_ms = (time.perf_counter() - start_time) * 1000
            sources = _extract_sources(retrieved_context)
            coaching_topic = None
            try:
                if (not request.coaching_mode) and intent == "KNOWLEDGE":
                    titles = [s["title"] for s in sources if s.get("title") and s["title"] != "Unknown"]
                    if titles:
                        coaching_topic = await bump_topic_streak(conversation_id, Counter(titles).most_common(1)[0][0])
            except Exception as exc:
                logger.debug(f"topic-streak hook skipped: {exc}")

            suggest_coaching = bool(coaching_topic)
            coaching_done = request.coaching_mode and intent == "COACHING" and "?" not in full_answer
            stream_new_ban_ttl = 0
            if stream_off_scope_detected:
                new_count, _ = await _handle_off_scope_violation(current_user.user_id)
                if new_count == settings.warning_off_scope_violations:
                    warning_text = "\n\n**Peringatan**: Harap tanyakan hal seputar materi Amartha. Jika Anda terus bertanya di luar topik sekali lagi, Ai Trainer akan dinonaktifkan sementara."
                    yield f"data: {json.dumps({'token': warning_text})}\n\n"
                    full_answer += warning_text
                elif new_count >= settings.max_off_scope_violations:
                    stream_new_ban_ttl = await _get_ban_ttl(current_user.user_id)

            try:
                await append_to_history(conversation_id=conversation_id, user_message=request.query, assistant_message=full_answer)
            except Exception as hist_err:
                logger.warning(f"append_to_history (pre-done) failed: {hist_err}")
            try:
                await add_seen_chunk_ids(conversation_id, retrieved_context)
            except Exception as seen_err:
                logger.debug(f"seen chunk tracking skipped: {seen_err}")

            done_payload: dict = {
                "sources": sources,
                "conversation_id": conversation_id,
                "cached": False,
                "latency_ms": round(latency_ms, 2),
                "suggest_coaching": suggest_coaching,
                "coaching_topic": coaching_topic,
                "coaching_done": coaching_done,
            }
            if stream_new_ban_ttl > 0:
                done_payload["ban_remaining_seconds"] = stream_new_ban_ttl
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

            try:
                if (
                    intent not in ("GREETING", "AMBIGUOUS", "MALICIOUS", "TOPIC_LIST", "COACHING", "OFF_SCOPE")
                    and not is_low_relevance_stream
                    and not _is_bare_affirmation(request.query)
                    and not context.get("skip_cache", False)
                ):
                    ns = cache_namespace_for(was_personalized=was_personalized, user_id=current_user.user_id)
                    await set_cached_response(
                        query=_raw_query_for_cache,
                        answer=full_answer,
                        sources=sources,
                        course_id=request.course_id,
                        cache_namespace=ns,
                    )
                await _emit_log()
                await _schedule_afk_ltm_sync(conversation_id, current_user.user_id)
                await _track_session_courses(conversation_id, retrieved_context)
                if _should_eval_turn(
                    intent=intent,
                    intent_scores=stream_intent_scores,
                    max_dense_score=stream_max_score,
                    answer=full_answer,
                    is_low_relevance=is_low_relevance_stream,
                ):
                    await _enqueue_eval(
                        turn_id=turn_id,
                        query=resolved_query,
                        answer=full_answer,
                        retrieved_context=retrieved_context,
                        intent=intent,
                        intent_scores=stream_intent_scores,
                    )
            except Exception as bg_err:
                logger.warning(f"Stream background task error: {bg_err}")
        except Exception:
            logger.exception("Raw OpenRouter stream error", query=request.query[:60])
            yield f"event: error\ndata: {json.dumps({'error': 'CAG pipeline failed'})}\n\n"
            await _emit_log()
        finally:
            await _emit_log()
            sem_release()

    return StreamingResponse(_stream_cag_raw(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    cag_graph = get_cag_graph()

    async def _stream_cag_body():
        nonlocal resolved_query
        from app.graph.pipeline import StreamLeakGuard, _sanitize_answer
        full_answer = ""
        retrieved_context: list = []
        intent = "KNOWLEDGE"
        stream_intent_scores: dict = {}
        # Semantic-gate trace captured from pre_processor's on_chain_end output
        # so stream eval/cache decisions mirror the non-stream path.
        # Semantic-gate trace (Jun 2026) — captured from pre_processor's
        # on_chain_end output. Mirrors the non-stream path's `result.gate_score`.
        stream_gate_score = None
        # Real total LLM tokens (input+output) from the final generate_node
        # AIMessage's response_metadata.token_usage. Populated by the
        # on_chain_end capture below; defaults to None (and falls back to
        # token_count in the worst case where the provider omits usage).
        stream_total_tokens: int | None = None
        stream_prompt_tokens = 0
        stream_completion_tokens = 0
        stream_cached_tokens = 0
        stream_provider = None
        stream_generation_id = None
        stream_cost = 0.0
        turn_id = str(uuid.uuid4())  # correlates this turn's agent_logs row to its async eval score
        stream_off_scope_detected = False
        leak_guard = StreamLeakGuard()
        token_count = 0
        # ponytail: hoisted defaults so the finally-block log emitter can run on
        # EVERY exit path (normal done, client disconnect mid-stream, stall
        # timeout, pipeline error, GeneratorExit). Before this, add_log lived
        # only on the success path → knowledge turns with long answers routinely
        # lost their agent_logs row when the client closed the SSE before
        # reading to the end, so the dashboard only ever showed greetings.
        stream_max_score = None
        is_low_relevance_stream = False
        _logged = False
        restart_buffer = ""



        def _dedupe_stream_text(text: str) -> str:
            nonlocal full_answer, restart_buffer
            if not text:
                return ""
            if restart_buffer:
                restart_buffer += text
                if full_answer.startswith(restart_buffer):
                    return ""
                if restart_buffer.startswith(full_answer):
                    emit = restart_buffer[len(full_answer):]
                    full_answer += emit
                    restart_buffer = ""
                    return emit
                emit = restart_buffer
                full_answer += emit
                restart_buffer = ""
                return emit
            if full_answer and full_answer.startswith(text):
                restart_buffer = text
                return ""
            full_answer += text
            return text

        async def _emit_log() -> None:
            """Write the agent_logs row once, regardless of how the stream ended.

            Cache/history/eval stay on the success path only (they gate on a
            complete answer); this is logging-only so a partial/disconnected
            turn is still observable in the dashboard — answer may be empty,
            retrieved_context stays empty in CAG mode unless a future node emits
            explicit source context before the break.
            """
            nonlocal _logged, stream_prompt_tokens, stream_cached_tokens
            nonlocal stream_completion_tokens, stream_provider, stream_cost
            nonlocal stream_generation_id, stream_total_tokens
            if _logged:
                return
            _logged = True
            try:
                usage = await _complete_openrouter_usage(OpenRouterUsage(
                    prompt_tokens=stream_prompt_tokens,
                    cached_tokens=stream_cached_tokens,
                    completion_tokens=stream_completion_tokens,
                    provider=stream_provider,
                    cost=stream_cost,
                    generation_id=stream_generation_id,
                ))
                stream_prompt_tokens = usage.prompt_tokens
                stream_cached_tokens = usage.cached_tokens
                stream_completion_tokens = usage.completion_tokens
                stream_provider = usage.provider
                stream_cost = usage.cost
                stream_generation_id = usage.generation_id
                if stream_total_tokens is None and (usage.prompt_tokens or usage.completion_tokens):
                    stream_total_tokens = usage.prompt_tokens + usage.completion_tokens

                await batch_logger.add_log({
                    "turn_id": turn_id,
                    "endpoint": "chat-stream",
                    "conversation_id": conversation_id,
                    "query": request.query,
                    "rewritten_query": resolved_query,
                    "answer": full_answer,
                    "chunks_retrieved": len(retrieved_context),
                    "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                    "llm_tokens_used": stream_total_tokens if stream_total_tokens else token_count,
                    "or_prompt_tokens": stream_prompt_tokens,
                    "or_cached_tokens": stream_cached_tokens,
                    "or_completion_tokens": stream_completion_tokens,
                    "or_provider": stream_provider,
                    "or_generation_id": stream_generation_id,
                    "or_cost": stream_cost,
                    "cache_hit": False,
                    "retrieved_context": retrieved_context,
                    **_quality_log_fields(
                        intent,
                        stream_intent_scores,
                        stream_max_score,
                        _serialize_gate_score(stream_gate_score),
                    ),
                })
            except Exception as log_err:
                logger.warning(f"Stream add_log failed: {log_err}")

        try:
            config: RunnableConfig = {"run_name": "ava-chat-stream"}

            if resolved_query != request.query:
                yield f"event: resolved\ndata: {json.dumps({'resolved_query': resolved_query})}\n\n"

            token_count = 0
            # Canned-response nodes (malicious / topic_list / greeting
            # Tier-2 / ambiguity Tier-2 / low_relevance / off_scope) return
            # an AIMessage directly without invoking an LLM, so no
            # `on_chat_model_stream` event fires for them. We emit their
            # content from on_chain_end instead. Per-node flag prevents
            # double-emission when greeting/ambiguity fall through to LLM
            # (e.g. "hmm" routes through LLM in _handle_ambiguity).
            streamed_nodes: set[str] = set()

            # Manual iteration (not `async for`) so each event fetch is bounded
            # by a STALL timeout that resets on every emission. astream_events
            # itself has no timeout: if upstream hangs with zero tokens, an
            # `async for` would block __anext__ forever, the generator would
            # never exit, and the `finally` that releases the pipeline permit
            # would never run — leaking a slot exactly like the pre-stream path
            # did. wait_for bounds a SILENT graph to one stall window; a
            # slow-but-streaming answer never trips it because every token is an
            # event that resets the clock.
            _events = cag_graph.astream_events(initial_state, config=config, version="v2")
            # Non-stream generate is atomic — astream_events emits NOTHING for the
            # 2-9s the LLM call blocks. That idle gap lets ngrok/free proxies cut
            # the SSE connection (and Starlette's own 60s no-yield timeout abort
            # the response). A short wait_for + emit an SSE comment ping (": ping",
            # ignored by the browser EventSource/reader) on each timeout keeps the
            # connection alive while the LLM works. A real stall is still detected:
            # if NO genuine event arrives within `_stall_s` (75s), we error out.
            _ping_s = 2.0
            _stall_s = settings.pipeline_stream_stall_timeout_s
            import time as _time
            _last_real_event = _time.monotonic()
            
            anext_task = None
            while True:
                if anext_task is None:
                    anext_task = asyncio.create_task(_events.__anext__())
                
                done, _ = await asyncio.wait([anext_task], timeout=_ping_s)
                
                if anext_task in done:
                    try:
                        event = anext_task.result()
                    except StopAsyncIteration:
                        break
                    except Exception as loop_exc:
                        # Re-raise to trigger the same outer fallback logic as before
                        raise loop_exc
                        
                    anext_task = None
                    _last_real_event = _time.monotonic()
                else:
                    # No event in 2s. If we're still within the overall stall
                    # window, emit a keepalive ping and keep waiting. Only a
                    # sustained 75s silence is a real stall.
                    if _time.monotonic() - _last_real_event < _stall_s:
                        yield ": ping\n\n"
                        continue
                    
                    logger.error(
                        "Stream stalled — no events within stall window; freeing slot",
                        stall_s=_stall_s,
                        conversation_id=conversation_id,
                        tokens_so_far=token_count,
                    )
                    if anext_task and not anext_task.done():
                        anext_task.cancel()
                    yield f"event: error\ndata: {json.dumps({'error': 'Response timed out'})}\n\n"
                    await _emit_log()
                    return

                kind = event.get("event", "")

                if kind == "on_chain_end" and event.get("name") == "pre_processor":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        if "intent" in output:
                            intent = output["intent"]
                        if "intent_scores" in output:
                            stream_intent_scores = output.get("intent_scores") or {}
                        if "gate_score" in output:
                            stream_gate_score = output.get("gate_score")
                        if "rewritten_query" in output:
                            new_rewrite = output.get("rewritten_query")
                            if new_rewrite and new_rewrite != resolved_query:
                                resolved_query = new_rewrite
                                yield f"event: resolved\ndata: {json.dumps({'resolved_query': resolved_query})}\n\n"

                # Real token usage for the agent_logs row. generate_node
                # returns the AIMessage in its `messages` field; the OpenRouter
                # / Gemini provider writes the final token_usage to the
                # message's response_metadata. This is the SAME source the
                # non-stream path reads on chat.py:839. Without this, stream
                # rows log the chunk-event counter (1-3 per token emitted) and
                # show 4-24 instead of the real 1000-3000 — a quiet data
                # corruption that made the dashboard's token column useless.
                if kind == "on_chain_end" and event.get("name") == "generate_node":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        if output.get("off_scope_detected"):
                            stream_off_scope_detected = True
                        msgs_out = output.get("messages") or []
                        if msgs_out:
                            last_msg = msgs_out[-1]
                            # generate_node now runs with streaming=True (see
                            # client.py get_generate_llm) so on_chat_model_stream
                            # fires per-token (handled at the handler below) and
                            # `streamed_nodes` marks the node as already emitted.
                            # This on_chain_end block is the FALLBACK: if streaming
                            # produced zero tokens (provider flake / stream corrupt
                            # surfaced as an empty/partial AIMessage), `streamed_nodes`
                            # won't contain generate_node → emit the whole content
                            # here as one token through the leak guard.
                            # History: streaming was disabled because google-ai-studio
                            # fallback corrupted SSE mid-generation. Generator is now
                            # deepseek-v4-flash pinned to alibaba,baidu,novita —
                            # google-ai-studio is no longer in the path. If a pinned
                            # provider corrupts the stream, the safety net at ~L1761
                            # re-runs the graph non-stream.
                            content = getattr(last_msg, "content", None) or ""
                            if content:
                                emit = ""
                                if "generate_node" not in streamed_nodes:
                                    emit = _dedupe_stream_text(content)
                                elif isinstance(content, str) and content.startswith(full_answer):
                                    emit = content[len(full_answer):]
                                    full_answer = content
                                if emit:
                                    safe = leak_guard.feed(emit)
                                    safe = re.sub(r"[ \t]*[—–][ \t]*", ", ", safe)
                                    if safe:
                                        yield f"data: {json.dumps({'token': safe})}\n\n"
                            rm = getattr(last_msg, "response_metadata", None) or {}
                            tu = rm.get("token_usage") or {}
                            usage = extract_openrouter_usage(last_msg)
                            stream_generation_id = usage.generation_id
                            real_total = tu.get("total_tokens")
                            if usage.prompt_tokens or usage.completion_tokens:
                                stream_total_tokens = real_total if isinstance(real_total, int) and real_total > 0 else usage.prompt_tokens + usage.completion_tokens
                                stream_prompt_tokens = usage.prompt_tokens
                                stream_completion_tokens = usage.completion_tokens
                                stream_cached_tokens = usage.cached_tokens
                                stream_provider = usage.provider
                                stream_cost = usage.cost
                # Canned-response nodes return an AIMessage directly without
                # invoking an LLM, so no `on_chat_model_stream` event fires
                # for them. Emit their content from on_chain_end instead.
                # Covers: greeting/ambiguity (Tier-2 hardcoded, no LLM),
                # malicious/topic_list/low_relevance/off_scope (deterministic
                # canned replies). Skip if LLM streaming already emitted
                # tokens for this node (e.g. "hmm" → LLM ambiguity).
                # `malicious` and `low_relevance` are nodes that return an AIMessage
                # directly without an LLM call (no on_chat_model_stream fires),
                # so emit their content here. All other intents now flow through
                # generate_node, which streams via on_chat_model_stream below.
                if (
                    kind == "on_chain_end"
                    and event.get("name") in ("malicious", "low_relevance")
                    and event.get("name") not in streamed_nodes
                ):
                    out = event.get("data", {}).get("output", {})
                    msgs = out.get("messages") if isinstance(out, dict) else None
                    if msgs:
                        content = getattr(msgs[-1], "content", None) or (
                            msgs[-1].get("content") if isinstance(msgs[-1], dict) else ""
                        )
                        if content:
                            full_answer += content
                            yield f"data: {json.dumps({'token': content})}\n\n"

                if kind == "on_chat_model_stream":
                    node_name = event.get("metadata", {}).get("langgraph_node")
                    if node_name == "generate_node":
                        streamed_nodes.add(node_name)
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            token = chunk.content
                            emit = _dedupe_stream_text(token)
                            token_count += 1

                            # Pass through leak guard. Greeting/ambiguity
                            # don't carry retrieved_context, but the guard
                            # is a no-op on clean preambles so it's safe.
                            safe = leak_guard.feed(emit)
                            if safe:
                                # Model still emits em/en-dashes despite the prompt ban. Swap to a
                                # comma (collapsing same-token surrounding spaces) so it reads
                                # natural instead of a stray hyphen. Per-token: a dash split across
                                # token boundaries is rare for this tokenizer.
                                safe = re.sub(r"[ \t]*[—–][ \t]*", ", ", safe)
                                yield f"data: {json.dumps({'token': safe})}\n\n"

                            # Periodic disconnect check — bail out early if user closed the tab.
                            if token_count % 10 == 0 and await req.is_disconnected():
                                logger.info("Client disconnected mid-stream", conversation_id=conversation_id, tokens=token_count)
                                await _emit_log()
                                return

        except Exception:
            logger.exception("Stream pipeline error", query=request.query[:60])
            yield f"event: error\ndata: {json.dumps({'error': 'CAG pipeline failed'})}\n\n"
            await _emit_log()
            return

        # Drain any buffered preamble. If the guard caught a leak, this
        # returns the sanitized version — emit it as a single token so the
        # user sees a coherent answer instead of nothing or raw context.
        tail = leak_guard.flush()
        if tail:
            yield f"data: {json.dumps({'token': tail})}\n\n"
        if leak_guard.leak_detected:
            # full_answer accumulated raw tokens (before guard); replace it
            # with the sanitized version so cache/history/eval store the
            # clean text. Combined sanitized output = whatever was already
            # streamed (nothing, since guard buffered) + tail.
            full_answer = tail

        latency_ms = (time.perf_counter() - start_time) * 1000
        sources = _extract_sources(retrieved_context)

        # Sanitize streamed output before persistence — if the LLM leaked
        # an <retrieved_context>/<user_history>/etc. block (Gemini Flash
        # Lite occasionally does this), the StreamLeakGuard above already
        # caught preamble leaks, but a leak that started mid-stream would
        # bypass the guard. This is the belt-and-suspenders pass for
        # cache/history/eval persistence. Cheap regex; no-op when clean.
        cleaned_answer = _sanitize_answer(full_answer)
        if cleaned_answer != full_answer:
            logger.warning(
                "Stream output leaked instruction block — sanitized before "
                f"cache/history/eval (orig_len={len(full_answer)} "
                f"clean_len={len(cleaned_answer)} conv={conversation_id})"
            )
            full_answer = cleaned_answer

        # Topic-streak auto-hook (moved from frontend): bump the same-topic
        # counter; returns the topic once the user has stuck with one topic for
        # `coaching_streak_threshold` turns, which drives a topic-specific offer.
        coaching_topic = None
        try:
            if (not request.coaching_mode) and intent == "KNOWLEDGE":
                titles = [s["title"] for s in sources if s.get("title") and s["title"] != "Unknown"]
                if titles:
                    dominant = Counter(titles).most_common(1)[0][0]
                    coaching_topic = await bump_topic_streak(conversation_id, dominant)
        except Exception as exc:
            logger.debug(f"topic-streak hook skipped: {exc}")

        # Affinity auto-hook: should we OFFER coaching after this answer? Only for
        # a normal KNOWLEDGE turn when coaching is OFF. Reuses the embedding
        # already computed for LTM (no extra embed call). Skipped when the streak
        # hook already fired. Never gates/hijacks the answer — a soft signal the
        # frontend turns into a clickable offer. Wrapped so a scoring hiccup can
        # never break the stream.
        suggest_coaching = bool(coaching_topic)

        # Coaching wrap-up signal — mirror of suggest_coaching, opposite direction.
        # When we're IN coaching mode and Ava delivered a final answer rather than a
        # guiding question, the Socratic loop is done → tell the frontend so it can
        # offer "back to Mentoring". A guiding-question turn always carries a "?"; a
        # wrap-up / frustration answer carries none (SOCRATIC_PROMPT enforces this).
        # Gated on the real intent (COACHING) so a greeting/off-scope turn while the
        # toggle is on never triggers it. Computed server-side: no marker ever
        # streams to the user, so there's nothing to leak.
        coaching_done = False

        # ponytail: persist history BEFORE the client gets the "done" signal.
        # Otherwise the frontend fires the next turn immediately on done, and
        # that turn's rewrite reads Redis before this append commits → it sees
        # stale history and resolves "yang pertama" against the PREVIOUS topic
        # (e.g. Microfinance) instead of the list Ava just gave. Cache-set/eval/
        # LTM stay after done (they don't gate the next turn's history read).
        # Empty-answer safety net. Two failure modes converge here:
        #   1. OpenRouter google-vertex flake: ainvoke returns "successfully"
        #      with 0 completion tokens (no exception → generate_node retry
        #      never triggers).
        #   2. LangGraph astream_events v2 + streaming=False ChatOpenAI
        #      sometimes does NOT emit generate_node's on_chain_end (the
        #      node ran, the LLM answered, but the event never lands) → the
        #      stream loop ends with full_answer empty even though the graph
        #      produced a real answer.
        # Both leave the user with an empty bubble → frontend "Hmm, jawabanku...".
        # Fix: re-run the graph via ainvoke (full non-stream, generate_node's
        # 4-attempt retry applies) and emit the result as a single token.
        # ainvoke reads result["messages"][-1].content directly — no dependency
        # on astream_events event emission. The 2nd run almost always succeeds
        # because the flake is sub-second transient. Bounded by the graph's
        # own timeout; the keepalive ping above keeps ngrok alive during it.
        if not full_answer.strip():
            logger.warning(
                "Stream produced empty answer (provider flake OR astream_events "
                f"dropped generate_node event); re-running graph via ainvoke "
                f"conv={conversation_id}"
            )
            try:
                _fb_result = await asyncio.wait_for(
                    cag_graph.ainvoke(initial_state, config={"run_name": "ava-chat-stream-fb"}),
                    timeout=settings.pipeline_total_timeout_s,
                )
                if _fb_result.get("off_scope_detected"):
                    stream_off_scope_detected = True
                _fb_msgs = _fb_result.get("messages") or []
                if _fb_msgs:
                    _fb_msg = _fb_msgs[-1]
                    _fb_content = getattr(_fb_msg, "content", None) or ""
                    if isinstance(_fb_content, str) and _fb_content.strip():
                        _fb_content = _sanitize_answer(_fb_content)
                        full_answer = _fb_content
                        # Refresh sources/retrieved_context from the fallback run
                        # so the done payload + agent_logs reflect what was used.
                        _fb_ctx = _fb_result.get("retrieved_context") or []
                        if _fb_ctx:
                            retrieved_context = _fb_ctx
                            sources = _extract_sources(retrieved_context)
                        # Token usage from the fallback AIMessage — check both
                        # response_metadata.token_usage (raw) and usage_metadata
                        # (LangChain-normalized) like _log_cache_usage does.
                        _rm = getattr(_fb_msg, "response_metadata", None) or {}
                        _tu = _rm.get("token_usage") or {}
                        _usage = extract_openrouter_usage(_fb_msg)
                        stream_generation_id = _usage.generation_id
                        _real_total = _tu.get("total_tokens")
                        if _usage.prompt_tokens or _usage.completion_tokens:
                            stream_total_tokens = _real_total if isinstance(_real_total, int) and _real_total > 0 else _usage.prompt_tokens + _usage.completion_tokens
                            stream_prompt_tokens = _usage.prompt_tokens
                            stream_completion_tokens = _usage.completion_tokens
                            stream_cached_tokens = _usage.cached_tokens
                            stream_provider = _usage.provider
                            stream_cost = _usage.cost

                        yield f"data: {json.dumps({'token': _fb_content})}\n\n"
            except Exception as fb_exc:
                logger.warning(f"Fallback ainvoke also failed: {type(fb_exc).__name__}: {fb_exc}")

        if not full_answer.strip():
            # Both the stream AND the fallback ainvoke came back empty — the
            # upstream is genuinely down. Surface a retryable error so the
            # frontend shows the "Waduh, coba lagi" message instead of a blank.
            yield f"event: error\ndata: {json.dumps({'error': 'empty response, please retry'})}\n\n"
            await _emit_log()
            return

        stream_new_ban_ttl: int = 0
        if stream_off_scope_detected:
            new_count, just_banned = await _handle_off_scope_violation(current_user.user_id)
            if new_count == settings.warning_off_scope_violations:
                warning_text = "\n\n**Peringatan**: Harap tanyakan hal seputar materi Amartha. Jika Anda terus bertanya di luar topik sekali lagi, Ai Trainer akan dinonaktifkan sementara."
                yield f"data: {json.dumps({'token': warning_text})}\n\n"
                full_answer += warning_text
            elif new_count >= settings.max_off_scope_violations:
                stream_new_ban_ttl = await _get_ban_ttl(current_user.user_id)

        try:
            await append_to_history(conversation_id=conversation_id, user_message=request.query, assistant_message=full_answer)
        except Exception as hist_err:
            logger.warning(f"append_to_history (pre-done) failed: {hist_err}")
        try:
            await add_seen_chunk_ids(conversation_id, retrieved_context)
        except Exception as seen_err:
            logger.debug(f"seen chunk tracking skipped: {seen_err}")

        done_payload: dict = {'sources': sources, 'conversation_id': conversation_id, 'cached': False, 'latency_ms': round(latency_ms, 2), 'suggest_coaching': suggest_coaching, 'coaching_topic': coaching_topic, 'coaching_done': coaching_done}
        if stream_new_ban_ttl > 0:
            done_payload['ban_remaining_seconds'] = stream_new_ban_ttl
        yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

        try:
            stream_dense_scores = [c.get("dense_score") for c in retrieved_context if isinstance(c.get("dense_score"), (int, float))]
            stream_max_score = max(stream_dense_scores) if stream_dense_scores else None
            # Under CAG mode, retrieval is disabled, so there is no low-relevance retrieval gate.
            is_low_relevance_stream = False
            # Mirrors the non-stream cache gate. The former MENTOR-shape
            # exclusion (learning_context >= threshold) was dropped with the
            # pre-processor slim-down — how-to answers are cacheable now. The
            # cache is Redis exact-match only (no semantic layer), so a cached
            # answer is only ever re-served to a byte-identical re-ask.
            if (
                intent not in ("GREETING", "AMBIGUOUS", "MALICIOUS", "TOPIC_LIST", "COACHING", "OFF_SCOPE")
                and not is_low_relevance_stream
                and not _is_bare_affirmation(request.query)
                and not context.get("skip_cache", False)
            ):
                ns = cache_namespace_for(was_personalized=was_personalized, user_id=current_user.user_id)
                await set_cached_response(
                    query=_raw_query_for_cache,
                    answer=full_answer,
                    sources=sources,
                    # Key the write on request.course_id — SAME value the read path
                    # uses (the read runs before retrieval, so it can't auto-detect
                    # a course). Keying on the chunk-detected effective_course_id
                    # would land the answer at :cache:600: while every re-ask looks
                    # in :cache:global:, a key the read can never produce → 0 hits.
                    course_id=request.course_id,
                    cache_namespace=ns,
                )
            await _emit_log()
            await _schedule_afk_ltm_sync(conversation_id, current_user.user_id)
            await _track_session_courses(conversation_id, retrieved_context)

            if _should_eval_turn(
                intent=intent,
                intent_scores=stream_intent_scores,
                max_dense_score=stream_max_score,
                answer=full_answer,
                is_low_relevance=is_low_relevance_stream,
            ):
                await _enqueue_eval(
                    turn_id=turn_id,
                    query=resolved_query,
                    answer=full_answer,
                    retrieved_context=retrieved_context,
                    intent=intent,
                    intent_scores=stream_intent_scores,
                )
        except Exception as bg_err:
            logger.warning(f"Stream background task error: {bg_err}")
        finally:
            # Log on EVERY exit path, not just success. A knowledge turn whose
            # answer is long enough that the client disconnects mid-stream used
            # to never reach add_log (it lived above the success-only tail) →
            # the dashboard saw greetings but not knowledge. _emit_log is
            # idempotent (_logged guard) so the success-path call is a no-op here.
            await _emit_log()

    async def _stream_cag():
        # Outer wrapper guarantees the pipeline permit is released on EVERY
        # exit path of the inner generator: normal completion, early `return`
        # (client disconnect mid-stream, pipeline error, stall timeout), an
        # unhandled raise, OR GeneratorExit when the client closes the
        # connection while we're blocked on a `yield`. The inner body has
        # sequential (not nested) try blocks, so its own early returns exit
        # before any single finally — only this outer finally sees them all.
        # sem_release is idempotent, so a pre-stream release (ownership 403 /
            # _prepare_cag_context blip) followed by this one is safe.
        try:
            async for _chunk in _stream_cag_body():
                yield _chunk
        finally:
            sem_release()

    return StreamingResponse(_stream_cag(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
