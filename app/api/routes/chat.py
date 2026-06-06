import asyncio
import json
import random
import time
import uuid
import datetime
from typing import Optional
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from loguru import logger

# Moved inline imports to file-level
from langchain_core.messages import HumanMessage, AIMessage

from app.agents.memory import (
    append_to_history,
    get_conversation_history,
    resolve_numeric_query,
    get_or_summarize_history,
    clear_conversation_history
)
from app.api.schemas import ChatRequest, ChatResponse, SourceReference
from app.api.concurrency import acquire_pipeline_slot, acquire_pipeline_slot_or_503
from app.database.postgres import AsyncSessionLocal
from app.database.models import UserProfile
from app.graph.pipeline import get_rag_graph
from app.utils.cache import get_cached_response, set_cached_response
from app.config.settings import get_settings
from app.utils.logger_batch import batch_logger
from app.api.auth import get_current_user, User
from app.llm.client import get_cheap_llm
from app.api.user_utils import is_real_user
from app.agents.long_term_memory_qdrant import qdrant_ltm
from app.database.redis_client import get_redis_client
from app.api.routes.ingest import get_arq_redis
from app.worker import sync_ltm_task, eval_turn_task  # noqa: E402

router = APIRouter()
settings = get_settings()

# Module-level set to keep references to background stream tasks alive
# (prevents GC from cancelling them mid-flight after the SSE generator returns).
_stream_bg_tasks: set[asyncio.Task] = set()

async def _schedule_afk_ltm_sync(conv_id: str, u_id: str):
    redis_client = get_redis_client()
    afk_seconds = settings.ltm_afk_threshold_seconds
    # Update last_active timestamp on every message (rolling window).
    # TTL must outlive the AFK threshold + a buffer so the worker can still
    # see it when the deferred job fires.
    await redis_client.set(
        f"rag:last_active:{conv_id}",
        str(time.time()),
        ex=afk_seconds + 3600,
    )

    # Deduplication: only enqueue if no task is already queued for this conversation
    sched_key = f"rag:ltm:scheduled:{conv_id}"
    already_scheduled = await redis_client.exists(sched_key)
    if already_scheduled:
        logger.debug("AFK LTM sync already scheduled, skipping", conversation_id=conv_id)
        return

    try:
        arq_redis = await get_arq_redis()
        await arq_redis.enqueue_job(
            'sync_ltm_task',
            conv_id,
            u_id,
            _defer_by=datetime.timedelta(seconds=afk_seconds)
        )
        # Mark as scheduled. TTL slightly larger than the defer window so the
        # key is still alive when the worker checks it.
        await redis_client.set(sched_key, "1", ex=afk_seconds + 600)
        logger.debug("AFK LTM sync scheduled", conversation_id=conv_id, defer_s=afk_seconds)
    except Exception as e:
        logger.warning(f"Failed to schedule AFK LTM sync: {e}")


async def _track_session_courses(conv_id: str, retrieved_context: list) -> None:
    """Capture distinct course_names from retrieved_context into a Redis set.

    LTM sync worker reads this set instead of asking the LLM to extract course
    names — the LLM hallucinates names that don't exist in the KB ("Amartha
    products"), but `chunk.metadata.course_name` is ground truth from the
    Moodle ingestion. Set has the same TTL as the LTM scheduling window.
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
        redis_client = get_redis_client()
        key = f"rag:courses:{conv_id}"
        await redis_client.sadd(key, *names)
        # Mirror the last_active TTL so the set is still readable when the
        # AFK worker fires after `ltm_afk_threshold_seconds`.
        await redis_client.expire(key, settings.ltm_afk_threshold_seconds + 3600)
    except Exception as e:
        logger.warning(f"Failed to track session courses: {e}")

DEV_BYPASS_USER_ID = "dev_user_123"


# Intents whose answers don't go through the RAG generator — no faithfulness
# signal worth measuring. GREETING/AMBIGUOUS are canned, MALICIOUS is a refusal,
# TOPIC_LIST reads from Postgres metadata, BRAINSTORM is reasoning-only,
# OFF_SCOPE is a canned redirect with no retrieval.
_EVAL_SKIP_INTENTS = {"GREETING", "AMBIGUOUS", "MALICIOUS", "TOPIC_LIST", "BRAINSTORM", "OFF_SCOPE"}


def _should_eval_turn(
    *,
    intent: str | None,
    intent_scores: dict | None,
    max_dense_score: float | None,
    answer: str | None,
) -> bool:
    """Sampling decision for post-hoc evaluation.

    Evaluates when ANY of:
    - Random draw under `eval_sample_rate` (baseline drift signal).
    - Empathy axis ≥ threshold (vent / resign — high-stakes turn).
    - Top dense cosine below threshold (suspected retrieval miss / potential halu).

    Skips canned-response intents and empty answers regardless.
    """
    if not settings.eval_enabled:
        return False
    if not answer or not answer.strip():
        return False
    if intent in _EVAL_SKIP_INTENTS:
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
        await eval_turn_task.enqueue(
            query=query,
            answer=answer,
            retrieved_context=retrieved_context or [],
            intent=intent,
            intent_scores=intent_scores or {},
            turn_id=turn_id,
        )
    except Exception as e:
        logger.warning(f"Failed to enqueue eval task: {e}")


def _quality_log_fields(
    intent: Optional[str],
    intent_scores: Optional[dict],
    max_dense_score: Optional[float],
) -> dict:
    """Build the durable quality-signal columns for an agent_logs row.

    Persists intent + retrieval signal to Postgres so monitoring works
    without any external tracing backend.
    """
    scores = intent_scores or {}

    def _f(key):
        v = scores.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    return {
        "intent": intent,
        "needs_lookup": _f("needs_lookup"),
        "needs_reasoning": _f("needs_reasoning"),
        "needs_empathy": _f("needs_empathy"),
        "max_dense_score": float(max_dense_score) if isinstance(max_dense_score, (int, float)) else None,
    }


async def _verify_conversation_ownership(conversation_id: str, current_user: User):
    """Ensure the user owns this conversation before accessing history.
    
    Migration note: if the stored owner is the dev bypass user (dev_user_123),
    the real authenticated user is allowed to take over ownership seamlessly.
    This handles the shift from no-JWT to JWT-authenticated requests.
    """
    redis = get_redis_client()
    owner_key = f"rag:conv_owner:{conversation_id}"
    stored_owner = await redis.get(owner_key)
    
    logger.info(
        "Checking ownership", 
        conversation_id=conversation_id, 
        current_user_id=current_user.user_id, 
        stored_owner=stored_owner
    )
    
    if stored_owner:
        # Allow real user to reclaim a conversation previously owned by dev bypass
        if stored_owner == DEV_BYPASS_USER_ID and current_user.user_id != DEV_BYPASS_USER_ID:
            logger.info(
                "Migrating conversation ownership from dev_user to real user",
                conversation_id=conversation_id,
                new_owner=current_user.user_id,
            )
            await redis.set(owner_key, current_user.user_id, ex=86400 * 7)
        elif stored_owner != current_user.user_id:
            logger.error(
                "Ownership mismatch 403",
                conversation_id=conversation_id,
                current_user_id=current_user.user_id,
                stored_owner=stored_owner
            )
            raise HTTPException(status_code=403, detail="Not authorized to access this conversation")
    else:
        # No owner yet — claim atomically with SET NX to prevent TOCTOU race.
        # Two concurrent requests for the same new conversation_id could both
        # see stored_owner=None and both "claim" it; NX ensures only one wins.
        claimed = await redis.set(owner_key, current_user.user_id, nx=True, ex=86400 * 7)
        if not claimed:
            # Another request claimed it between our GET and SET — re-read to verify
            actual_owner = await redis.get(owner_key)
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
async def _prepare_rag_context(
    request: ChatRequest,
    current_user: User,
    conversation_id: str,
    resolved_query: str,
) -> dict:
    """Shared context preparation for both /chat and /chat/stream.

    Computes the query embedding ONCE and reuses it for both the semantic
    cache lookup and the LTM lookup, avoiding redundant embedding API calls.
    """
    from llama_index.core import Settings as LISettings
    from app.config.embedding_config import ensure_llamaindex_configured
    from app.graph.intent_rules import classify as _tier1_classify
    import time as _time

    _t0 = _time.perf_counter()

    # Tier-1 pre-check: skip embedding entirely for greetings/fillers.
    _tier1_intent = _tier1_classify(resolved_query)
    _skip_embedding = _tier1_intent in ("GREETING", "AMBIGUOUS")

    # Embed once — reused by cache lookup, LTM lookup, and cache write.
    query_embedding = None
    if not _skip_embedding:
        try:
            ensure_llamaindex_configured()
            query_embedding = await LISettings.embed_model.aget_query_embedding(resolved_query)
            logger.debug(f"[TIMING] embedding: {_time.perf_counter()-_t0:.2f}s")
        except Exception as exc:
            logger.warning(f"Failed to compute query embedding once: {exc}")
            query_embedding = None

    # Cache pre-filter: skip cache lookup for queries that need fresh
    # synthesis/empathy. Cache stores KNOWLEDGE-shaped answers — feeding them
    # to opinion/vent queries causes shape mismatch (e.g. user asks "menurut
    # kamu mana paling kritis" but cache returns a flat list).
    # Cheap regex check; full intent classification still happens in graph.
    import re
    _OPINION_REGEX = re.compile(
        r"\b(menurut|menurutmu|pendapat|opini|kasih saran|sarankan|advice|"
        r"what (?:do you|would you) think|"
        r"capek|stress|bingung|pusing|frustrasi|nyerah|curhat|"
        r"gimana kalau|kalau aku|what if|bantuin mikir|help me think|"
        r"mana yang|mana yg|paling penting|paling kritis|paling baik|"
        r"role[\s-]?play|anggap kamu)\b",
        re.IGNORECASE,
    )
    skip_cache = _skip_embedding or bool(_OPINION_REGEX.search(resolved_query))
    if skip_cache:
        logger.debug(
            "Cache lookup skipped — greeting/filler or opinion/synthesis pattern",
            query=resolved_query[:60],
        )

    cached = None
    if not skip_cache:
        _t_cache_start = _time.perf_counter()
        
        private_ns = f"rag_user_{current_user.user_id}"
        global_ns = "rag"
        
        private_cached, global_cached = await asyncio.gather(
            get_cached_response(
                resolved_query,
                course_id=request.course_id,
                query_embedding=query_embedding,
                cache_namespace=private_ns,
            ),
            get_cached_response(
                resolved_query,
                course_id=request.course_id,
                query_embedding=query_embedding,
                cache_namespace=global_ns,
            )
        )
        cached = private_cached or global_cached
        logger.debug(f"[TIMING] get_cached_response (private+global): {_time.perf_counter()-_t_cache_start:.2f}s")
    if cached:
        return {"cached": cached, "query_embedding": query_embedding}

    # Build message history + LTM + UserProfile in parallel.
    # These three I/O calls are independent (Redis, Qdrant, Postgres) — fan
    # them out instead of awaiting serially. Only LTM and UserProfile depend
    # on `ltm_eligible`, so they're guarded with stub coroutines that early-
    # return for non-real users (dev bypass, anonymous, etc.).
    #
    # Tradeoff: on a brand-new session of a real user we'll fetch LTM +
    # UserProfile that get discarded (since `is_brand_new_session` is only
    # known AFTER history resolves). That's 1 Qdrant call + 1 Postgres
    # SELECT on first-turn-ever — acceptable to save 100-300ms on every
    # subsequent regular turn.
    user_id = current_user.user_id
    ltm_eligible = is_real_user(user_id=user_id, role=current_user.role)

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
            },
            "query_embedding": None,
            "was_personalized": False,
        }

    async def _load_ltm_if_eligible():
        if not ltm_eligible:
            return {"summary": "", "course_names": []}
        return await qdrant_ltm.load(
            user_id=user_id,
            query=resolved_query,
            query_embedding=query_embedding,
        )

    async def _load_user_profile_if_eligible():
        if not ltm_eligible:
            return None
        async with AsyncSessionLocal() as session:
            return await session.get(UserProfile, user_id)

    logger.debug(f"[TIMING] pre-gather: {_time.perf_counter()-_t0:.2f}s")
    _t_gather_start = _time.perf_counter()
    (
        (summary, recent_history),
        ltm_profile,
        user_profile_obj,
    ) = await asyncio.gather(
        get_or_summarize_history(
            conversation_id=conversation_id,
            llm=get_cheap_llm(),
            max_fresh_turns=5,
        ),
        _load_ltm_if_eligible(),
        _load_user_profile_if_eligible(),
    )
    logger.debug(f"[TIMING] gather(history+ltm+profile): {_time.perf_counter()-_t_gather_start:.2f}s, total_so_far: {_time.perf_counter()-_t0:.2f}s")

    messages = []
    for turn in recent_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=resolved_query))

    user_pref_dict = None

    # Skip LTM lookup entirely on the first turn of a brand-new session — there
    # is nothing in `recent_history` yet AND no prior summary, which means the
    # user has never spoken to A-Pedi before in this conversation. Loading LTM
    # here just bloats the prompt by ~200 tokens and is rarely useful for the
    # very first message ("hi", "halo", "apa itu X"). We still pay the fetch
    # cost (parallel gather above) but discard the payload so the prompt
    # stays lean.
    is_brand_new_session = not recent_history and not summary
    if is_brand_new_session:
        ltm_profile = {"summary": "", "course_names": []}
        user_profile_obj = None

    if ltm_eligible and not is_brand_new_session and user_profile_obj is not None:
        # Drop stale prefs — anything older than `user_pref_max_age_days` is
        # treated as expired so a one-off "pakai bahasa awam" three months ago
        # doesn't bind every future session.
        from datetime import datetime, timedelta, timezone

        updated_at = user_profile_obj.updated_at
        is_fresh = True
        age = None
        if updated_at is not None:
            # SQLAlchemy returns naive datetime for some drivers — normalize.
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - updated_at
            is_fresh = age <= timedelta(days=settings.user_pref_max_age_days)

        if is_fresh:
            user_pref_dict = {
                "role": user_profile_obj.role,
                "preferred_tone": user_profile_obj.preferred_tone,
                "formatting_pref": user_profile_obj.formatting_pref,
                "custom_instructions": user_profile_obj.custom_instructions
            }
        else:
            logger.info(
                "User preferences ignored — older than threshold",
                user_id=user_id,
                age_days=age.days if age else None,
                threshold_days=settings.user_pref_max_age_days,
            )

    initial_state = {
        "messages": messages,
        "conversation_id": conversation_id,
        "conversation_summary": summary,
        "user_profile": ltm_profile,
        "user_preferences": user_pref_dict,
    }

    # ── Cross-user cache-leak guard (FIX 1) ───────────────────────────────────
    # The query cache is keyed by {namespace}:{course_id|global}:{sha256(query)}
    # — NOT by user_id. Skip cache write only when user-specific content was
    # actually injected into the LLM prompt:
    #   - LTM episodes: contain user's past queries/preferences → personal
    #   - UserProfile prefs: custom tone/formatting → personal
    # Conversation summary is NOT personal — it's a history of topics, not
    # injected into the KB answer. Caching KB answers even when summary exists
    # is safe because the answer content is the same for all users asking the
    # same question. This dramatically improves cache hit rate.
    _ltm = ltm_profile or {}
    has_ltm = (
        bool((_ltm.get("summary") or "").strip())
        or bool(_ltm.get("course_names"))
        or bool(_ltm.get("unanswered_questions"))
    )
    has_prefs = bool(user_pref_dict) and any(
        bool((v or "").strip()) if isinstance(v, str) else bool(v)
        for v in user_pref_dict.values()
    )
    was_personalized = has_ltm or has_prefs

    return {
        "cached": None,
        "initial_state": initial_state,
        "query_embedding": query_embedding,
        "was_personalized": was_personalized,
    }

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


@router.post("/chat", response_model=ChatResponse, summary="Ask a question using the RAG pipeline")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    # Pipeline semaphore — caps concurrent RAG work to settings.max_concurrent_pipelines.
    # Acquired before any embedding/LLM work; released as the function returns.
    # Raises HTTP 503 with Retry-After: 5 on saturation so clients back off
    # instead of queueing invisibly.
    _sem = await acquire_pipeline_slot()
    try:
        return await _run_chat(request, background_tasks, current_user, _sem)
    finally:
        _sem.release()


async def _run_chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User,
    _sem,
) -> ChatResponse:
    start_time = time.perf_counter()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    await _verify_conversation_ownership(conversation_id, current_user)

    resolved_query = await resolve_numeric_query(request.query, conversation_id)
    logger.info("Chat request received", query=request.query[:80], resolved_query=resolved_query[:80] if resolved_query != request.query else None, conversation_id=conversation_id)

    context = await _prepare_rag_context(request, current_user, conversation_id, resolved_query)
    cached = context.get("cached")
    query_embedding = context.get("query_embedding")
    was_personalized = context.get("was_personalized", False)

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
        await append_to_history(conversation_id=conversation_id, user_message=resolved_query, assistant_message=cached["answer"])

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

    initial_state = context["initial_state"]
    rag_graph = get_rag_graph()

    result = None
    answer = None
    latency_ms = 0.0

    try:
        result = await rag_graph.ainvoke(
            initial_state,
            config={"run_name": "a-pedi-chat"},
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        final_message = result["messages"][-1]
        answer = final_message.content if hasattr(final_message, "content") else str(final_message)
        llm_tokens_used = 0
        if hasattr(final_message, "response_metadata"):
            llm_tokens_used = final_message.response_metadata.get("token_usage", {}).get("total_tokens", 0)
    except Exception as exc:
        logger.error("RAG pipeline error", error=str(exc), query=request.query[:60])
        raise HTTPException(status_code=500, detail="RAG pipeline failed") from exc

    rewritten_query = result.get("rewritten_query") or resolved_query
    resolved_query = rewritten_query
    intent = result.get("intent", "KNOWLEDGE")
    # Correlation key tying this agent_logs row to its async faithfulness score.
    turn_id = str(uuid.uuid4())
    
    actual_chunks = 0
    max_chunk_score = None
    max_chunk_sparse = None
    retrieved_context = result.get("retrieved_context") or []
    
    if retrieved_context:
        real_chunks = [c for c in retrieved_context if c.get("source") not in (None, "", "None")]
        actual_chunks = len(real_chunks)
        # Gate signals mirror _route_after_rag: raw dense cosine (absolute [0,1])
        # OR raw BM25 lexical match. The fused `score` is per-query normalized
        # and can't be compared against an absolute threshold.
        dense_scores = [c.get("dense_score") for c in retrieved_context if isinstance(c.get("dense_score"), (int, float))]
        if dense_scores:
            max_chunk_score = max(dense_scores)
        sparse_scores = [c.get("sparse_score") for c in retrieved_context if isinstance(c.get("sparse_score"), (int, float))]
        if sparse_scores:
            max_chunk_sparse = max(sparse_scores)

    effective_course_id = _auto_detect_course_id(retrieved_context, request.course_id)
    sources = _extract_sources(retrieved_context)

    # Skip cache write when retrieval missed: no chunks OR both gate signals
    # below floor (mirrors _route_after_rag's dense-OR-sparse logic). These
    # responses are canned "not found" messages — caching them would poison
    # later hits when the KB grows.
    _dense_miss = max_chunk_score is None or max_chunk_score < settings.kb_min_dense_score
    _sparse_miss = max_chunk_sparse is None or max_chunk_sparse < settings.kb_min_sparse_score
    is_low_relevance = actual_chunks == 0 or (_dense_miss and _sparse_miss)
    # FIX 1: Personalized answers are now cached using a user-scoped namespace
    # so they can't leak to another user but still provide cache hits for this user.
    if (
        intent not in ("GREETING", "AMBIGUOUS", "MALICIOUS", "BRAINSTORM")
        and not is_low_relevance
    ):
        ns = f"rag_user_{current_user.user_id}" if was_personalized else "rag"
        background_tasks.add_task(
            set_cached_response,
            query=resolved_query,
            answer=answer,
            sources=sources, # we can pass dict directly since the schema validation handles it
            course_id=effective_course_id,
            query_embedding=query_embedding,
            cache_namespace=ns,
        )
    background_tasks.add_task(
        append_to_history,
        conversation_id=conversation_id,
        user_message=resolved_query,
        assistant_message=answer,
    )
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
            "cache_hit": False,
            "retrieved_context": retrieved_context,
            **_quality_log_fields(intent, result.get("intent_scores"), max_chunk_score),
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


@router.post("/chat/sync_memory/{conversation_id}", summary="Sync chat history to Long-Term Memory")
async def sync_memory(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    return {"status": "ignored", "reason": "handled_by_afk_worker_in_background"}


@router.post("/chat/stream", summary="Stream a RAG response via Server-Sent Events")
async def chat_stream(
    request: ChatRequest,
    req: Request,
    current_user: User = Depends(get_current_user),
):
    # Pipeline semaphore — held for the ENTIRE SSE stream, not just the
    # function body. The release callable is captured by both inner
    # generators (cache-hit and rag-stream) and invoked in their `finally`
    # so the permit returns when the stream ends or the client disconnects.
    # Raises HTTP 503 with Retry-After: 5 on saturation.
    sem_release = await acquire_pipeline_slot_or_503()
    start_time = time.perf_counter()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    await _verify_conversation_ownership(conversation_id, current_user)

    resolved_query = await resolve_numeric_query(request.query, conversation_id)
    logger.info("Stream request received", query=request.query[:80], resolved_query=resolved_query[:80] if resolved_query != request.query else None, conversation_id=conversation_id)

    context = await _prepare_rag_context(request, current_user, conversation_id, resolved_query)
    cached = context.get("cached")
    query_embedding = context.get("query_embedding")
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

                await append_to_history(conversation_id=conversation_id, user_message=resolved_query, assistant_message=cached["answer"])

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

    # ── Redis greeting cache check (before Tier-1) ───────────────────────────
    # Any query that was previously answered as GREETING (from LangGraph or
    # Tier-1 fast-path) is cached here with 1h TTL. This catches sapaan like
    # "assalamualaikum" that don't match Tier-1 regex but were cached after
    # first LangGraph run. Serves ~7ms from Redis, bypassing everything.
    _low_q = resolved_query.lower().strip()
    _greeting_cache_key = f"rag:greeting:{_low_q[:80]}"
    _redis_client = get_redis_client()
    _cached_greeting_resp = None
    try:
        _cached_greeting_resp = await _redis_client.get(_greeting_cache_key)
    except Exception:
        pass

    if _cached_greeting_resp:
        _cached_greeting_text = _cached_greeting_resp if isinstance(_cached_greeting_resp, str) else _cached_greeting_resp.decode()
        logger.info("Greeting Redis cache HIT", query=resolved_query[:60])

        async def _stream_greeting_cached():
            try:
                sem_release()
                latency_ms = (time.perf_counter() - start_time) * 1000
                yield f"data: {json.dumps({'token': _cached_greeting_text})}\n\n"
                yield f"event: done\ndata: {json.dumps({'sources': [], 'conversation_id': conversation_id, 'cached': True, 'latency_ms': round(latency_ms, 2)})}\n\n"
                await append_to_history(conversation_id=conversation_id, user_message=resolved_query, assistant_message=_cached_greeting_text)
                await _schedule_afk_ltm_sync(conversation_id, current_user.user_id)
            finally:
                pass

        return StreamingResponse(_stream_greeting_cached(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ── Tier-1 fast-path: bypass LangGraph entirely for GREETING/AMBIGUOUS ──
    # LangGraph astream_events has significant overhead (~15-50s) even for
    # hardcoded responses. For Tier-1 intents we already know the response —
    # stream it directly without touching the graph.
    from app.graph.intent_rules import classify as _t1_classify, _is_greeting as _is_pure_greeting, _is_identity_question
    _t1_intent = _t1_classify(resolved_query)
    if _t1_intent in ("GREETING", "AMBIGUOUS"):
        low_q = _low_q

        # Determine fast reply
        if _t1_intent == "GREETING":
            if _is_pure_greeting(low_q):
                if any(c in low_q for c in ("halo", "hai", "hei", "pagi", "siang", "sore", "malam", "selamat", "permisi", "assalam", "waalaikum")):
                    _fast_reply = "Halo! Ada yang bisa aku bantu seputar materi Amarthapedia?"
                else:
                    _fast_reply = "Hi! Anything I can help with from Amarthapedia?"
            elif _is_identity_question(low_q):
                if any(c in low_q for c in ("kamu", "lu", "lo", "ini apps", "ini aplikasi", "perkenalkan")):
                    _fast_reply = "Aku A-Pedi, asisten AI di Amarthapedia — LMS internal Amartha untuk karyawan. Bisa bantu cari info dari materi training soal produk, kebijakan, atau topik lain di Amarthapedia. Mau tanya soal apa?"
                else:
                    _fast_reply = "I'm A-Pedi, the AI assistant for Amarthapedia — Amartha's internal LMS for employees. What would you like to know?"
            else:
                _fast_reply = "Halo! Ada yang bisa aku bantu seputar materi Amarthapedia?"
        else:  # AMBIGUOUS / pure filler
            if any(ord(c) > 127 for c in resolved_query):
                _fast_reply = "Ada yang bisa aku bantu? Boleh sebut topiknya ya."
            else:
                _fast_reply = "Anything I can help with? Feel free to name a topic."

        # Cache this greeting response for 1 hour
        try:
            await _redis_client.set(_greeting_cache_key, _fast_reply, ex=3600)
        except Exception:
            pass

        async def _stream_tier1():
            try:
                sem_release()
                latency_ms = (time.perf_counter() - start_time) * 1000
                yield f"data: {json.dumps({'token': _fast_reply})}\n\n"
                yield f"event: done\ndata: {json.dumps({'sources': [], 'conversation_id': conversation_id, 'cached': False, 'latency_ms': round(latency_ms, 2)})}\n\n"
                await append_to_history(conversation_id=conversation_id, user_message=resolved_query, assistant_message=_fast_reply)
                await _schedule_afk_ltm_sync(conversation_id, current_user.user_id)
            finally:
                pass

        logger.info(f"Tier-1 fast-path bypass LangGraph: {_t1_intent}", query=resolved_query[:60])
        return StreamingResponse(_stream_tier1(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    initial_state = context["initial_state"]
    rag_graph = get_rag_graph()

    async def _stream_rag():
        nonlocal resolved_query
        from app.graph.pipeline import StreamLeakGuard, _sanitize_answer
        full_answer = ""
        retrieved_context = []
        intent = "KNOWLEDGE"
        stream_intent_scores: dict = {}
        turn_id = str(uuid.uuid4())  # correlates this turn's agent_logs row to its async eval score
        leak_guard = StreamLeakGuard()

        try:
            config = {"run_name": "a-pedi-chat-stream"}

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

            async for event in rag_graph.astream_events(initial_state, config=config, version="v2"):
                kind = event.get("event", "")

                if kind == "on_chain_end" and event.get("name") == "rag_node":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and "retrieved_context" in output:
                        retrieved_context = output["retrieved_context"] or []

                if kind == "on_chain_end" and event.get("name") == "pre_processor":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        if "intent" in output:
                            intent = output.get("intent")
                        if "intent_scores" in output:
                            stream_intent_scores = output.get("intent_scores") or {}
                        if "rewritten_query" in output:
                            new_rewrite = output.get("rewritten_query")
                            if new_rewrite and new_rewrite != resolved_query:
                                resolved_query = new_rewrite
                                yield f"event: resolved\ndata: {json.dumps({'resolved_query': resolved_query})}\n\n"

                # Canned-response nodes return an AIMessage directly without
                # invoking an LLM, so no `on_chat_model_stream` event fires
                # for them. Emit their content from on_chain_end instead.
                # Covers: greeting/ambiguity (Tier-2 hardcoded, no LLM),
                # malicious/topic_list/low_relevance/off_scope (deterministic
                # canned replies). Skip if LLM streaming already emitted
                # tokens for this node (e.g. "hmm" → LLM ambiguity).
                if (
                    kind == "on_chain_end"
                    and event.get("name") in (
                        "greeting", "ambiguity",
                        "malicious", "topic_list", "low_relevance", "off_scope",
                    )
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
                            canned_emitted = True

                if kind == "on_chat_model_stream":
                    node_name = event.get("metadata", {}).get("langgraph_node")
                    if node_name in ("generate_node", "greeting", "ambiguity"):
                        streamed_nodes.add(node_name)
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            token = chunk.content
                            full_answer += token
                            token_count += 1

                            # Pass through leak guard. Greeting/ambiguity
                            # don't carry retrieved_context, but the guard
                            # is a no-op on clean preambles so it's safe.
                            safe = leak_guard.feed(token)
                            if safe:
                                yield f"data: {json.dumps({'token': safe})}\n\n"

                            # Periodic disconnect check — bail out early if user closed the tab.
                            if token_count % 10 == 0 and await req.is_disconnected():
                                logger.info("Client disconnected mid-stream", conversation_id=conversation_id, tokens=token_count)
                                return

        except Exception as exc:
            logger.error("Stream pipeline error", error=str(exc), query=request.query[:60])
            yield f"event: error\ndata: {json.dumps({'error': 'RAG pipeline failed'})}\n\n"
            # Still append partial history so the turn isn't silently lost.
            # full_answer may be empty/partial — that's acceptable for error turns.
            if resolved_query:
                try:
                    await append_to_history(
                        conversation_id=conversation_id,
                        user_message=resolved_query,
                        assistant_message="[error: pipeline failed]",
                    )
                except Exception:
                    pass
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

        yield f"event: done\ndata: {json.dumps({'sources': sources, 'conversation_id': conversation_id, 'cached': False, 'latency_ms': round(latency_ms, 2)})}\n\n"

        # ── Greeting Redis cache write ─────────────────────────────────────
        # Cache GREETING responses so repeated sapaan ("assalamualaikum",
        # "selamat pagi", etc.) from 13k users are served instantly next time
        # without going through LangGraph again. TTL 1h — greeting responses
        # are static and don't need freshness guarantees.
        if intent == "GREETING" and full_answer and len(full_answer) < 500:
            try:
                _greeting_key = f"rag:greeting:{resolved_query.lower().strip()[:80]}"
                await get_redis_client().set(_greeting_key, full_answer, ex=3600)
                logger.debug("Greeting response cached", query=resolved_query[:60])
            except Exception:
                pass

        try:
            effective_course_id = _auto_detect_course_id(retrieved_context, request.course_id)
            stream_dense_scores = [c.get("dense_score") for c in retrieved_context if isinstance(c.get("dense_score"), (int, float))]
            stream_max_score = max(stream_dense_scores) if stream_dense_scores else None
            stream_sparse_scores = [c.get("sparse_score") for c in retrieved_context if isinstance(c.get("sparse_score"), (int, float))]
            stream_max_sparse = max(stream_sparse_scores) if stream_sparse_scores else None
            # Mirror _route_after_rag: low-relevance only when BOTH signals miss.
            _s_dense_miss = stream_max_score is None or stream_max_score < settings.kb_min_dense_score
            _s_sparse_miss = stream_max_sparse is None or stream_max_sparse < settings.kb_min_sparse_score
            is_low_relevance_stream = (not retrieved_context) or (_s_dense_miss and _s_sparse_miss)
            if (
                intent not in ("GREETING", "AMBIGUOUS", "MALICIOUS", "TOPIC_LIST", "BRAINSTORM")
                and not is_low_relevance_stream
            ):
                ns = f"rag_user_{current_user.user_id}" if was_personalized else "rag"
                await set_cached_response(
                    query=resolved_query,
                    answer=full_answer,
                    sources=sources,
                    course_id=effective_course_id,
                    query_embedding=query_embedding,
                    cache_namespace=ns,
                )
            await append_to_history(conversation_id=conversation_id, user_message=resolved_query, assistant_message=full_answer)
            await batch_logger.add_log({
                "turn_id": turn_id,
                "endpoint": "chat-stream",
                "conversation_id": conversation_id,
                "query": request.query,
                "rewritten_query": resolved_query,
                "answer": full_answer,
                "chunks_retrieved": len(retrieved_context),
                "latency_ms": round(latency_ms, 2),
                "llm_tokens_used": token_count,
                "cache_hit": False,
                "retrieved_context": retrieved_context,
                **_quality_log_fields(intent, stream_intent_scores, stream_max_score),
            })
            await _schedule_afk_ltm_sync(conversation_id, current_user.user_id)
            await _track_session_courses(conversation_id, retrieved_context)

            if _should_eval_turn(
                intent=intent,
                intent_scores=stream_intent_scores,
                max_dense_score=stream_max_score,
                answer=full_answer,
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
            # Always release the pipeline permit at end of stream (normal
            # completion OR client disconnect OR exception in the generator).
            sem_release()

    return StreamingResponse(_stream_rag(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
