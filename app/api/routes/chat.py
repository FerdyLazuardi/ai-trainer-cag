import asyncio
from collections import Counter
import json
import re
import time
from typing import Any, Optional
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger

from app.agents import conversation_state as _cs
from app.api.auth import User, get_current_user
from app.api.concurrency import acquire_pipeline_slot, acquire_pipeline_slot_or_503
from app.api.routes.chat_helpers import (
    StreamTextDeduper,
    _AFFIRMATION_TOKENS,
    _EVAL_SKIP_INTENTS,
    _auto_detect_course_id,
    _build_enriched_user_context,
    _enqueue_eval,
    _extract_sources,
    _is_bare_affirmation,
    _is_transient_stream_error,
    _quality_log_fields,
    _serialize_gate_score,
    _should_eval_turn,
    _should_write_response_cache,
    _user_log_fields,
    cache_namespace_for,
    compute_was_personalized,
)
from app.api.schemas import ChatRequest, ChatResponse, SourceReference
from app.api.user_utils import is_real_user
from app.config.settings import get_settings
from app.database.models import UserLTMMemory
from app.database.postgres import AsyncSessionLocal
from app.database.redis_client import get_redis_client
from app.llm.cag_client import (
    OpenRouterUsage,
    extract_openrouter_usage,
    fetch_openrouter_generation_usage,
)
from app.utils.cache import get_cached_response, set_cached_response
from app.utils.logger_batch import batch_logger

router = APIRouter()
settings = get_settings()

_META_CONVO_RE = re.compile(
    r"(?:udah|sudah|udh|tadi|barusan|kita|kami)\b[^.?!\n]{0,30}"
    r"(?:bahas|dibahas|ngomong|omongin|diskusi|obrol)"
    r"|(?:yang|apa)\b[^.?!\n]{0,20}(?:tadi|barusan|kita|kami|sebelumnya)\s+(?:di)?(?:bahas|omongin|diskusi)"
    r"|itu aja[^.?!\n]{0,25}(?:bahas|omongin)"
    r"|what (?:did|have|were) we (?:discuss|talk|cover|go over|chat)",
    re.IGNORECASE,
)

_OPINION_REGEX = re.compile(
    r"\b(menurut|menurutmu|pendapat|opini|kasih saran|sarankan|advice|"
    r"what (?:do you|would you) think|"
    r"capek?|cape lah|lelah|males|stress|bingung|pusing|frustrasi|nyerah|curhat|"
    r"gimana kalau|kalau aku|what if|bantuin mikir|help me think|"
    r"mana yang|mana yg|paling penting|paling kritis|paling baik|"
    r"role[\s-]?play|anggap kamu)\b",
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


_stream_bg_tasks: set[asyncio.Task] = set()


# ── STM facade ──────────────────────────────────────────────────────────────
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
    try:
        await _cs.schedule_summary_refresh(
            get_redis_client(), conv_id, max_fresh_turns=settings.max_fresh_turns
        )
    except Exception:
        pass


async def _schedule_afk_ltm_sync(conv_id: str, u_id: str):
    await _cs.schedule_afk_sync(get_redis_client(), conv_id, u_id)


async def _track_session_courses(conv_id: str, retrieved_context: list) -> None:
    if not retrieved_context:
        return
    names = {
        (c.get("course_name") or "").strip()
        for c in retrieved_context
        if (c.get("course_name") or "").strip() not in ("", "?", "Unknown")
    }
    if not names:
        return
    try:
        await _cs.add_courses(get_redis_client(), conv_id, names)
    except Exception as e:
        logger.warning(f"Failed to track session courses: {e}")


DEV_BYPASS_USER_ID = "dev_user_123"


async def _verify_conversation_ownership(conversation_id: str, current_user: User):
    """Ensure the user owns this conversation before accessing history."""
    redis = get_redis_client()
    conv_key = _cs._conv_key(conversation_id)
    ttl = settings.conversation_ttl_seconds

    stored_owner = await redis.hget(conv_key, "owner")
    if not stored_owner:
        legacy_owner = await redis.get(f"rag:conv_owner:{conversation_id}")
        if legacy_owner:
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
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hsetnx(conv_key, "owner", current_user.user_id)
            pipe.expire(conv_key, ttl)
            results = await pipe.execute()
        if not results[0]:
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
    """Shared context preparation for both /chat and /chat/stream."""
    from app.graph.intent_rules import classify as _tier1_classify

    _t0 = time.perf_counter()
    _tier1_intent = _tier1_classify(resolved_query)
    _skip_embedding = _tier1_intent in ("GREETING", "AMBIGUOUS")

    skip_cache = (
        _skip_embedding
        or _tier1_intent == "TOPIC_LIST"
        or request.coaching_mode
        or _is_bare_affirmation(resolved_query)
        or bool(_OPINION_REGEX.search(resolved_query))
        or bool(_META_CONVO_RE.search(resolved_query))
    )
    if skip_cache:
        logger.debug("Cache lookup skipped", query=resolved_query[:60])

    user_id = current_user.user_id
    ltm_eligible = is_real_user(user_id=user_id, role=current_user.role)
    user_context = await _build_enriched_user_context(current_user)

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

    logger.debug(f"[TIMING] pre-history: {time.perf_counter()-_t0:.2f}s")
    _t_hist = time.perf_counter()
    summary, recent_history = await get_or_summarize_history(
        conversation_id=conversation_id,
        llm=get_cheap_llm(),
        max_fresh_turns=settings.max_fresh_turns,
        persist=False,
    )
    asyncio.create_task(_schedule_summary_refresh(conversation_id))

    seen_chunk_ids = await get_seen_chunk_ids(conversation_id)
    logger.debug(f"[TIMING] history: {time.perf_counter()-_t_hist:.2f}s")
    if (recent_history or summary) and len(resolved_query.split()) <= 3:
        skip_cache = True
        logger.debug("Cache lookup skipped - short follow-up query (context-dependent)")

    cached = None
    if not skip_cache:
        _t_cache_start = time.perf_counter()
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
            ),
        )
        cached = private_cached or global_cached
        logger.debug(f"[TIMING] get_cached_response (private+global): {time.perf_counter()-_t_cache_start:.2f}s")

    if cached:
        return {
            "cached": cached,
            "query_embedding": None,
            "initial_state": {"user_context": user_context},
            "user_context": user_context,
            "was_personalized": False,
            "skip_cache": skip_cache,
        }

    messages: list[BaseMessage] = [
        HumanMessage(content=turn["content"]) if turn["role"] == "user" else AIMessage(content=turn["content"])
        for turn in recent_history
    ]
    messages.append(HumanMessage(content=resolved_query))

    ltm_profile = {"learning_summary": ""}
    if ltm_eligible:
        async with AsyncSessionLocal() as session:
            user_profile_obj = await session.get(UserLTMMemory, user_id)
        if user_profile_obj is not None:
            ltm_profile = {"learning_summary": user_profile_obj.learning_summary or ""}

    initial_state = {
        "messages": messages,
        "conversation_id": conversation_id,
        "conversation_summary": summary,
        "user_profile": ltm_profile,
        "user_preferences": None,
        "user_context": user_context,
        "query_embedding": None,
        "query_embedding_text": None,
        "rewritten_queries": None,
        "retrieval_query": resolved_query,
        "seen_chunk_ids": list(seen_chunk_ids),
        "coaching_mode": request.coaching_mode,
    }

    was_personalized = compute_was_personalized(
        ltm_profile=ltm_profile,
        recent_history=recent_history,
        summary=summary,
        user_context=user_context,
    )

    return {
        "cached": None,
        "initial_state": initial_state,
        "query_embedding": None,
        "was_personalized": was_personalized,
        "skip_cache": skip_cache,
    }


_prepare_rag_context = _prepare_cag_context


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
        return max(ttl, 0) if ttl > 0 else 0
    except Exception as exc:
        logger.warning(f"Failed to check user ban TTL: {exc}")
        return 0


async def _handle_off_scope_violation(user_id: str) -> tuple[int, bool]:
    """Increment violation count for the user. Returns (new_count, just_banned)."""
    try:
        redis_client = get_redis_client()
        cfg = get_settings()
        key = f"ava:user:{user_id}:off_scope_violations"
        new_count = await redis_client.incr(key)
        await redis_client.expire(key, cfg.off_scope_violation_window_seconds)

        just_banned = False
        if new_count >= cfg.max_off_scope_violations:
            await redis_client.setex(
                f"ava:user:{user_id}:banned",
                cfg.off_scope_ban_duration_seconds,
                "1",
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
    logger.info(
        "Chat request received",
        query=request.query[:80],
        resolved_query=resolved_query[:80] if resolved_query != request.query else None,
        conversation_id=conversation_id,
    )

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
                **_user_log_fields(current_user, context),
            },
        )
        await append_to_history(conversation_id=conversation_id, user_message=request.query, assistant_message=cached["answer"])
        background_tasks.add_task(_schedule_afk_ltm_sync, conversation_id, current_user.user_id)

        return ChatResponse(
            answer=cached["answer"],
            sources=[SourceReference(**s) for s in cached["sources"]],
            conversation_id=conversation_id,
            resolved_query=resolved_query if resolved_query != request.query else None,
            cached=True,
            latency_ms=round(latency_ms, 2),
        )

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
    cfg: Any,
    answer: str,
) -> tuple[str, int]:
    off_scope_ban_ttl: int = 0
    if off_scope_detected:
        new_count, _ = await _handle_off_scope_violation(user_id)
        if new_count == cfg.warning_off_scope_violations:
            warning_text = "\n\n**Peringatan**: Harap tanyakan hal seputar materi Amartha. Jika Anda terus bertanya di luar topik sekali lagi, Ai Trainer akan dinonaktifkan sementara."
            answer += warning_text
        elif new_count >= cfg.max_off_scope_violations:
            off_scope_ban_ttl = await _get_ban_ttl(user_id)
    return answer, off_scope_ban_ttl


async def _execute_chat_flow(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User,
    context: dict,
    resolved_query: str,
    start_time: float,
    conversation_id: str,
) -> ChatResponse:
    cfg = get_settings()
    initial_state = context["initial_state"]
    was_personalized = context.get("was_personalized", False)
    cag_graph = get_cag_graph()

    try:
        result = await asyncio.wait_for(
            cag_graph.ainvoke(initial_state, config={"run_name": "ava-chat"}),
            timeout=cfg.pipeline_total_timeout_s,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000
        final_message = result["messages"][-1]
        from app.graph.pipeline import _sanitize_answer
        raw_answer = final_message.content if hasattr(final_message, "content") else str(final_message)
        sanitized = _sanitize_answer(raw_answer)

        off_scope_detected = result.get("off_scope_detected", False)
        answer, off_scope_ban_ttl = await _process_off_scope_status(
            off_scope_detected, current_user.user_id, cfg, sanitized
        )
        usage_data = await _extract_message_token_usage(final_message)

    except asyncio.TimeoutError as exc:
        logger.error(
            "Chat pipeline timed out",
            timeout_s=cfg.pipeline_total_timeout_s,
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

    if _should_write_response_cache(
        intent=intent,
        is_low_relevance=is_low_relevance,
        query=request.query,
        skip_cache=context.get("skip_cache", False),
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
            "llm_tokens_used": usage_data["llm_tokens_used"],
            "or_prompt_tokens": usage_data["or_prompt_tokens"],
            "or_cached_tokens": usage_data["or_cached_tokens"],
            "or_completion_tokens": usage_data["or_completion_tokens"],
            "or_provider": usage_data["or_provider"],
            "or_generation_id": usage_data["or_generation_id"],
            "or_cost": usage_data["or_cost"],
            "cache_hit": False,
            "retrieved_context": retrieved_context,
            **_user_log_fields(current_user, context),
            **_quality_log_fields(
                intent,
                result.get("intent_scores"),
                max_chunk_score,
                _serialize_gate_score(result.get("gate_score")),
            ),
        },
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
    if not is_real_user(current_user.user_id, current_user.role):
        return {"completed": False}
    async with AsyncSessionLocal() as session:
        profile = await session.get(UserLTMMemory, current_user.user_id)
    return {"completed": bool(profile and profile.onboarding_completed_at is not None)}


@router.post("/user/onboarding/complete", summary="Mark the onboarding tour as seen")
async def complete_onboarding(
    current_user: User = Depends(get_current_user),
) -> dict:
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
    from app.graph.pipeline import _load_course_names, _load_h2_topics, resolve_user_role
    try:
        topics = await _load_course_names()
        user_ctx = {
            "location": current_user.location if current_user else "",
            "grade": current_user.grade if current_user else "",
        }
        h2_topics = await _load_h2_topics(resolve_user_role(user_ctx))
    except Exception as exc:
        logger.warning(f"/chat/topics load failed: {exc}")
        topics = []
        h2_topics = []
    return {"topics": topics, "h2_topics": h2_topics}


@router.get("/chat/sections", summary="List Moodle sections and their items (instant, no LLM)")
async def list_sections(
    current_user: Optional[User] = Depends(get_current_user),
) -> dict:
    from app.graph.pipeline import _load_h2_topics, _load_section_map, resolve_user_role
    try:
        user_ctx = {
            "location": current_user.location if current_user else "",
            "grade": current_user.grade if current_user else "",
        }
        resolved_role = resolve_user_role(user_ctx)
        sections = await _load_section_map(resolved_role)
        h2_topics = await _load_h2_topics(resolved_role)
    except Exception as exc:
        logger.warning(f"/chat/sections load failed: {exc}")
        sections = {}
        h2_topics = []
    return {"sections": sections, "h2_topics": h2_topics}


@router.post("/chat/sync_memory/{conversation_id}", summary="Sync chat history to Long-Term Memory")
async def sync_memory(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
):
    return {"status": "ignored", "reason": "handled_by_afk_worker_in_background"}


@router.post("/chat/stream", summary="Stream a CAG response via Server-Sent Events")
async def chat_stream(
    request: ChatRequest,
    req: Request,
    current_user: User = Depends(get_current_user),
):
    sem_release = await acquire_pipeline_slot_or_503()
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
        _raw_query_for_cache = resolved_query
        logger.info(
            "Stream request received",
            query=request.query[:80],
            resolved_query=resolved_query[:80] if resolved_query != request.query else None,
            conversation_id=conversation_id,
        )

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
                try:
                    await _schedule_afk_ltm_sync(conversation_id, current_user.user_id)
                except Exception as e:
                    logger.warning(f"Cache-hit AFK LTM schedule failed: {e}")
            finally:
                pass

        return StreamingResponse(
            _stream_cached(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
        deduper = StreamTextDeduper()
        token_count = 0
        answer_emitted = False
        stream_max_score = None
        is_low_relevance_stream = False
        _logged = False

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
                    "answer": deduper.full_answer,
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
                    **_user_log_fields(current_user, context),
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
                    deduper.full_answer += content
                    answer_emitted = True
                    yield f"data: {json.dumps({'token': content})}\n\n"
            else:
                _ping_s = 2.0
                _stall_s = settings.pipeline_stream_stall_timeout_s
                _max_retries = 1
                _attempt = 0

                while True:
                    try:
                        _gen = stream_openrouter_generate(initial_state, config=config)
                        _last_real = time.monotonic()
                        _pending = None
                        while True:
                            if _pending is None:
                                _pending = asyncio.create_task(_gen.__anext__())
                            _done, _ = await asyncio.wait([_pending], timeout=_ping_s)
                            if _pending in _done:
                                try:
                                    ev = _pending.result()
                                except StopAsyncIteration:
                                    _pending = None
                                    break
                                except BaseException:
                                    _pending = None
                                    raise
                                _pending = None
                                _last_real = time.monotonic()
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
                                emit = deduper.feed(ev.get("text") or "")
                                if not emit:
                                    continue
                                token_count += 1
                                safe = leak_guard.feed(emit)
                                if safe:
                                    safe = re.sub(r"[ \t]*[—–][ \t]*", ", ", safe)
                                    answer_emitted = True
                                    yield f"data: {json.dumps({'token': safe})}\n\n"
                                if token_count % 5 == 0 and await req.is_disconnected():
                                    logger.info("Client disconnected mid-stream", conversation_id=conversation_id, tokens=token_count)
                                    deduper.full_answer = ""
                                    await _emit_log()
                                    return
                            else:
                                if time.monotonic() - _last_real < _stall_s:
                                    yield ": ping\n\n"
                                    continue
                                logger.error(
                                    "OpenRouter stream stalled — no events within stall window; freeing slot",
                                    stall_s=_stall_s,
                                    conversation_id=conversation_id,
                                    tokens_so_far=token_count,
                                )
                                if _pending and not _pending.done():
                                    _pending.cancel()
                                raise asyncio.TimeoutError("openrouter stream stalled")
                        break
                    except BaseException as _stream_exc:
                        if (
                            _is_transient_stream_error(_stream_exc)
                            and _attempt < _max_retries
                            and token_count == 0
                            and not deduper.full_answer.strip()
                        ):
                            _attempt += 1
                            logger.warning(
                                f"Transient OpenRouter stream error, retry {_attempt}/{_max_retries}: {type(_stream_exc).__name__}: {_stream_exc}"
                            )
                            await asyncio.sleep(0.5 * _attempt)
                            continue
                        raise

            tail = leak_guard.flush()
            if tail:
                answer_emitted = True
                yield f"data: {json.dumps({'token': tail})}\n\n"
            if leak_guard.leak_detected:
                deduper.full_answer = tail

            cleaned_answer = _sanitize_answer(deduper.full_answer)
            if cleaned_answer != deduper.full_answer:
                deduper.full_answer = cleaned_answer
            if "[OFFSCOPE]" in deduper.full_answer.upper():
                stream_off_scope_detected = True

            if not answer_emitted and not deduper.full_answer.strip():
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
                            deduper.full_answer = _sanitize_answer(fb_content)
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
                            yield f"data: {json.dumps({'token': deduper.full_answer})}\n\n"
                except Exception as fb_exc:
                    logger.warning(f"Fallback ainvoke failed after empty raw stream: {type(fb_exc).__name__}: {fb_exc}")

            if not deduper.full_answer.strip():
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
            coaching_done = request.coaching_mode and intent == "COACHING" and "?" not in deduper.full_answer
            stream_new_ban_ttl = 0
            if stream_off_scope_detected:
                new_count, _ = await _handle_off_scope_violation(current_user.user_id)
                if new_count == settings.warning_off_scope_violations:
                    warning_text = "\n\n**Peringatan**: Harap tanyakan hal seputar materi Amartha. Jika Anda terus bertanya di luar topik sekali lagi, Ai Trainer akan dinonaktifkan sementara."
                    yield f"data: {json.dumps({'token': warning_text})}\n\n"
                    deduper.full_answer += warning_text
                elif new_count >= settings.max_off_scope_violations:
                    stream_new_ban_ttl = await _get_ban_ttl(current_user.user_id)

            try:
                await append_to_history(
                    conversation_id=conversation_id,
                    user_message=request.query,
                    assistant_message=deduper.full_answer,
                )
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
                if _should_write_response_cache(
                    intent=intent,
                    is_low_relevance=is_low_relevance_stream,
                    query=request.query,
                    skip_cache=context.get("skip_cache", False),
                ):
                    ns = cache_namespace_for(was_personalized=was_personalized, user_id=current_user.user_id)
                    await set_cached_response(
                        query=_raw_query_for_cache,
                        answer=deduper.full_answer,
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
                    answer=deduper.full_answer,
                    is_low_relevance=is_low_relevance_stream,
                ):
                    await _enqueue_eval(
                        turn_id=turn_id,
                        query=resolved_query,
                        answer=deduper.full_answer,
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

    return StreamingResponse(
        _stream_cag_raw(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
