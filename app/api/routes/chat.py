import asyncio
import json
import time
import uuid
import datetime
from typing import Optional
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

# Moved inline imports to file-level
from langchain_core.messages import HumanMessage, AIMessage
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from app.agents.memory import (
    append_to_history,
    get_conversation_history,
    resolve_numeric_query,
    get_or_summarize_history,
    clear_conversation_history
)
from app.api.schemas import ChatRequest, ChatResponse, SourceReference
from app.database.postgres import get_db
from app.database.models import UserProfile
from app.graph.pipeline import get_rag_graph
from app.utils.cache import get_cached_response, set_cached_response
from app.config.settings import get_settings
from app.observability import get_langfuse_client
from app.utils.logger_batch import batch_logger
from app.api.auth import get_current_user, User
from app.llm.client import get_cheap_llm
from app.api.user_utils import is_real_user
from app.agents.long_term_memory_qdrant import qdrant_ltm
from app.database.redis_client import get_redis_client
from app.api.routes.ingest import get_arq_redis

router = APIRouter()
settings = get_settings()

def _flush_langfuse():
    """Background: flush Langfuse so traces are sent immediately."""
    lf = get_langfuse_client()
    if lf:
        try:
            lf.flush()
        except Exception as e:
            logger.warning(f"Langfuse flush error: {e}")

async def _schedule_afk_ltm_sync(conv_id: str, u_id: str):
    redis_client = get_redis_client()
    # Update last_active timestamp on every message (rolling window)
    await redis_client.set(f"rag:last_active:{conv_id}", str(time.time()), ex=86400)

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
            _defer_by=datetime.timedelta(seconds=10)
        )
        # Mark as scheduled (TTL slightly > 10s so key covers the defer window)
        await redis_client.set(sched_key, "1", ex=60)
        logger.debug("AFK LTM sync scheduled", conversation_id=conv_id)
    except Exception as e:
        logger.warning(f"Failed to schedule AFK LTM sync: {e}")

DEV_BYPASS_USER_ID = "dev_user_123"

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
        # No owner yet — claim it
        logger.info(
            "Claiming conversation ownership",
            conversation_id=conversation_id,
            new_owner=current_user.user_id
        )
        await redis.set(owner_key, current_user.user_id, ex=86400 * 7)
async def _prepare_rag_context(
    request: ChatRequest, 
    current_user: User, 
    conversation_id: str,
    resolved_query: str,
    db: AsyncSession
) -> dict:
    """Shared context preparation for both /chat and /chat/stream."""
    
    # Check cache first
    cached = await get_cached_response(resolved_query, course_id=request.course_id)
    if cached:
        return {"cached": cached}

    # Build message history
    summary, recent_history = await get_or_summarize_history(
        conversation_id=conversation_id,
        llm=get_cheap_llm(),
        max_fresh_turns=5,
    )

    messages = []
    for turn in recent_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=resolved_query))

    # Long-Term Memory (LTM)
    user_id = current_user.user_id
    ltm_eligible = is_real_user(user_id=user_id, role=current_user.role)
    ltm_profile = {"summary": "", "course_names": []}
    user_pref_dict = None

    if ltm_eligible:
        ltm_profile = await qdrant_ltm.load(user_id=user_id, query=resolved_query)
        # Fetch persistent preferences
        user_profile_obj = await db.get(UserProfile, user_id)
        if user_profile_obj:
            user_pref_dict = {
                "role": user_profile_obj.role,
                "preferred_tone": user_profile_obj.preferred_tone,
                "formatting_pref": user_profile_obj.formatting_pref,
                "custom_instructions": user_profile_obj.custom_instructions
            }

    initial_state = {
        "messages": messages,
        "conversation_id": conversation_id,
        "conversation_summary": summary,
        "user_profile": ltm_profile,
        "user_preferences": user_pref_dict,
    }
    
    return {"cached": None, "initial_state": initial_state}

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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    start_time = time.perf_counter()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    await _verify_conversation_ownership(conversation_id, current_user)

    resolved_query = await resolve_numeric_query(request.query, conversation_id)
    logger.info("Chat request received", query=request.query[:80], resolved_query=resolved_query[:80] if resolved_query != request.query else None, conversation_id=conversation_id)

    context = await _prepare_rag_context(request, current_user, conversation_id, resolved_query, db)
    cached = context.get("cached")

    if cached:
        latency_ms = (time.perf_counter() - start_time) * 1000
        try:
            langfuse = get_client()
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="peped-chat",
                input={"query": request.query, "resolved_query": resolved_query, "conversation_id": conversation_id},
            ) as root_obs:
                with propagate_attributes(
                    trace_name="peped-chat",
                    session_id=conversation_id,
                    user_id=f"{current_user.username} (ID: {current_user.user_id})",
                    tags=["api-chat", settings.app_env, "cache-hit"],
                ):
                    root_obs.update(
                        output=cached["answer"],
                        usage={"input": 0, "output": 0},
                        metadata={"latency_ms": round(latency_ms, 2), "cache_hit": True, "env": settings.app_env},
                    )
        except Exception as e:
            logger.warning(f"Langfuse cache-hit trace failed: {e}")

        background_tasks.add_task(_flush_langfuse)
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

    langfuse_handler = None
    trace_id = None
    result = None
    answer = None
    latency_ms = 0.0

    try:
        langfuse = get_client()

        with propagate_attributes(
            trace_name="peped-chat",
            session_id=conversation_id,
            user_id=f"{current_user.username} (ID: {current_user.user_id})",
            tags=["api-chat", settings.app_env],
            metadata={
                "cache_hit": False,
                "env": settings.app_env,
                "original_query": request.query,
                "resolved_query": resolved_query,
            }
        ):
            langfuse_handler = CallbackHandler()
            result = await rag_graph.ainvoke(
                initial_state,
                config={"run_name": "peped-chat", "callbacks": [langfuse_handler]},
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            final_message = result["messages"][-1]
            answer = final_message.content if hasattr(final_message, "content") else str(final_message)

            langfuse.set_current_trace_io(
                input={"query": request.query, "resolved_query": resolved_query, "conversation_id": conversation_id},
                output={"answer": answer}
            )

        if hasattr(langfuse_handler, "last_trace_id"):
            trace_id = langfuse_handler.last_trace_id

    except Exception as exc:
        logger.error("RAG pipeline error", error=str(exc), query=request.query[:60])
        background_tasks.add_task(_flush_langfuse)
        raise HTTPException(status_code=500, detail="RAG pipeline failed") from exc

    rewritten_query = result.get("rewritten_query") or resolved_query
    resolved_query = rewritten_query
    intent = result.get("intent", "KNOWLEDGE")
    
    actual_chunks = 0
    max_chunk_score = None
    retrieved_context = result.get("retrieved_context") or []
    
    if retrieved_context:
        real_chunks = [c for c in retrieved_context if c.get("source") not in (None, "", "None")]
        actual_chunks = len(real_chunks) if real_chunks else len(retrieved_context)
        scores = [c.get("score") for c in retrieved_context if isinstance(c.get("score"), (int, float))]
        if scores:
            max_chunk_score = max(scores)

    effective_course_id = _auto_detect_course_id(retrieved_context, request.course_id)
    sources = _extract_sources(retrieved_context)

    if trace_id and retrieved_context:
        try:
            from langfuse import get_client as _get_lf
            _lf = _get_lf()

            # ── Hybrid scores (pre-Cohere, raw LlamaIndex dense+BM25) ──
            hybrid_scores = [c.get("hybrid_score") for c in retrieved_context if isinstance(c.get("hybrid_score"), (int, float))]
            # ── Cohere scores (post-rerank) — same as final .score ──
            cohere_scores  = [c.get("cohere_score") for c in retrieved_context if isinstance(c.get("cohere_score"), (int, float))]

            if hybrid_scores:
                _lf.score(trace_id=trace_id, name="retriever_hybrid_max",  value=round(max(hybrid_scores), 4))
                _lf.score(trace_id=trace_id, name="retriever_hybrid_avg",  value=round(sum(hybrid_scores) / len(hybrid_scores), 4))
            if cohere_scores:
                _lf.score(trace_id=trace_id, name="retriever_cohere_max",  value=round(max(cohere_scores), 4))
                _lf.score(trace_id=trace_id, name="retriever_cohere_avg",  value=round(sum(cohere_scores) / len(cohere_scores), 4))
            if max_chunk_score is not None:
                _lf.score(trace_id=trace_id, name="retriever_max_score",   value=float(max_chunk_score))
        except Exception as e:
            logger.warning(f"Failed to submit Langfuse retrieval scores: {e}")

    background_tasks.add_task(_flush_langfuse)
    if intent not in ("GREETING", "AMBIGUOUS", "MALICIOUS"):
        background_tasks.add_task(
            set_cached_response,
            query=resolved_query,
            answer=answer,
            sources=sources, # we can pass dict directly since the schema validation handles it
            course_id=effective_course_id,
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
            "conversation_id": conversation_id,
            "query": request.query,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "chunks_retrieved": actual_chunks,
            "latency_ms": round(latency_ms, 2),
            "cache_hit": False,
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
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    start_time = time.perf_counter()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    await _verify_conversation_ownership(conversation_id, current_user)

    resolved_query = await resolve_numeric_query(request.query, conversation_id)
    logger.info("Stream request received", query=request.query[:80], resolved_query=resolved_query[:80] if resolved_query != request.query else None, conversation_id=conversation_id)

    context = await _prepare_rag_context(request, current_user, conversation_id, resolved_query, db)
    cached = context.get("cached")

    if cached:
        async def _stream_cached():
            latency_ms = (time.perf_counter() - start_time) * 1000
            if resolved_query != request.query:
                yield f"event: resolved\ndata: {json.dumps({'resolved_query': resolved_query})}\n\n"

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

        return StreamingResponse(_stream_cached(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    initial_state = context["initial_state"]
    rag_graph = get_rag_graph()

    async def _stream_rag():
        nonlocal resolved_query
        full_answer = ""
        retrieved_context = []
        intent = "KNOWLEDGE"

        try:
            langfuse = get_client()
            with propagate_attributes(
                trace_name="peped-chat-stream",
                session_id=conversation_id,
                user_id=f"{current_user.username} (ID: {current_user.user_id})",
                tags=["api-chat-stream", settings.app_env],
                metadata={
                    "cache_hit": False,
                    "env": settings.app_env,
                    "original_query": request.query,
                    "resolved_query": resolved_query,
                    "streaming": True,
                }
            ):
                langfuse_handler = CallbackHandler()
                config = {"run_name": "peped-chat-stream", "callbacks": [langfuse_handler]}

                if resolved_query != request.query:
                    yield f"event: resolved\ndata: {json.dumps({'resolved_query': resolved_query})}\n\n"

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
                            if "rewritten_query" in output:
                                new_rewrite = output.get("rewritten_query")
                                if new_rewrite and new_rewrite != resolved_query:
                                    resolved_query = new_rewrite
                                    yield f"event: resolved\ndata: {json.dumps({'resolved_query': resolved_query})}\n\n"

                    if kind == "on_chat_model_stream":
                        node_name = event.get("metadata", {}).get("langgraph_node")
                        if node_name in ("generate_node", "greeting", "ambiguity", "malicious"):
                            chunk = event.get("data", {}).get("chunk")
                            if chunk and hasattr(chunk, "content") and chunk.content:
                                token = chunk.content
                                full_answer += token
                                yield f"data: {json.dumps({'token': token})}\n\n"

                if await req.is_disconnected():
                    logger.info("Client disconnected during stream", conversation_id=conversation_id)
                    return

        except Exception as exc:
            logger.error("Stream pipeline error", error=str(exc), query=request.query[:60])
            yield f"event: error\ndata: {json.dumps({'error': 'RAG pipeline failed'})}\n\n"
            return

        latency_ms = (time.perf_counter() - start_time) * 1000
        sources = _extract_sources(retrieved_context)

        # ── Langfuse: log hybrid + Cohere scores for stream endpoint ──
        try:
            langfuse_stream = get_client()
            cur_trace_id = getattr(langfuse_handler, "last_trace_id", None) if langfuse_handler else None

            hybrid_scores_s = [c.get("hybrid_score") for c in retrieved_context if isinstance(c.get("hybrid_score"), (int, float))]
            cohere_scores_s  = [c.get("cohere_score")  for c in retrieved_context if isinstance(c.get("cohere_score"),  (int, float))]

            if cur_trace_id:
                if hybrid_scores_s:
                    langfuse_stream.score(trace_id=cur_trace_id, name="retriever_hybrid_max",  value=round(max(hybrid_scores_s), 4))
                    langfuse_stream.score(trace_id=cur_trace_id, name="retriever_hybrid_avg",  value=round(sum(hybrid_scores_s) / len(hybrid_scores_s), 4))
                if cohere_scores_s:
                    langfuse_stream.score(trace_id=cur_trace_id, name="retriever_cohere_max",  value=round(max(cohere_scores_s), 4))
                    langfuse_stream.score(trace_id=cur_trace_id, name="retriever_cohere_avg",  value=round(sum(cohere_scores_s) / len(cohere_scores_s), 4))
        except Exception as _lf_err:
            logger.warning(f"Stream Langfuse score logging failed: {_lf_err}")

        yield f"event: done\ndata: {json.dumps({'sources': sources, 'conversation_id': conversation_id, 'cached': False, 'latency_ms': round(latency_ms, 2)})}\n\n"

        try:
            effective_course_id = _auto_detect_course_id(retrieved_context, request.course_id)
            if intent not in ("GREETING", "AMBIGUOUS", "MALICIOUS"):
                await set_cached_response(query=resolved_query, answer=full_answer, sources=sources, course_id=effective_course_id)
            await append_to_history(conversation_id=conversation_id, user_message=resolved_query, assistant_message=full_answer)
            await batch_logger.add_log({
                "conversation_id": conversation_id,
                "query": request.query,
                "rewritten_query": resolved_query,
                "answer": full_answer,
                "chunks_retrieved": len(retrieved_context),
                "latency_ms": round(latency_ms, 2),
                "cache_hit": False,
            })
            await _schedule_afk_ltm_sync(conversation_id, current_user.user_id)
            # Fix: use asyncio.create_task instead of bare sync call inside async generator
            asyncio.create_task(asyncio.to_thread(_flush_langfuse))
        except Exception as bg_err:
            logger.warning(f"Stream background task error: {bg_err}")

    return StreamingResponse(_stream_rag(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
