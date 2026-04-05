"""
POST /chat endpoint — the primary RAG pipeline entrypoint.

Langfuse v4 Tracing Strategy (CLEAN DASHBOARD):
- ONE root trace per user request using start_as_current_observation()
- All LangGraph sub-operations (classifier, rag_node, generate_node, ChatOpenAI)
  are nested INSIDE that single root trace.
- Dashboard shows only 1 row per user request; click to see full detail.
- Cache hits also get their own single trace tagged "cache_hit".
"""
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.memory import append_to_history, get_conversation_history, resolve_numeric_query
from app.api.schemas import ChatRequest, ChatResponse, SourceReference
from app.database.models import AgentLog
from app.database.postgres import get_db, AsyncSessionLocal
from app.graph.pipeline import get_rag_graph
from app.utils.cache import get_cached_response, set_cached_response
from app.config.settings import get_settings
from app.observability import get_langfuse_client
from app.utils.logger_batch import batch_logger
from app.api.auth import get_current_user, User

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


@router.post("/chat", response_model=ChatResponse, summary="Ask a question using the RAG pipeline")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """
    Process a user query through the full Hybrid RAG pipeline.
    Creates exactly ONE Langfuse trace per request (clean dashboard).
    Click any trace to see all internal steps nested inside.
    """
    start_time = time.perf_counter()
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Resolve numeric query to follow-up question if applicable
    resolved_query = await resolve_numeric_query(request.query, conversation_id)

    logger.info("Chat request received", query=request.query[:80], resolved_query=resolved_query[:80] if resolved_query != request.query else None, conversation_id=conversation_id)

    # ── 1. Cache check ────────────────────────────────────────────────────
    cached = await get_cached_response(resolved_query, course_id=request.course_id)
    if cached:
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Official Langfuse v4 context manager pattern (from langfuse.com/docs/observability/sdk/instrumentation)
        try:
            from langfuse import get_client, propagate_attributes
            langfuse = get_client()  # use the official global client, not the custom singleton
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="peped-chat",
                input={"query": request.query, "resolved_query": resolved_query, "conversation_id": conversation_id},
            ) as root_obs:
                with propagate_attributes(
                    trace_name="peped-chat",  # this is what sets the label in the dashboard
                    session_id=conversation_id,
                    user_id=conversation_id,
                    tags=["api-chat", settings.app_env, "cache-hit"],
                ):
                    root_obs.update(
                        output=cached["answer"],
                        usage={"input": 0, "output": 0},
                        metadata={
                            "latency_ms": round(latency_ms, 2),
                            "cache_hit": True,
                            "env": settings.app_env,
                        },
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

        # FIX: Append to history EVEN for cache hits, so follow-up queries aren't stuck on old context
        await append_to_history(
            conversation_id=conversation_id,
            user_message=resolved_query,
            assistant_message=cached["answer"],
        )

        return ChatResponse(
            answer=cached["answer"],
            sources=[SourceReference(**s) for s in cached["sources"]],
            conversation_id=conversation_id,
            resolved_query=resolved_query if resolved_query != request.query else None,
            cached=True,
            latency_ms=round(latency_ms, 2),
        )

    # ── 2. Build message history ──────────────────────────────────────────
    from langchain_core.messages import HumanMessage, AIMessage
    from app.agents.memory import get_or_summarize_history
    from app.llm.client import get_cheap_llm

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

    # ── 2b. Long-Term Memory (LTM): Semantic per-request retrieval ──────────
    # Uses QdrantLTMService: embeds the current query and retrieves the top-2
    # most semantically relevant past episodes for this user.
    # Per-request retrieval (not just new_session) ensures the most relevant
    # context is always injected, regardless of session state.
    from app.api.user_utils import is_real_user
    from app.agents.long_term_memory_qdrant import qdrant_ltm

    user_id = current_user.user_id
    ltm_eligible = is_real_user(user_id=user_id, role=current_user.role)

    ltm_profile = {"summary": "", "course_names": []}

    if ltm_eligible:
        ltm_profile = await qdrant_ltm.load(user_id=user_id, query=resolved_query)
        if ltm_profile["summary"]:
            logger.info(
                "LTM profile loaded (semantic)",
                user_id=user_id,
                course_names=len(ltm_profile.get("course_names", [])),
            )
        else:
            logger.debug("No relevant LTM episodes found for user", user_id=user_id)

    rag_graph = get_rag_graph()
    initial_state = {
        "messages": messages,
        "conversation_id": conversation_id,
        "conversation_summary": summary,
        "user_profile": ltm_profile,
    }

    # ── 3. Run RAG pipeline inside a SINGLE Langfuse trace ────────────────
    #
    # Official Langfuse v4 context manager pattern:
    # https://langfuse.com/docs/observability/sdk/instrumentation#context-manager
    #
    #   langfuse = get_client()   ← must use the official get_client(), not custom singleton
    #   with langfuse.start_as_current_observation(as_type="span", name="...") as obs:
    #       with propagate_attributes(trace_name="...", session_id=..., user_id=...):
    #           handler = CallbackHandler()   ← inherits the active OTel context automatically
    #           result = await graph.ainvoke(..., config={"callbacks": [handler]})
    #
    # trace_name in propagate_attributes = the display name in the Langfuse dashboard.
    # All LangGraph sub-steps nest inside ONE trace row. Token usage visible in detail.

    from langfuse.langchain import CallbackHandler
    from langfuse import get_client, propagate_attributes

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
            user_id=conversation_id,
            tags=["api-chat", settings.app_env],
            metadata={
                "cache_hit": False,
                "env": settings.app_env,
                "original_query": request.query,
                "resolved_query": resolved_query,
            }
        ):
            # NO wrapper observation here!
            # The CallbackHandler natively creates a TRACE (like in the LangGraph screenshot)
            # This TRACE natively tracks duration, cost, and tokens accurately.
            langfuse_handler = CallbackHandler()

            result = await rag_graph.ainvoke(
                initial_state,
                config={
                    "run_name": "peped-chat",
                    "callbacks": [langfuse_handler],
                },
            )

            # Compute latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Get final answer
            final_message = result["messages"][-1]
            answer = final_message.content if hasattr(final_message, "content") else str(final_message)

            # Override the trace I/O from raw state to clean query/answer
            langfuse.set_current_trace_io(
                input={"query": request.query, "resolved_query": resolved_query, "conversation_id": conversation_id},
                output={"answer": answer}
            )

        # Capture trace_id for scoring
        if hasattr(langfuse_handler, "last_trace_id"):
            trace_id = langfuse_handler.last_trace_id

    except Exception as exc:
        logger.error("RAG pipeline error", error=str(exc), query=request.query[:60])
        background_tasks.add_task(_flush_langfuse)
        raise HTTPException(status_code=500, detail="RAG pipeline failed") from exc

    # ── 4. Extract retrieval info from state ──────────────────────────────
    rewritten_query = resolved_query
    actual_chunks = 0
    max_chunk_score = None

    retrieved_context = result.get("retrieved_context") or []
    if retrieved_context:
        real_chunks = [c for c in retrieved_context if c.get("source") not in (None, "", "None")]
        actual_chunks = len(real_chunks) if real_chunks else len(retrieved_context)
        scores = [c.get("score") for c in retrieved_context if isinstance(c.get("score"), (int, float))]
        if scores:
            max_chunk_score = max(scores)

    # ── 4.1 Auto-detect course_id from retrieved chunks ───────────────────
    # If user didn't send course_id (or sent 0 from test UI), extract it from the 
    # Knowledge_Base chunk metadata (each point in Qdrant KB has course_id in payload).
    effective_course_id = request.course_id
    if effective_course_id in (None, 0) and retrieved_context:
        # Pick the most frequent course_id from top chunks (majority vote)
        from collections import Counter
        cids = [
            c.get("course_id") for c in retrieved_context
            if c.get("course_id") not in (None, "", 0)
        ]
        if cids:
            most_common = Counter(cids).most_common(1)[0][0]
            try:
                effective_course_id = int(most_common)
            except (ValueError, TypeError):
                pass
            logger.info("Auto-detected course_id from chunks", course_id=effective_course_id)

    # ── 5. Log Retriever Score to Langfuse ────────────────────────────────
    if trace_id and max_chunk_score is not None:
        try:
            from langfuse import get_client as _get_lf
            _lf = _get_lf()
            logger.info(f"Submitting max_chunk_score={max_chunk_score} to trace_id={trace_id}")
            _lf.score(
                trace_id=trace_id,
                name="retriever_max_score",
                value=float(max_chunk_score),
            )
        except Exception as e:
            logger.warning(f"Failed to submit Langfuse score: {e}")

    # ── 6. Flush Langfuse in background ───────────────────────────────────
    background_tasks.add_task(_flush_langfuse)

    # ── 7. Store cache + conversation memory ──────────────────────────────
    # Map retrieved context to SourceReference objects for the response
    sources = []
    if retrieved_context:
        for c in retrieved_context:
            # Only include chunks that have a source (avoid empty/placeholder chunks)
            if c.get("source") and c.get("source") != "Unknown":
                sources.append(
                    SourceReference(
                        chunk_id=c.get("chunk_id") or str(uuid.uuid4()),
                        document_id=c.get("document_id") or "Unknown",
                        source=c.get("source"),
                        title=c.get("course_name") or c.get("title") or "Unknown",
                        chunk_index=c.get("chunk_index") or 0,
                        score=c.get("score") or 0.0,
                    )
                )

    background_tasks.add_task(
        set_cached_response,
        query=resolved_query,
        answer=answer,
        sources=[s.model_dump() for s in sources],    )

    background_tasks.add_task(
        set_cached_response,
        query=resolved_query,
        answer=answer,
        sources=[s.model_dump() for s in sources],
        course_id=effective_course_id,
    )
    background_tasks.add_task(
        append_to_history,
        conversation_id=conversation_id,
        user_message=resolved_query,
        assistant_message=answer,
    )

    # ── 7b. Long-Term Memory (LTM) Architecture ─────────────────────────
    # LTM updates are no longer performed per-turn to optimize cost and latency.
    # Synchronization is now handled by a background worker after a 30-minute
    # period of user inactivity (AFK), utilizing cost-efficient models (Gemini Flash).

    # ── 8. Log to PostgreSQL ──────────────────────────────────────────────
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
    
    # ── 9. Schedule AFK LTM Sync (30 minutes, deduplicated) ─────────────────
    # One deferred arq task per conversation — dedup via Redis NX key.
    # If user sends multiple messages, only ONE task reaches the worker;
    # all others are skipped via the dedup_key guard in sync_ltm_task.
    async def _schedule_afk_ltm_sync(conv_id: str, u_id: str):
        from app.database.redis_client import get_redis_client
        from app.api.routes.ingest import get_arq_redis
        import time
        import datetime

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
            await arq_redis.close()
            # Mark as scheduled (TTL slightly > 10s so key covers the defer window)
            await redis_client.set(sched_key, "1", ex=60)
            logger.debug("AFK LTM sync scheduled", conversation_id=conv_id)
        except Exception as e:
            logger.warning(f"Failed to schedule AFK LTM sync: {e}")

    background_tasks.add_task(_schedule_afk_ltm_sync, conversation_id, current_user.user_id)

    return ChatResponse(
        answer=answer,
        sources=sources,
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
    """Retrieve the chat history from memory for a specific conversation ID."""
    history = await get_conversation_history(conversation_id)
    return history


@router.delete("/chat/history/{conversation_id}", summary="Clear chat history for a session")
async def delete_history(
    conversation_id: str,
    current_user: Optional[User] = Depends(get_current_user),
):
    """Clear the chat history from memory for a specific conversation ID."""
    from app.agents.memory import clear_conversation_history
    await clear_conversation_history(conversation_id)
    return {"status": "success", "message": "Conversation history cleared"}


@router.post("/chat/sync_memory/{conversation_id}", summary="Sync chat history to Long-Term Memory")
async def sync_memory(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """
    Deprecated: Frontend trigger for LTM sync is no longer required.
    LTM is now automatically handled by the AFK 30-minute worker.
    """
    return {"status": "ignored", "reason": "handled_by_afk_worker_in_background"}
tional[User] = Depends(get_current_user),
) -> list[dict]:
    """Retrieve the chat history from memory for a specific conversation ID."""
    history = await get_conversation_history(conversation_id)
    return history


@router.delete("/chat/history/{conversation_id}", summary="Clear chat history for a session")
async def delete_history(
    conversation_id: str,
    current_user: Optional[User] = Depends(get_current_user),
):
    """Clear the chat history from memory for a specific conversation ID."""
    from app.agents.memory import clear_conversation_history
    await clear_conversation_history(conversation_id)
    return {"status": "success", "message": "Conversation history cleared"}


@router.post("/chat/sync_memory/{conversation_id}", summary="Sync chat history to Long-Term Memory")
async def sync_memory(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """
    Deprecated: Frontend trigger for LTM sync is no longer required.
    LTM is now automatically handled by the AFK 30-minute worker.
    """
    return {"status": "ignored", "reason": "handled_by_afk_worker_in_background"}
