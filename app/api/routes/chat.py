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

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.memory import append_to_history, get_conversation_history
from app.api.schemas import ChatRequest, ChatResponse, SourceReference
from app.database.models import AgentLog
from app.database.postgres import get_db, AsyncSessionLocal
from app.graph.pipeline import get_rag_graph
from app.utils.cache import get_cached_response, set_cached_response
from app.config.settings import get_settings
from app.observability import get_langfuse_client

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


async def log_to_db(
    conversation_id: str,
    query: str,
    rewritten_query: str,
    answer: str,
    chunks_retrieved: int,
    latency_ms: float,
    cache_hit: bool,
):
    """Background task to persist logs to PostgreSQL."""
    async with AsyncSessionLocal() as session:
        try:
            log_entry = AgentLog(
                conversation_id=conversation_id,
                query=query,
                rewritten_query=rewritten_query,
                answer=answer,
                chunks_retrieved=chunks_retrieved,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
            )
            session.add(log_entry)
            await session.commit()
        except Exception as exc:
            logger.error("Failed to log to database in background", error=str(exc))


@router.post("/chat", response_model=ChatResponse, summary="Ask a question using the RAG pipeline")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Process a user query through the full Hybrid RAG pipeline.
    Creates exactly ONE Langfuse trace per request (clean dashboard).
    Click any trace to see all internal steps nested inside.
    """
    start_time = time.perf_counter()
    conversation_id = request.conversation_id or str(uuid.uuid4())

    logger.info("Chat request received", query=request.query[:80], conversation_id=conversation_id)

    # ── 1. Cache check ────────────────────────────────────────────────────
    cached = await get_cached_response(request.query)
    if cached:
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Official Langfuse v4 context manager pattern (from langfuse.com/docs/observability/sdk/instrumentation)
        try:
            from langfuse import get_client, propagate_attributes
            langfuse = get_client()  # use the official global client, not the custom singleton
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="peped-chat",
                input={"query": request.query, "conversation_id": conversation_id},
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
            log_to_db,
            conversation_id=conversation_id,
            query=request.query,
            rewritten_query=request.query,
            answer=cached["answer"],
            chunks_retrieved=0,
            latency_ms=round(latency_ms, 2),
            cache_hit=True,
        )

        return ChatResponse(
            answer=cached["answer"],
            sources=[SourceReference(**s) for s in cached["sources"]],
            conversation_id=conversation_id,
            cached=True,
            latency_ms=round(latency_ms, 2),
        )

    # ── 2. Build message history ──────────────────────────────────────────
    from langchain_core.messages import HumanMessage, AIMessage

    history = await get_conversation_history(conversation_id)
    recent_history = history[-(3 * 2):] if len(history) > 6 else history
    messages = []
    for turn in recent_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    messages.append(HumanMessage(content=request.query))

    rag_graph = get_rag_graph()
    initial_state = {"messages": messages, "conversation_id": conversation_id}

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
                input={"query": request.query, "conversation_id": conversation_id},
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
    rewritten_query = request.query
    actual_chunks = 0
    max_chunk_score = None

    retrieved_context = result.get("retrieved_context") or []
    if retrieved_context:
        real_chunks = [c for c in retrieved_context if c.get("source") not in (None, "", "None")]
        actual_chunks = len(real_chunks) if real_chunks else len(retrieved_context)
        scores = [c.get("score") for c in retrieved_context if isinstance(c.get("score"), (int, float))]
        if scores:
            max_chunk_score = max(scores)

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
    await set_cached_response(query=request.query, answer=answer, sources=[])
    await append_to_history(
        conversation_id=conversation_id,
        user_message=request.query,
        assistant_message=answer,
    )

    # ── 8. Log to PostgreSQL ──────────────────────────────────────────────
    background_tasks.add_task(
        log_to_db,
        conversation_id=conversation_id,
        query=request.query,
        rewritten_query=rewritten_query,
        answer=answer,
        chunks_retrieved=actual_chunks,
        latency_ms=round(latency_ms, 2),
        cache_hit=False,
    )

    logger.info(
        "Chat response sent",
        query=request.query[:60],
        latency_ms=round(latency_ms, 2),
        chunks_retrieved=actual_chunks,
        max_chunk_score=max_chunk_score,
    )

    return ChatResponse(
        answer=answer,
        sources=[],
        conversation_id=conversation_id,
        cached=False,
        latency_ms=round(latency_ms, 2),
    )
