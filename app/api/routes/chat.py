"""
POST /chat endpoint — the primary RAG pipeline entrypoint.

Flow:
  1. Check Redis cache for repeated queries
  2. Retrieve conversation history from Redis
  3. Run LangGraph RAG pipeline (query_processor → hybrid_retriever → reranker → context_builder → response_generator)
  4. Cache the result
  5. Persist query log to PostgreSQL
  6. Return answer + sources
"""
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.memory import append_to_history, get_conversation_history
from app.api.schemas import ChatRequest, ChatResponse, SourceReference
from app.database.models import AgentLog
from app.database.postgres import get_db
from app.graph.pipeline import get_rag_graph
from app.utils.cache import get_cached_response, set_cached_response

router = APIRouter()


@router.post("/chat", response_model=ChatResponse, summary="Ask a question using the RAG pipeline")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Process a user query through the full Hybrid RAG pipeline.

    - Returns cached answer if query was recently processed.
    - Otherwise runs the full LangGraph pipeline.
    - Logs the interaction to PostgreSQL.
    """
    start_time = time.perf_counter()
    conversation_id = request.conversation_id or str(uuid.uuid4())

    logger.info(
        "Chat request received",
        query=request.query[:80],
        conversation_id=conversation_id,
    )

    # ── 1. Cache check ────────────────────────────────────────────────────
    cached = await get_cached_response(request.query)
    if cached:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return ChatResponse(
            answer=cached["answer"],
            sources=[SourceReference(**s) for s in cached["sources"]],
            conversation_id=conversation_id,
            cached=True,
            latency_ms=round(latency_ms, 2),
        )

    # ── 2. Run Agentic RAG pipeline ───────────────────────────────────────
    # Retrieve conversation history from Redis for multi-turn continuity
    from langchain_core.messages import HumanMessage, AIMessage
    
    history = await get_conversation_history(conversation_id)
    
    # Convert history to proper LangChain message objects
    messages = []
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    # Add the current user query
    messages.append(HumanMessage(content=request.query))

    rag_graph = get_rag_graph()
    initial_state = {
        "messages": messages,
        "conversation_id": conversation_id,
    }

    try:
        result = await rag_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("RAG pipeline error", error=str(exc), query=request.query[:60])
        raise HTTPException(status_code=500, detail="RAG pipeline failed") from exc

    # The prebuilt create_react_agent stores the final AI response in the last message
    final_message = result["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)
    
    # We no longer strictly track `sources_raw` and `top_k` chunk lists in the same way 
    # since the agent handles retrieval dynamically. But we can log that it ran.
    latency_ms = (time.perf_counter() - start_time) * 1000

    # ── 3. Store in cache ─────────────────────────────────────────────────
    await set_cached_response(
        query=request.query,
        answer=answer,
        sources=[],  # Dynamic agent inline cites instead of returning raw payload references
    )

    # ── 4. Append to conversation memory ─────────────────────────────────
    await append_to_history(
        conversation_id=conversation_id,
        user_message=request.query,
        assistant_message=answer,
    )

    # ── 5. Log to PostgreSQL ──────────────────────────────────────────────
    
    # Deduce if a tool was called by checking the message length
    # A standard query without tool calling: UserMessage -> AIMessage (len 2)
    # A query with tool calling: User -> AI(ToolCall) -> ToolMessage -> AIMessage (len 4+)
    tool_was_called = len(result.get("messages", [])) > 2
    chunks_retrieved = 1 if tool_was_called else 0
    
    log_entry = AgentLog(
        conversation_id=conversation_id,
        query=request.query,
        rewritten_query="react_agent",
        answer=answer,
        chunks_retrieved=chunks_retrieved,
        latency_ms=round(latency_ms, 2),
        cache_hit=False,
    )
    db.add(log_entry)

    logger.info(
        "Chat response sent",
        query=request.query[:60],
        latency_ms=round(latency_ms, 2),
        tool_was_called=tool_was_called,
        sources=0,
    )

    return ChatResponse(
        answer=answer,
        sources=[],
        conversation_id=conversation_id,
        cached=False,
        latency_ms=round(latency_ms, 2),
    )
