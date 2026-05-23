"""
Askfer routes — public portfolio chat endpoints.

POST /askfer/stream   → SSE stream answering questions about Ferdy's portfolio.
POST /askfer/sync     → admin-only re-sync of homepage + projects + CV.

A-Pedi (`/chat`, `/chat/stream`) is untouched.
"""
import asyncio
import json
import secrets
import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.askfer_deps import rate_limit_by_ip
from app.api.routes.ingest import get_arq_redis
from app.api.schemas import AskferRequest, AskferSyncRequest
from app.config.settings import get_settings
from app.database.postgres import get_db
from app.graph.askfer_pipeline import get_askfer_graph
from app.observability import get_langfuse_client
from app.utils.cache import get_cached_response, set_cached_response
from app.utils.logger_batch import batch_logger

router = APIRouter()
settings = get_settings()

_CACHE_NS = "askfer"

_stream_bg_tasks: set[asyncio.Task] = set()


def _flush_langfuse():
    lf = get_langfuse_client()
    if lf:
        try:
            lf.flush()
        except Exception as exc:
            logger.warning(f"Askfer Langfuse flush error: {exc}")


def _extract_askfer_sources(retrieved_context: list) -> list:
    """Dedupe sources by url/source so each project surfaces once."""
    sources = []
    seen: set[str] = set()
    for c in retrieved_context or []:
        url = c.get("project_url") or c.get("source") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        sources.append({
            "doc_type": c.get("doc_type", ""),
            "title": c.get("title") or c.get("project_slug") or "Unknown",
            "url": url,
            "score": c.get("score") or 0.0,
        })
    return sources


@router.post("/askfer/stream", summary="Ask Askfer about Ferdy's portfolio (SSE)")
async def askfer_stream(
    request: AskferRequest,
    req: Request,
    db: AsyncSession = Depends(get_db),
    client_ip: str = Depends(rate_limit_by_ip),
):
    """Stateless SSE stream. No auth, no conversation history."""
    start_time = time.perf_counter()
    query = request.query
    logger.info("Askfer request", query=query[:80], ip=client_ip)

    # Cache lookup (askfer namespace — fully isolated from A-Pedi)
    from llama_index.core import Settings as LISettings
    from app.config.embedding_config import ensure_llamaindex_configured

    query_embedding = None
    try:
        ensure_llamaindex_configured()
        query_embedding = await LISettings.embed_model.aget_query_embedding(query)
    except Exception as exc:
        logger.warning(f"Askfer embed-once failed: {exc}")

    cached = await get_cached_response(
        query,
        course_id=None,
        query_embedding=query_embedding,
        cache_namespace=_CACHE_NS,
    )

    if cached:
        async def _stream_cached():
            latency_ms = (time.perf_counter() - start_time) * 1000
            words = cached["answer"].split(" ")
            chunk_size = 4
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if i > 0:
                    chunk = " " + chunk
                yield f"data: {json.dumps({'token': chunk})}\n\n"
                await asyncio.sleep(0.02)
            yield (
                f"event: done\ndata: "
                f"{json.dumps({'sources': cached.get('sources', []), 'cached': True, 'latency_ms': round(latency_ms, 2)})}\n\n"
            )
        return StreamingResponse(
            _stream_cached(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    initial_state = {"messages": [HumanMessage(content=query)]}
    askfer_graph = get_askfer_graph()

    async def _stream_askfer():
        full_answer = ""
        retrieved_context: list = []
        intent = "KNOWLEDGE"
        langfuse_handler: Optional[CallbackHandler] = None

        try:
            langfuse = get_client()
            with propagate_attributes(
                trace_name="askfer-stream",
                session_id=f"ip:{client_ip}",
                user_id=f"anon-{client_ip}",
                tags=["askfer", settings.app_env],
                metadata={
                    "collection": settings.qdrant_personal_collection,
                    "streaming": True,
                    "query": query[:80],
                },
            ):
                langfuse_handler = CallbackHandler()
                config = {"run_name": "askfer-stream", "callbacks": [langfuse_handler]}

                token_count = 0

                async for event in askfer_graph.astream_events(initial_state, config=config, version="v2"):
                    kind = event.get("event", "")

                    if kind == "on_chain_end" and event.get("name") == "rag_node":
                        out = event.get("data", {}).get("output", {})
                        if isinstance(out, dict) and "retrieved_context" in out:
                            retrieved_context = out["retrieved_context"] or []

                    if kind == "on_chain_end" and event.get("name") == "pre_processor":
                        out = event.get("data", {}).get("output", {})
                        if isinstance(out, dict) and "intent" in out:
                            intent = out["intent"]

                    if kind == "on_chat_model_stream":
                        node_name = event.get("metadata", {}).get("langgraph_node")
                        if node_name in ("generate_node", "greeting", "off_scope", "malicious"):
                            chunk = event.get("data", {}).get("chunk")
                            if chunk and hasattr(chunk, "content") and chunk.content:
                                token = chunk.content
                                full_answer += token
                                token_count += 1
                                yield f"data: {json.dumps({'token': token})}\n\n"

                                if token_count % 10 == 0 and await req.is_disconnected():
                                    logger.info("Askfer client disconnected", ip=client_ip, tokens=token_count)
                                    return

        except Exception as exc:
            logger.error(f"Askfer pipeline error: {exc}", query=query[:60])
            yield f"event: error\ndata: {json.dumps({'error': 'Askfer pipeline failed'})}\n\n"
            return

        latency_ms = (time.perf_counter() - start_time) * 1000
        sources = _extract_askfer_sources(retrieved_context)

        # Langfuse retrieval scores
        try:
            cur_trace_id = getattr(langfuse_handler, "last_trace_id", None) if langfuse_handler else None
            if cur_trace_id and retrieved_context:
                lf = get_client()
                hybrid = [c.get("hybrid_score") for c in retrieved_context if isinstance(c.get("hybrid_score"), (int, float))]
                cohere = [c.get("score") for c in retrieved_context if isinstance(c.get("score"), (int, float))]
                if hybrid:
                    lf.score(trace_id=cur_trace_id, name="retriever_hybrid_max", value=round(max(hybrid), 4))
                    lf.score(trace_id=cur_trace_id, name="retriever_hybrid_avg", value=round(sum(hybrid)/len(hybrid), 4))
                if cohere:
                    lf.score(trace_id=cur_trace_id, name="retriever_cohere_max", value=round(max(cohere), 4))
                    lf.score(trace_id=cur_trace_id, name="retriever_cohere_avg", value=round(sum(cohere)/len(cohere), 4))
        except Exception as exc:
            logger.warning(f"Askfer Langfuse score logging failed: {exc}")

        yield (
            f"event: done\ndata: "
            f"{json.dumps({'sources': sources, 'cached': False, 'latency_ms': round(latency_ms, 2)})}\n\n"
        )

        # Background: cache write + log
        try:
            if intent == "KNOWLEDGE" and full_answer:
                await set_cached_response(
                    query=query,
                    answer=full_answer,
                    sources=sources,
                    course_id=None,
                    query_embedding=query_embedding,
                    cache_namespace=_CACHE_NS,
                )
            await batch_logger.add_log({
                "endpoint": "askfer",
                "conversation_id": f"ip:{client_ip}",
                "query": query,
                "rewritten_query": query,
                "answer": full_answer,
                "chunks_retrieved": len(retrieved_context),
                "latency_ms": round(latency_ms, 2),
                "cache_hit": False,
            })
            task = asyncio.create_task(asyncio.to_thread(_flush_langfuse))
            _stream_bg_tasks.add(task)
            task.add_done_callback(_stream_bg_tasks.discard)
        except Exception as exc:
            logger.warning(f"Askfer background task error: {exc}")

    return StreamingResponse(
        _stream_askfer(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/askfer/sync", summary="Trigger portfolio re-sync (admin)")
async def askfer_sync(
    request: AskferSyncRequest,
    x_admin_secret: str = Header(..., alias="X-Admin-Secret"),
):
    """Enqueue an arq job to re-scrape homepage + projects + CV. Requires
    `X-Admin-Secret` header matching ASKFER_ADMIN_SECRET (constant-time check)."""
    if not settings.askfer_admin_secret:
        raise HTTPException(status_code=503, detail="Askfer admin secret not configured")
    if not secrets.compare_digest(x_admin_secret, settings.askfer_admin_secret):
        raise HTTPException(status_code=403, detail="Forbidden")

    arq_redis = await get_arq_redis()
    job = await arq_redis.enqueue_job('sync_portfolio_task', request.force_reingest)
    logger.info("Askfer portfolio sync enqueued", force_reingest=request.force_reingest)
    return {"message": "Portfolio sync enqueued", "job_id": job.job_id if job else None}
