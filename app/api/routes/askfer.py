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

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from loguru import logger

from app.api.askfer_deps import rate_limit_by_ip, rate_limit_sync_by_ip
from app.api.concurrency import acquire_pipeline_slot_or_503
from app.api.schemas import AskferRequest, AskferSyncRequest
from app.config.settings import get_settings
from app.graph.askfer_pipeline import get_askfer_graph
from app.utils.cache import get_cached_response, set_cached_response
from app.utils.logger_batch import batch_logger
from app.worker import sync_portfolio_task  # noqa: E402

router = APIRouter()
settings = get_settings()

_CACHE_NS = "askfer"

_stream_bg_tasks: set[asyncio.Task] = set()


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
    client_ip: str = Depends(rate_limit_by_ip),
):
    """Stateless SSE stream. No auth, no conversation history."""
    # Pipeline semaphore — held for the entire SSE stream. Both the cache-hit
    # and rag-stream paths release the permit in their generator's `finally`.
    # Raises HTTP 503 with Retry-After: 5 on saturation.
    sem_release = await acquire_pipeline_slot_or_503()
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
            try:
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
            finally:
                sem_release()
        return StreamingResponse(
            _stream_cached(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    initial_state = {"messages": [HumanMessage(content=query)]}
    askfer_graph = get_askfer_graph()

    async def _stream_askfer():
        from app.graph.pipeline import StreamLeakGuard, _sanitize_answer
        full_answer = ""
        retrieved_context: list = []
        intent = "KNOWLEDGE"
        leak_guard = StreamLeakGuard()

        try:
            config = {"run_name": "askfer-stream"}

            token_count = 0
            # Track which terminal nodes finished. The malicious AND
            # off_scope nodes both return canned AIMessages (no LLM call →
            # no on_chat_model_stream events) so we emit their content
            # from on_chain_end instead.
            canned_emitted = False

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

                # Canned-response nodes: emit their AIMessage content as a
                # single token chunk since no LLM stream fires for them.
                if (
                    kind == "on_chain_end"
                    and event.get("name") in ("malicious", "off_scope")
                    and not canned_emitted
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
                    if node_name in ("generate_node", "greeting", "off_scope", "malicious"):
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            token = chunk.content
                            full_answer += token
                            token_count += 1
                            safe = leak_guard.feed(token)
                            if safe:
                                yield f"data: {json.dumps({'token': safe})}\n\n"

                            if token_count % 10 == 0 and await req.is_disconnected():
                                logger.info("Askfer client disconnected", ip=client_ip, tokens=token_count)
                                return

        except Exception as exc:
            logger.error(f"Askfer pipeline error: {exc}", query=query[:60])
            yield f"event: error\ndata: {json.dumps({'error': 'Askfer pipeline failed'})}\n\n"
            return

        # Drain buffered preamble — emit sanitized tail (or clean preamble).
        tail = leak_guard.flush()
        if tail:
            yield f"data: {json.dumps({'token': tail})}\n\n"
        if leak_guard.leak_detected:
            full_answer = tail
        # Belt-and-suspenders pass for any mid-stream leak that bypassed
        # the preamble guard.
        cleaned_answer = _sanitize_answer(full_answer)
        if cleaned_answer != full_answer:
            logger.warning(
                "Askfer stream leak sanitized before cache/log "
                f"(orig_len={len(full_answer)} clean_len={len(cleaned_answer)})"
            )
            full_answer = cleaned_answer

        latency_ms = (time.perf_counter() - start_time) * 1000
        sources = _extract_askfer_sources(retrieved_context)

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
                "llm_tokens_used": token_count,
                "cache_hit": False,
            })
        except Exception as exc:
            logger.warning(f"Askfer background task error: {exc}")
        finally:
            # Release the pipeline permit at end of stream (normal completion,
            # client disconnect, or exception inside the generator).
            sem_release()

    return StreamingResponse(
        _stream_askfer(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/askfer/sync", summary="Trigger portfolio re-sync (admin)")
async def askfer_sync(
    request: AskferSyncRequest,
    x_admin_secret: str = Header(..., alias="X-Admin-Secret"),
    _client_ip: str = Depends(rate_limit_sync_by_ip),
):
    """Enqueue a streaq task to re-scrape homepage + projects + CV. Requires
    `X-Admin-Secret` header matching ASKFER_ADMIN_SECRET (constant-time check)."""
    if not settings.askfer_admin_secret:
        raise HTTPException(status_code=503, detail="Askfer admin secret not configured")
    if not secrets.compare_digest(x_admin_secret, settings.askfer_admin_secret):
        raise HTTPException(status_code=403, detail="Forbidden")

    # streaq: enqueue returns a Task; `await task` publishes to Redis. The
    # `force_reingest` arg name must match the worker task signature
    # (app/worker.py:sync_portfolio_task).
    task = await sync_portfolio_task.enqueue(force_reingest=request.force_reingest)
    logger.info("Askfer portfolio sync enqueued", force_reingest=request.force_reingest)
    return {"message": "Portfolio sync enqueued", "job_id": task.id}
