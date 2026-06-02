"""
Concurrency guards for hot-path endpoints.

`max_concurrent_pipelines` is declared in settings but was never enforced
before — a burst of 50 simultaneous /chat requests would all start
embedding + LLM calls in parallel and OOM the 1.5 GB uvicorn worker.
This module turns the setting into a real asyncio.Semaphore and exposes
two helpers:

  - `acquire_pipeline_slot()` for non-streaming endpoints: returns the
    semaphore, release in `finally`.
  - `acquire_pipeline_slot_or_503()`: for streaming endpoints that need
    the slot held for the entire SSE lifetime. Returns a release
    callable so the generator's `finally` can let go when the stream
    ends or the client disconnects.

A 5s acquire timeout matches the original setting; on saturation the
route returns 503 with `Retry-After: 5` so clients back off cleanly
instead of queueing invisibly.
"""
from __future__ import annotations

import asyncio
from typing import Callable

from fastapi import HTTPException, status

from app.config.settings import get_settings

_settings = get_settings()
_pipeline_sem: asyncio.Semaphore = asyncio.Semaphore(_settings.max_concurrent_pipelines)


def _release_slot() -> None:
    """Release one permit on the pipeline semaphore."""
    _pipeline_sem.release()


async def acquire_pipeline_slot() -> asyncio.Semaphore:
    """Acquire one permit for a non-streaming RAG call.

    Returns the semaphore itself; release in a `finally` block:

        sem = await acquire_pipeline_slot()
        try:
            ...
        finally:
            sem.release()

    Raises HTTP 503 with `Retry-After: 5` if no permit becomes available
    within `pipeline_acquire_timeout_s` seconds.
    """
    try:
        await asyncio.wait_for(
            _pipeline_sem.acquire(),
            timeout=_settings.pipeline_acquire_timeout_s,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is at capacity. Please retry shortly.",
            headers={"Retry-After": "5"},
        ) from exc
    return _pipeline_sem


async def acquire_pipeline_slot_or_503() -> Callable[[], None]:
    """Acquire a permit for a streaming RAG call (SSE).

    Returns a synchronous `release` callable that the StreamingResponse
    generator MUST invoke in its `finally` block, so the permit is held
    for the entire stream lifetime and released when the stream ends
    or the client disconnects.

        sem_release = await acquire_pipeline_slot_or_503()
        async def _stream():
            try:
                ...
                yield "..."
            finally:
                sem_release()

    Raises HTTP 503 with `Retry-After: 5` on saturation.
    """
    try:
        await asyncio.wait_for(
            _pipeline_sem.acquire(),
            timeout=_settings.pipeline_acquire_timeout_s,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is at capacity. Please retry shortly.",
            headers={"Retry-After": "5"},
        ) from exc
    return _release_slot
