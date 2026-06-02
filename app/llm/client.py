"""
OpenRouter LLM client via LangChain's ChatOpenAI integration.
Includes tenacity retry wrapping for resilience.
"""
from functools import lru_cache

import httpx
from langchain_openai import ChatOpenAI
from loguru import logger
from openai import APIError, APIStatusError, APITimeoutError, RateLimitError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config.settings import get_settings

settings = get_settings()


# 9Router (the self-hosted gateway) sits behind a WAF that blocks any
# request whose User-Agent contains "OpenAI" — the default UA emitted by
# the openai-python SDK. The OpenAI SDK rewrites the UA per-request, so
# httpx default_headers gets overridden. We use an event hook that fires
# AFTER the SDK builds the request to forcefully replace the UA.
_LLM_USER_AGENT = "ai-lms-agent/1.0"


async def _strip_openai_ua(request: httpx.Request) -> None:
    request.headers["User-Agent"] = _LLM_USER_AGENT


def _make_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        event_hooks={"request": [_strip_openai_ua]},
    )


# OpenAI SDK's built-in max_retries=1 is a no-op when http_async_client is
# provided (langchain ChatOpenAI bypasses the default sync/async client
# when a custom async client is set, so urllib3.Retry is never reached).
# We wrap ainvoke() with tenacity so transient OpenRouter errors
# (429, 500, 502, 503, 504, timeouts) get exponential backoff retry.
# Non-retriable errors (400 bad request, 401 auth, 402 out of credits,
# 403 forbidden) bubble up immediately — those won't fix themselves.
_RETRYABLE_EXCEPTIONS = (
    RateLimitError,        # 429
    APITimeoutError,       # request timed out
    APIStatusError,        # 5xx; we'll filter on status_code below
    APIError,              # catch-all for connection drops
)
_NON_RETRYABLE_5XX = (500, 501, 502, 503, 504)


def _should_retry(exc: BaseException) -> bool:
    """Decide whether a given exception is worth retrying.

    Returns True for:
      - RateLimitError (429) — back off and retry
      - APITimeoutError — transient network
      - APIStatusError with status_code in {500,501,502,503,504}
      - Other APIError subclasses (connection drops, protocol errors)
    Returns False for:
      - 400, 401, 402, 403, 404 — these are caller errors that retry
        won't fix
      - Any non-APIError exception (e.g. our own ValueError, KeyError)
    """
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APITimeoutError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _NON_RETRYABLE_5XX
    if isinstance(exc, APIError):
        # Connection drops, protocol errors, etc. — retry.
        return True
    return False


def _wrap_with_retry(llm: ChatOpenAI) -> None:
    """Replace llm.ainvoke with a tenacity-wrapped version.

    The singleton is created once per process (@lru_cache), so the
    monkey-patch is a one-time operation. Each call gets its own retry
    state via the @retry decorator, so concurrent callers don't share
    retry counters.

    Backoff: 1s, 2s, 4s, 8s (capped at 8s) on the first two retries.
    Max 3 attempts total (1 initial + 2 retries).

    Retries on: 429, 5xx, APITimeoutError, generic APIError (connection
    drops, protocol errors). Does NOT retry on: 400, 401, 402, 403, 404 —
    those are caller errors that retry won't fix.
    """
    original_ainvoke = llm.ainvoke

    def _log_before_sleep(retry_state):
        logger.warning(
            "LLM call failed, retrying",
            attempt=retry_state.attempt_number,
            next_sleep=retry_state.next_action.sleep if retry_state.next_action else None,
            error_type=type(retry_state.outcome.exception()).__name__ if retry_state.outcome else None,
            error=str(retry_state.outcome.exception())[:200] if retry_state.outcome else None,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        # Use a custom predicate (not retry_if_exception_type) so we can
        # apply both the type filter AND the status-code filter in one
        # step. Otherwise tenacity matches the type and re-runs before
        # _should_retry gets a chance to veto 4xx errors.
        retry=retry_if_exception(lambda exc: _should_retry(exc)),
        reraise=True,
        before_sleep=_log_before_sleep,
    )
    async def _retried(*args, **kwargs):
        return await original_ainvoke(*args, **kwargs)

    # ChatOpenAI is a pydantic BaseModel, which blocks arbitrary attribute
    # assignment. Use object.__setattr__ to bypass the pydantic handler.
    object.__setattr__(llm, "ainvoke", _retried)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return the singleton LLM client configured for 9Router/OpenRouter."""
    llm = ChatOpenAI(
        model=settings.llm_model,

        # openai_api_key="ollama",
        # openai_api_base=settings.ollama_base_url,

        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        request_timeout=60,
        max_retries=1,
        http_async_client=_make_http_client(),
        # OpenRouter-specific headers for app attribution (ignored by Ollama)
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent",
        },
    )
    _wrap_with_retry(llm)
    return llm


@lru_cache(maxsize=1)
def get_cheap_llm() -> ChatOpenAI:
    """Return a cheaper, faster LLM for background tasks (memory summarization,
    eval judge, structured-output classifiers).

    Gemini 2.5 Flash Lite via OpenRouter — 4x cheaper on input, 8x cheaper
    on output than the main gemini-2.5-flash model. Note: NOT the
    discontinued 2.0-flash-lite-001 (which 404s on OpenRouter as of 2026).
    The judge LLM is intentionally pinned regardless of `LLM_MODEL` so eval
    scores stay comparable week-over-week.
    """
    llm = ChatOpenAI(
        model=settings.cheap_llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.3,
        max_tokens=1000,
        request_timeout=30,
        max_retries=1,
        http_async_client=_make_http_client(),
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Background Worker)",
        },
    )
    _wrap_with_retry(llm)
    return llm


@lru_cache(maxsize=1)
def get_preprocessor_llm() -> ChatOpenAI:
    """Cheap LLM for the pre-processor's intent classification + query rewrite.

    The pre-processor is the single most expensive per-call system in the
    pipeline (~1.0K input tokens on every turn) and runs BEFORE retrieval —
    so its cost is paid on every request including off-topic/off-scope
    queries that get short-circuited downstream.

    Routing it to Gemini 2.5 Flash Lite (4-8x cheaper than the main model)
    drops pre-processor cost by ~75% with no measurable impact on routing
    accuracy — the classification task is structured-output + a small set
    of intent labels, which Flash Lite handles reliably (verified against
    the safety benchmark: same intent + safety scores as the main model
    on 10/11 cases; only the known user-failure-case borderline ID-slang
    remains at S=0.5 either way).

    Temperature=0.0 + larger max_tokens than get_cheap_llm because the
    pre-processor needs a deterministic safety_preserved_query (longest
    output is ~200 chars).
    """
    llm = ChatOpenAI(
        model=settings.cheap_llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.0,
        max_tokens=500,
        request_timeout=30,
        max_retries=1,
        http_async_client=_make_http_client(),
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Pre-Processor)",
        },
    )
    _wrap_with_retry(llm)
    return llm


@lru_cache(maxsize=1)
def get_generate_llm() -> ChatOpenAI:
    """Cheap LLM for the final answer generation (generate / greeting /
    ambiguity nodes).

    This is the largest input cost in the pipeline (~1.6K tokens per call
    carrying the system prompt + RAG context + history summary). Routing
    it to Gemini 2.5 Flash Lite drops the per-turn cost by ~75-80% with
    no measurable quality regression on the verified cases:

      - Safety escalation: produces the right contact info (safespace email
        + WhatsApp) with a valid empathetic lead-in.
      - Brainstorm / vent: warm tone, lists concrete next steps.
      - Standard knowledge / procedural: bulleted structure preserved,
        contact details present.
      - 4-8x cheaper than gemini-2.5-flash on both input ($0.075 vs $0.30
        per 1M) and output ($0.30 vs $2.50 per 1M).

    The `eval` runner and the safety-aware grader still call the main
    `get_llm()` (flash) so the safety benchmark stays comparable
    week-over-week.

    Temperature=0.0 mirrors the main LLM config; max_tokens=600 leaves
    room for the longest typical brainstorm answer.
    """
    llm = ChatOpenAI(
        model=settings.cheap_llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.0,
        max_tokens=600,
        request_timeout=30,
        max_retries=1,
        http_async_client=_make_http_client(),
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Generate)",
        },
    )
    _wrap_with_retry(llm)
    return llm
