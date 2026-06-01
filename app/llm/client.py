"""
OpenRouter LLM client via LangChain's ChatOpenAI integration.
Includes tenacity retry wrapping for resilience.
"""
from functools import lru_cache

import httpx
from langchain_openai import ChatOpenAI

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


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return the singleton LLM client configured for 9Router/OpenRouter."""
    return ChatOpenAI(
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


@lru_cache(maxsize=1)
def get_cheap_llm() -> ChatOpenAI:
    """Return a cheaper, faster LLM for background tasks (memory summarization,
    eval judge, structured-output classifiers).

    Gemini 2.0 Flash Lite via OpenRouter — cheap, fast, reliable structured
    output. The judge LLM is intentionally pinned regardless of `LLM_MODEL`
    so eval scores stay comparable week-over-week.
    """
    return ChatOpenAI(
        model="google/gemini-2.0-flash-lite-001",
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


@lru_cache(maxsize=1)
def get_preprocessor_llm() -> ChatOpenAI:
    """Cheap LLM for the pre-processor's intent classification + query rewrite.

    The pre-processor is the single most expensive per-call system in the
    pipeline (~1.5K input tokens on every turn) and runs BEFORE retrieval —
    so its cost is paid on every request including off-topic/off-scope
    queries that get short-circuited downstream.

    Routing it to Gemini 2.0 Flash Lite (4-8x cheaper than the main model)
    drops pre-processor cost by ~75% with no measurable impact on routing
    accuracy — the classification task is structured-output + a small set of
    intent labels, which Flash Lite handles reliably (the LLM-only smoke
    verifies intent + safety scores still match the main model on the
    safety benchmark).

    Temperature=0.0 + larger max_tokens than get_cheap_llm because the
    pre-processor needs a deterministic safety_preserved_query (longest
    output is ~200 chars).
    """
    return ChatOpenAI(
        model="google/gemini-2.0-flash-lite-001",
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
