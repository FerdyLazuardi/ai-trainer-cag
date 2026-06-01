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

    Gemini 2.5 Flash Lite via OpenRouter — 4x cheaper on input, 8x cheaper
    on output than the main gemini-2.5-flash model. Note: NOT the
    discontinued 2.0-flash-lite-001 (which 404s on OpenRouter as of 2026).
    The judge LLM is intentionally pinned regardless of `LLM_MODEL` so eval
    scores stay comparable week-over-week.
    """
    return ChatOpenAI(
        model="google/gemini-2.5-flash-lite",
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
    return ChatOpenAI(
        model="google/gemini-2.5-flash-lite",
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
    return ChatOpenAI(
        model="google/gemini-2.5-flash-lite",
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
