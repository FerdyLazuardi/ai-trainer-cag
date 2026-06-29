"""
OpenRouter LLM client via LangChain's ChatOpenAI integration.
"""
from functools import lru_cache

import httpx
from langchain_openai import ChatOpenAI
from loguru import logger

from app.config.settings import get_settings

settings = get_settings()


_LLM_USER_AGENT = "ai-lms-agent/1.0"


async def _strip_openai_ua(request: httpx.Request) -> None:
    request.headers["User-Agent"] = _LLM_USER_AGENT


@lru_cache(maxsize=1)
def _shared_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        event_hooks={"request": [_strip_openai_ua]},
    )


def _provider_extra_body(model: str, *, include_usage: bool = True) -> dict:
    body: dict = {}
    if include_usage:
        body["usage"] = {"include": True}
    m = model.lower()
    
    # As per settings.py, Gemini stays pinned to google-vertex since it has
    # implicit cache wins and AI Studio auto-routing has been hitting rate limits
    # causing 10-20s latency spikes. allow_fallbacks=False: when vertex 429s,
    # do NOT fall through to google-ai-studio — ai-studio corrupts the SSE
    # stream mid-generation ("JSON error injected into SSE stream"), which
    # with streaming=True kills the reply at 0 tokens. A clean vertex 429
    # surfaces as a retry-able error (generate_node's ainvoke retry catches
    # it) instead of a silent mid-stream abort.
    if m.startswith("google/"):
        body["provider"] = {"order": ["google-vertex"], "allow_fallbacks": False}

    elif m.startswith("deepseek/") or m.startswith("xiaomi/"):
        main_model = settings.llm_model.lower()
        is_main_family = (
            (m.startswith("deepseek/") and main_model.startswith("deepseek/")) or
            (m.startswith("xiaomi/") and main_model.startswith("xiaomi/"))
        )
        if is_main_family:
            default_provider = "xiaomi" if m.startswith("xiaomi/") else "deepseek"
            order = settings.llm_provider_order_list or [default_provider]
        else:
            order = ["xiaomi"] if m.startswith("xiaomi/") else ["deepseek"]

        body["provider"] = {"order": order, "allow_fallbacks": True}
        if m.startswith("deepseek/"):
            body["reasoning"] = {"effort": "none", "exclude": True}
    return body


def create_llm(
    model: str,
    temperature: float,
    max_tokens: int,
    extra_body: dict | None = None,
    default_headers: dict | None = None,
    model_kwargs: dict | None = None,
    streaming: bool = False,
    stream_usage: bool = True,
    request_timeout: int = 60,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        max_retries=3,
        http_async_client=_shared_http_client(),
        streaming=streaming,
        stream_usage=stream_usage,
        extra_body=extra_body or _provider_extra_body(model),
        default_headers=default_headers or {},
        model_kwargs=model_kwargs or {},
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return create_llm(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent",
        },
    )


@lru_cache(maxsize=1)
def get_cheap_llm() -> ChatOpenAI:
    return create_llm(
        model=settings.cheap_llm_model,
        temperature=settings.cheap_llm_temperature,
        max_tokens=1000,
        request_timeout=30,
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Background Worker)",
        },
    )


@lru_cache(maxsize=1)
def get_judge_llm() -> ChatOpenAI:
    return create_llm(
        model=settings.judge_llm_model,
        temperature=settings.judge_llm_temperature,
        max_tokens=1024,
        model_kwargs={"response_format": {"type": "json_object"}},
        extra_body=_provider_extra_body(settings.judge_llm_model),
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Judge)",
        },
    )


def assert_judge_model_distinct(
    judge_model: str, generator_models: set[str]
) -> None:
    if judge_model in generator_models:
        raise RuntimeError(
            f"JUDGE_LLM_MODEL={judge_model!r} matches a generator model "
            f"({sorted(generator_models)!r}). The eval judge must be a "
            f"different model family than the generator — a same-family judge "
            f"shares fabrication patterns and undercounts the hallucination "
            f"rate (judge grading itself). Set JUDGE_LLM_MODEL to a distinct "
            f"model (default: deepseek/deepseek-v4-pro)."
        )


@lru_cache(maxsize=1)
def get_preprocessor_llm() -> ChatOpenAI:
    return create_llm(
        model=settings.cheap_llm_model,
        temperature=settings.preprocessor_llm_temperature,
        max_tokens=500,
        request_timeout=30,
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Pre-Processor)",
        },
    )


@lru_cache(maxsize=1)
def get_generate_llm() -> ChatOpenAI:
    return create_llm(
        model=settings.llm_model,
        temperature=settings.generate_llm_temperature,
        max_tokens=settings.llm_max_tokens,
        request_timeout=60,
        streaming=False,
        default_headers={
            "HTTP-Referer": "https://github.com/peped-BE",
            "X-Title": "AI LMS RAG Agent (Generate)",
        },
    )


@lru_cache(maxsize=1)
def get_chat_llm() -> ChatOpenAI:
    return create_llm(
        model=settings.llm_model,
        temperature=settings.chat_llm_temperature,
        max_tokens=settings.llm_max_tokens,
        request_timeout=60,
        streaming=False,
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Chat)",
        },
    )


@lru_cache(maxsize=1)
def get_empathy_llm() -> ChatOpenAI:
    return create_llm(
        model=settings.llm_model,
        temperature=settings.empathy_llm_temperature,
        max_tokens=settings.llm_max_tokens,
        request_timeout=30,
        streaming=True,
        stream_usage=True,
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Empathy)",
        },
    )
