"""
OpenRouter LLM client via LangChain's ChatOpenAI integration.
Includes tenacity retry wrapping for resilience.
"""
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config.settings import get_settings

settings = get_settings()


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
        # OpenRouter-specific headers for app attribution (ignored by Ollama)
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent",
        },
    )


@lru_cache(maxsize=1)
def get_cheap_llm() -> ChatOpenAI:
    """Return a cheaper, faster LLM for background tasks like memory summarization."""
    # Using Gemini 2.5 Flash via 9Router for cost-efficiency
    return ChatOpenAI(
        model="google/gemini-2.0-flash-lite-001",
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.3,
        max_tokens=1000,
        request_timeout=30,
        max_retries=1,
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Background Worker)",
        },
    )
