import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import get_settings
from app.llm.client import get_generate_llm_nostream

settings = get_settings()


CAG_SYSTEM_PROMPT = (
    "You are Ava, Amartha's LMS assistant. Answer only from the knowledge base "
    "provided in the next message. If the answer is not there, say that the "
    "material has not been found yet and ask the user to confirm with their BM."
)


@dataclass(frozen=True)
class OpenRouterUsage:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    completion_tokens: int = 0
    provider: str | None = None
    cost: float = 0.0
    generation_id: str | None = None


def build_cag_messages(
    *,
    kb_pack_text: str,
    user_query: str,
    user_context: dict | None = None,
    conversation_summary: str | None = None,
    user_preferences: dict | None = None,
    system_prompt: str = CAG_SYSTEM_PROMPT,
) -> list:
    messages: list = [SystemMessage(content=system_prompt), SystemMessage(content=kb_pack_text)]
    dynamic = _dynamic_context(
        user_context=user_context,
        conversation_summary=conversation_summary,
        user_preferences=user_preferences,
    )
    if dynamic:
        messages.append(HumanMessage(content=dynamic))
    messages.append(HumanMessage(content=user_query))
    return messages


async def generate_cag_answer(
    *,
    kb_pack_text: str,
    user_query: str,
    user_context: dict | None = None,
    conversation_summary: str | None = None,
    user_preferences: dict | None = None,
):
    return await get_generate_llm_nostream().ainvoke(
        build_cag_messages(
            kb_pack_text=kb_pack_text,
            user_query=user_query,
            user_context=user_context,
            conversation_summary=conversation_summary,
            user_preferences=user_preferences,
        )
    )


def extract_openrouter_usage(message: Any) -> OpenRouterUsage:
    rm = getattr(message, "response_metadata", None) or {}
    usage = rm.get("token_usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    um = getattr(message, "usage_metadata", None) or {}
    input_details = um.get("input_token_details") or {}

    return OpenRouterUsage(
        prompt_tokens=int(usage.get("prompt_tokens") or um.get("input_tokens") or 0),
        cached_tokens=int(details.get("cached_tokens") or input_details.get("cache_read") or 0),
        cache_write_tokens=int(details.get("cache_write_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or um.get("output_tokens") or 0),
        provider=rm.get("model_name") or rm.get("model") or rm.get("provider_name") or rm.get("provider"),
        cost=float(usage.get("cost") or 0.0),
        generation_id=rm.get("id") or rm.get("generation_id"),
    )


async def fetch_openrouter_generation_usage(
    generation_id: str | None,
    *,
    client: httpx.AsyncClient | None = None,
    sleep=asyncio.sleep,
) -> OpenRouterUsage:
    generation_id = (generation_id or "").strip()
    if not generation_id or not settings.openrouter_api_key:
        return OpenRouterUsage(generation_id=generation_id or None)

    async def _get(c):
        return await c.get(
            settings.openrouter_base_url.rstrip("/") + "/generation",
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            params={"id": generation_id},
            timeout=10,
        )

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient()

    try:
        for delay in (0, 0.25, 0.75):
            if delay:
                await sleep(delay)
            response = await _get(client)
            if response.status_code == 404:
                continue
            if response.status_code != 200:
                return OpenRouterUsage(generation_id=generation_id)
            data = (response.json() or {}).get("data") or {}
            return OpenRouterUsage(
                prompt_tokens=int(data.get("tokens_prompt") or data.get("native_tokens_prompt") or 0),
                cached_tokens=int(data.get("native_tokens_cached") or 0),
                completion_tokens=int(data.get("tokens_completion") or data.get("native_tokens_completion") or 0),
                provider=data.get("model") or data.get("provider_name"),
                cost=float(data.get("total_cost") or data.get("usage") or 0.0),
                generation_id=data.get("id") or generation_id,
            )
    finally:
        if owns_client:
            await client.aclose()

    return OpenRouterUsage(generation_id=generation_id)


def _dynamic_context(
    *,
    user_context: dict | None,
    conversation_summary: str | None,
    user_preferences: dict | None,
) -> str:
    lines: list[str] = []
    if user_context:
        clean = {k: v for k, v in user_context.items() if v}
        if clean:
            lines.append("<user_context>")
            lines.extend(f"{k}: {v}" for k, v in clean.items())
            lines.append("</user_context>")
    if user_preferences:
        clean = {k: v for k, v in user_preferences.items() if v}
        if clean:
            lines.append("<user_preferences>")
            lines.extend(f"{k}: {v}" for k, v in clean.items())
            lines.append("</user_preferences>")
    if conversation_summary:
        lines.extend(["<previous_context>", conversation_summary, "</previous_context>"])
    return "\n".join(lines)
