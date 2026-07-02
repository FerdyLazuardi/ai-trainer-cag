from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_generate_llm_nostream


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
        provider=rm.get("provider_name") or rm.get("provider") or rm.get("model_name"),
    )


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
