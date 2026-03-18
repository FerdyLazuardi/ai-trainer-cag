"""
Redis-backed conversation memory.
Stores and retrieves conversation history as a JSON list per conversation_id.
"""
import json

from loguru import logger

from app.config.settings import get_settings
from app.database.redis_client import get_redis_client

settings = get_settings()

_PREFIX = "rag:conv:"


def _conv_key(conversation_id: str) -> str:
    return f"{_PREFIX}{conversation_id}"


async def get_conversation_history(conversation_id: str) -> list[dict]:
    """
    Retrieve conversation history for a given session.

    Returns:
        List of message dicts: [{"role": "user"|"assistant", "content": "..."}]
    """
    if not conversation_id:
        return []

    redis = get_redis_client()
    key = _conv_key(conversation_id)

    try:
        raw = await redis.get(key)
        if raw:
            return json.loads(raw)
        return []
    except Exception as exc:
        logger.warning("Failed to read conversation history", error=str(exc))
        return []


async def append_to_history(
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    max_turns: int = 10,
) -> None:
    """
    Append a user/assistant exchange to the conversation history.

    Maintains a rolling window of max_turns exchanges.
    """
    if not conversation_id:
        return

    redis = get_redis_client()
    key = _conv_key(conversation_id)
    ttl = settings.conversation_ttl_seconds

    try:
        history = await get_conversation_history(conversation_id)

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})

        # Keep only last max_turns * 2 messages
        if len(history) > max_turns * 2:
            history = history[-(max_turns * 2):]

        await redis.set(key, json.dumps(history), ex=ttl)
        logger.debug(
            "Conversation updated",
            conversation_id=conversation_id,
            turns=len(history) // 2,
        )
    except Exception as exc:
        logger.warning("Failed to update conversation history", error=str(exc))
