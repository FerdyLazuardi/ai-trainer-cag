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


async def clear_conversation_history(conversation_id: str) -> None:
    """
    Clear conversation history and summary for a given session.
    """
    if not conversation_id:
        return

    redis = get_redis_client()
    key = _conv_key(conversation_id)
    summary_key = f"rag:summary:{conversation_id}"
    last_active_key = f"rag:last_active:{conversation_id}"
    scheduled_key = f"rag:ltm:scheduled:{conversation_id}"

    try:
        await redis.delete(key, summary_key, last_active_key, scheduled_key)
        logger.info("Conversation history cleared", conversation_id=conversation_id)
    except Exception as exc:
        logger.warning("Failed to clear conversation history", error=str(exc))


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


def extract_follow_up_questions(assistant_message: str) -> list[str]:
    """
    Extract follow-up questions from assistant message.
    
    Parses the assistant message to find the "**Apa kamu penasaran tentang:**" section
    and extracts numbered questions (1., 2., 3.).
    
    Args:
        assistant_message: The assistant's response text
        
    Returns:
        List of question texts (without numbers). Empty list if no follow-ups found.
    """
    if not assistant_message:
        return []
    
    try:
        # Look for the follow-up section marker
        marker = "**Apa kamu penasaran tentang:**"
        if marker not in assistant_message:
            return []
        
        # Extract the section after the marker
        section_start = assistant_message.index(marker) + len(marker)
        section = assistant_message[section_start:]
        
        # Parse lines starting with "1.", "2.", "3."
        questions = []
        lines = section.split('\n')
        
        for line in lines:
            line = line.strip()
            # Check if line starts with "1.", "2.", or "3."
            for num in ['1.', '2.', '3.']:
                if line.startswith(num):
                    # Extract question text (remove the number prefix)
                    question_text = line[len(num):].strip()
                    if question_text:
                        questions.append(question_text)
                    break
        
        return questions
    except Exception as exc:
        logger.warning("Failed to extract follow-up questions", error=str(exc))
        return []


async def resolve_numeric_query(query: str, conversation_id: str) -> str:
    """
    Resolve numeric input (1, 2, 3) to corresponding follow-up question.
    
    If the query is a single digit (1-3) and the last assistant message contains
    follow-up questions, returns the corresponding full question text.
    Otherwise returns the query unchanged.
    
    Args:
        query: User's input query
        conversation_id: Conversation session ID
        
    Returns:
        Resolved question text if numeric input maps to a follow-up question,
        otherwise the original query unchanged.
    """
    # Check if query is numeric (1, 2, or 3)
    stripped_query = query.strip()
    if stripped_query not in ['1', '2', '3']:
        return query
    
    # Get conversation history
    history = await get_conversation_history(conversation_id)
    if not history:
        return query
    
    # Extract last assistant message
    last_assistant_message = None
    for message in reversed(history):
        if message.get("role") == "assistant":
            last_assistant_message = message.get("content", "")
            break
    
    if not last_assistant_message:
        return query
    
    # Extract follow-up questions
    follow_ups = extract_follow_up_questions(last_assistant_message)
    if not follow_ups:
        return query
    
    # Map numeric input to corresponding question (1→first, 2→second, 3→third)
    try:
        question_index = int(stripped_query) - 1  # Convert to 0-based index
        if 0 <= question_index < len(follow_ups):
            resolved = follow_ups[question_index]
            logger.info(
                "Resolved numeric query to follow-up question",
                original_query=query,
                resolved_query=resolved,
                conversation_id=conversation_id,
            )
            return resolved
        else:
            # Out of range - return original query
            return query
    except (ValueError, IndexError):
        return query


SUMMARY_TRIGGER_TURNS = 6  # summarize setelah 6 turns penuh (12 messages)

async def get_or_summarize_history(
    conversation_id: str,
    llm,
    max_fresh_turns: int = 5,
) -> tuple[str, list[dict]]:
    """
    Return (summary_str, recent_turns_list) using Rolling Batch Summarization.
    """
    history = await get_conversation_history(conversation_id)
    
    redis = get_redis_client()
    summary_key = f"rag:summary:{conversation_id}"

    # 1. Check if conversation length is within the fresh turn window.
    # Return full history and existing summary if available without invoking LLM.
    if len(history) // 2 <= max_fresh_turns:
        cached_summary = await redis.get(summary_key)
        return cached_summary or "", history

    # 2. Window exceeded: Trigger Batch Summarization.
    # Extract the oldest turns to be consolidated into the persistent summary.
    turns_to_summarize = history[:(max_fresh_turns * 2)]
    
    # Retain the most recent turn as the starting point for the next window.
    fresh_turns = history[(max_fresh_turns * 2):]

    # 3. Retrieve the existing consolidated summary.
    old_summary = await redis.get(summary_key) or ""

    # 4. Recursive Refinement: Merge existing summary with the overflowing turns.
    from langchain_core.messages import HumanMessage as HM
    old_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:300]}"
        for m in turns_to_summarize
    )
    
    prompt = (
        "Refine the following conversation summary to include the key points from the new dialogue segment. "
        "Maintain a concise, 2-3 sentence overview in the same language.\n\n"
        f"Existing Summary:\n{old_summary}\n\n"
        f"New context to integrate:\n{old_text}\n\n"
        "Updated Summary:"
    )
    
    resp = await llm.ainvoke([HM(content=prompt)])
    new_summary = resp.content.strip()

    # 5. Persist the updated consolidated summary to Redis.
    await redis.set(summary_key, new_summary, ex=settings.conversation_ttl_seconds)

    # 6. Update the conversation history in Redis by removing the summarized segments.
    key = _conv_key(conversation_id)
    await redis.set(key, json.dumps(fresh_turns), ex=settings.conversation_ttl_seconds)

    logger.info("Conversation rolling batch summary updated", conversation_id=conversation_id)
    return new_summary, fresh_turns
