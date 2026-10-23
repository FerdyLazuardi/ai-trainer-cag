"""
Redis schema for conversation state.

Collapses 6 separate keys per conversation into 1 HASH with per-field TTL
via Redis 8's HSETEX / HEXPIRE. Server requirement: Redis 8.0+ for HSETEX,
Redis 7.4+ for HEXPIRE (HSETEX itself is 8.0+).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from loguru import logger
from redis.exceptions import WatchError

from app.config.settings import get_settings

Redis = Any

settings = get_settings()

# ── Constants ────────────────────────────────────────────────────────────────
_CONV_KEY_PREFIX = "rag:conv:"
_LTM_LOCK_PREFIX = "rag:ltm:syncing:"
_SUMMARY_REFRESH_PREFIX = "rag:sumref:"   # NX dedup lock for async summary refresh
_MAX_WATCH_RETRIES = 5
_recent_history_len: dict[str, int] = {}


def _conv_key(conv_id: str) -> str:
    return f"{_CONV_KEY_PREFIX}{conv_id}"


def _ltm_lock_key(conv_id: str) -> str:
    return f"{_LTM_LOCK_PREFIX}{conv_id}"


def _key_ttl() -> int:
    """Key-level EXPIRE applied on EVERY write (C6)."""
    return settings.conversation_ttl_seconds


async def _incr_stm_append_failure_metric(redis: Redis) -> None:
    """Fire-and-forget counter for append_to_history failures (C6)."""
    try:
        from datetime import datetime, timezone

        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        key = f"rag:metrics:stm_append_failure:{day}"
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 7 * 24 * 3600)
        await pipe.execute()
    except Exception:
        pass


# ── Read ─────────────────────────────────────────────────────────────────────
async def get_history(redis: Redis, conv_id: str) -> list[dict]:
    """Hot-path history read. Returns [] if missing or corrupt."""
    if not conv_id:
        return []
    try:
        raw = await redis.hget(_conv_key(conv_id), "history")
        return json.loads(raw) if raw else []
    except (json.JSONDecodeError, TypeError):
        return []
    except Exception:
        return []


async def get_summary(redis: Redis, conv_id: str) -> str:
    """Returns "" if absent."""
    if not conv_id:
        return ""
    try:
        return (await redis.hget(_conv_key(conv_id), "summary")) or ""
    except Exception:
        return ""


async def get_history_and_summary(
    redis: Redis, conv_id: str
) -> tuple[list[dict], str]:
    """Single-RTT HMGET for the chat handler's gather() hot path."""
    if not conv_id:
        return [], ""
    try:
        raw_h, raw_s = await redis.hmget(
            _conv_key(conv_id), "history", "summary"
        )
        history = json.loads(raw_h) if raw_h else []
        summary = raw_s or ""
        if not isinstance(history, list):
            history = []
        return history, summary
    except (json.JSONDecodeError, TypeError):
        return [], ""
    except Exception:
        return [], ""


async def get_last_active(redis: Redis, conv_id: str) -> float | None:
    if not conv_id:
        return None
    try:
        raw = await redis.hget(_conv_key(conv_id), "last_active")
        return float(raw) if raw else None
    except (ValueError, TypeError):
        return None
    except Exception:
        return None


async def is_scheduled(redis: Redis, conv_id: str) -> bool:
    if not conv_id:
        return False
    try:
        return bool(await redis.hget(_conv_key(conv_id), "scheduled"))
    except Exception:
        return False


async def get_courses(redis: Redis, conv_id: str) -> list[str]:
    """Returns deduped, sorted list. Empty list if absent."""
    if not conv_id:
        return []
    try:
        raw = await redis.hget(_conv_key(conv_id), "courses")
        if not raw:
            return []
        names = json.loads(raw)
        if not isinstance(names, list):
            return []
        return sorted({
            n.strip() for n in names
            if isinstance(n, str) and n.strip()
        })
    except (json.JSONDecodeError, TypeError):
        return []
    except Exception:
        return []


# ── Write ────────────────────────────────────────────────────────────────────
async def get_seen_chunk_ids(redis: Redis, conv_id: str) -> set[str]:
    if not conv_id:
        return set()
    try:
        raw = await redis.hget(_conv_key(conv_id), "seen_chunks")
        values = json.loads(raw) if raw else []
        if not isinstance(values, list):
            return set()
        return {str(v) for v in values if v}
    except (json.JSONDecodeError, TypeError):
        return set()
    except Exception:
        return set()


async def append_to_history(
    redis: Redis, conv_id: str,
    user_message: str, assistant_message: str,
    *,
    max_turns: int = 10,
) -> int:
    """Atomic field+TTL set via WATCH+HGET+HSETEX."""
    if not conv_id:
        return 0
    key = _conv_key(conv_id)
    ttl = settings.conversation_ttl_seconds
    max_msgs = max_turns * 2
    user_msg = {"role": "user", "content": user_message}
    asst_msg = {"role": "assistant", "content": assistant_message}

    for _attempt in range(_MAX_WATCH_RETRIES):
        try:
            async with redis.pipeline(transaction=True) as pipe:
                await pipe.watch(key)
                raw = await pipe.hget(key, "history")
                try:
                    history = json.loads(raw) if raw else []
                    if not isinstance(history, list):
                        history = []
                except (json.JSONDecodeError, TypeError):
                    history = []
                history.append(user_msg)
                history.append(asst_msg)
                if len(history) > max_msgs:
                    history = history[-max_msgs:]
                pipe.multi()
                pipe.hsetex(
                    key,
                    mapping={"history": json.dumps(history)},
                    ex=ttl,
                )
                pipe.expire(key, _key_ttl())
                await pipe.execute()
                return len(history)
        except WatchError:
            continue
        except Exception:
            await _incr_stm_append_failure_metric(redis)
            return 0
    logger.warning(
        "STM append_turn exhausted WATCH retries; dropping turn",
        conv_id=conv_id,
        attempts=_MAX_WATCH_RETRIES,
    )
    await _incr_stm_append_failure_metric(redis)
    return 0


async def set_summary(redis: Redis, conv_id: str, summary: str) -> None:
    """Single-field summary write + key-level EXPIRE (C6)."""
    if not conv_id:
        return
    key = _conv_key(conv_id)
    try:
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hsetex(
                key,
                mapping={"summary": summary},
                ex=settings.conversation_ttl_seconds,
            )
            pipe.expire(key, _key_ttl())
            await pipe.execute()
    except Exception:
        pass


async def _persist_summary_and_trim(
    redis: Redis,
    conv_id: str,
    *,
    new_summary: str,
    summarized_head: list[dict],
    head_len: int,
) -> None:
    """Atomically persist the refreshed summary AND trim the summarized head
    off `history`, under WATCH/MULTI so a concurrent append_to_history is
    never clobbered (C7)."""
    if not conv_id:
        return
    key = _conv_key(conv_id)
    ttl = settings.conversation_ttl_seconds
    for _attempt in range(_MAX_WATCH_RETRIES):
        try:
            async with redis.pipeline(transaction=True) as pipe:
                await pipe.watch(key)
                raw = await pipe.hget(key, "history")
                try:
                    cur = json.loads(raw) if raw else []
                    if not isinstance(cur, list):
                        cur = []
                except (json.JSONDecodeError, TypeError):
                    cur = []

                do_trim = len(cur) >= head_len and cur[:head_len] == summarized_head
                mapping: dict[str, str] = {"summary": new_summary}
                if do_trim:
                    mapping["history"] = json.dumps(cur[head_len:])

                pipe.multi()
                pipe.hsetex(key, mapping=mapping, ex=ttl)
                pipe.expire(key, _key_ttl())
                await pipe.execute()
                return
        except WatchError:
            continue
        except Exception:
            return
    logger.warning(
        "STM summary persist exhausted WATCH retries", conv_id=conv_id
    )


async def set_last_active(redis: Redis, conv_id: str) -> None:
    if not conv_id:
        return
    key = _conv_key(conv_id)
    try:
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hsetex(
                key,
                mapping={"last_active": str(time.time())},
                ex=settings.ltm_afk_threshold_seconds + 3600,
            )
            pipe.expire(key, _key_ttl())
            await pipe.execute()
    except Exception:
        pass


async def set_scheduled(redis: Redis, conv_id: str) -> None:
    if not conv_id:
        return
    key = _conv_key(conv_id)
    try:
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hsetex(
                key,
                mapping={"scheduled": "1"},
                ex=settings.ltm_afk_threshold_seconds + 600,
            )
            pipe.expire(key, _key_ttl())
            await pipe.execute()
    except Exception:
        pass


async def add_courses(
    redis: Redis, conv_id: str, names: set[str]
) -> None:
    """Read-merge-write the courses field."""
    if not conv_id or not names:
        return
    key = _conv_key(conv_id)
    ttl = settings.ltm_afk_threshold_seconds + 3600
    try:
        raw = await redis.hget(key, "courses")
        try:
            existing = set(json.loads(raw)) if raw else set()
            if not isinstance(existing, set):
                existing = set(existing) if hasattr(existing, "__iter__") else set()
        except (json.JSONDecodeError, TypeError):
            existing = set()
        existing = {n for n in existing if isinstance(n, str) and n.strip()}
        merged = sorted(existing | names)
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hsetex(key, mapping={"courses": json.dumps(merged)}, ex=ttl)
            pipe.expire(key, _key_ttl())
            await pipe.execute()
    except Exception:
        pass


async def add_seen_chunk_ids(
    redis: Redis, conv_id: str, chunk_ids: list[str], *, max_ids: int = 200
) -> None:
    if not conv_id or not chunk_ids:
        return
    key = _conv_key(conv_id)
    try:
        raw = await redis.hget(key, "seen_chunks")
        try:
            existing = json.loads(raw) if raw else []
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, TypeError):
            existing = []

        merged = list(dict.fromkeys([str(x) for x in existing + chunk_ids if x]))
        merged = merged[-max_ids:]
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hsetex(key, mapping={"seen_chunks": json.dumps(merged)}, ex=_key_ttl())
            pipe.expire(key, _key_ttl())
            await pipe.execute()
    except Exception:
        pass


async def bump_topic_streak(
    redis: Redis, conv_id: str, topic: str, *, threshold: int
) -> str | None:
    """Track consecutive same-topic turns in the conversation HASH."""
    if not conv_id or not topic or not topic.strip():
        return None
    topic = topic.strip()
    key = _conv_key(conv_id)
    try:
        raw = await redis.hget(key, "streak")
        st = json.loads(raw) if raw else {}
        cur_t_raw = st.get("t")
        cur_t = cur_t_raw if isinstance(cur_t_raw, str) else ""
        cur_n = int(st.get("n", 0))

        if cur_t == topic:
            new_t, new_n = topic, cur_n + 1
        else:
            decayed = max(0, cur_n - 1)
            new_t, new_n = (cur_t, decayed) if decayed > 0 and cur_t else (topic, 1)

        reached = new_n >= threshold
        write_val: dict[str, str | int | None] = (
            {"t": None, "n": 0} if reached else {"t": new_t, "n": new_n}
        )
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hsetex(key, mapping={"streak": json.dumps(write_val)}, ex=_key_ttl())
            pipe.expire(key, _key_ttl())
            await pipe.execute()
        return new_t if reached else None
    except Exception:
        return None


# ── Compound operations ──────────────────────────────────────────────────────
async def schedule_afk_sync(
    redis: Redis, conv_id: str, user_id: str
) -> bool:
    """Combines: set_last_active + is_scheduled check + (if not scheduled)
    enqueue sync_ltm_task + set_scheduled.

    Gated on `settings.ltm_enabled`: returns False immediately with zero Redis
    or worker overhead when LTM is disabled.
    """
    if not settings.ltm_enabled or not conv_id or not user_id:
        return False
    await set_last_active(redis, conv_id)
    if await is_scheduled(redis, conv_id):
        return False
    afk_seconds = settings.ltm_afk_threshold_seconds
    from app.worker import sync_ltm_task

    try:
        await sync_ltm_task.enqueue(conv_id, user_id).start(delay=afk_seconds, priority="high")
    except Exception as e:
        logger.warning(f"Failed to enqueue AFK LTM sync: {e}")
        return False
    await set_scheduled(redis, conv_id)
    return True


async def clear_conversation(redis: Redis, conv_id: str) -> None:
    """DEL the single HASH key."""
    if not conv_id:
        return
    _recent_history_len.pop(conv_id, None)
    try:
        await redis.delete(_conv_key(conv_id))
    except Exception:
        pass


# ── High-level history operations ────────────────────────────────────────────
SUMMARY_TRIGGER_TURNS = 6


def extract_follow_up_questions(assistant_message: str) -> list[str]:
    """Extract numbered follow-up questions from an assistant message."""
    if not assistant_message:
        return []
    try:
        match = re.search(
            r'penasaran tentang.*?:?\*?\*?\s*(.*)',
            assistant_message,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            lines = match.group(1).strip().split('\n')
        else:
            lines = assistant_message.strip().split('\n')[-10:]

        questions = []
        for line in lines:
            line = line.strip()
            match_num = re.match(r'^(?:\*\*)?([1-5])[\.\)]\s*(?:\*\*)?(.*)', line)
            if match_num:
                questions.append(match_num.group(2).strip())
        return questions
    except Exception as exc:
        logger.warning("Failed to extract follow-up questions", error=str(exc))
        return []


async def resolve_numeric_query(
    redis: Redis, query: str, conv_id: str
) -> str:
    """Resolve numeric input (1, 2, 3) to the matching follow-up question."""
    stripped_query = query.strip()
    if not re.fullmatch(r"[1-5]", stripped_query):
        return query

    history = await get_history(redis, conv_id)
    if not history:
        return query

    last_assistant_message = None
    for message in reversed(history):
        if message.get("role") == "assistant":
            last_assistant_message = message.get("content", "")
            break

    if not last_assistant_message:
        return query

    follow_ups = extract_follow_up_questions(last_assistant_message)
    if not follow_ups:
        return query

    try:
        question_index = int(stripped_query) - 1
        if 0 <= question_index < len(follow_ups):
            resolved = follow_ups[question_index]
            logger.info(
                "Resolved numeric query to follow-up question",
                original_query=query,
                resolved_query=resolved,
                conversation_id=conv_id,
            )
            return resolved
        return query
    except (ValueError, IndexError):
        return query


async def get_or_summarize_history(
    redis: Redis,
    conv_id: str,
    llm,
    max_fresh_turns: int = settings.max_fresh_turns,
    *,
    persist: bool = True,
) -> tuple[str, list[dict]]:
    """Return (summary_str, recent_turns_list) using Rolling Batch Summarization."""
    history, cached_summary = await get_history_and_summary(redis, conv_id)
    if conv_id:
        if len(_recent_history_len) >= 4096:
            _recent_history_len.clear()
        _recent_history_len[conv_id] = len(history)

    # 1. Within the fresh-turn window: no LLM call, return as-is.
    if len(history) // 2 <= max_fresh_turns:
        return cached_summary or "", history

    # 2. Window exceeded.
    turns_to_summarize = history[:(max_fresh_turns * 2)]
    fresh_turns = history[(max_fresh_turns * 2):]
    old_summary = cached_summary or ""

    # 2a. READ-ONLY caller: return cached summary + fresh slice, no LLM/write.
    if not persist:
        return old_summary, fresh_turns

    # 3. Recursive refinement: merge existing summary with overflowing turns.
    from langchain_core.messages import HumanMessage as HM
    old_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:300]}"
        for m in turns_to_summarize
    )
    from app.llm.prompts import STM_SUMMARY_PROMPT

    prompt = STM_SUMMARY_PROMPT.format(old_summary=old_summary, old_text=old_text)

    try:
        resp = await llm.ainvoke(
            [HM(content=prompt)],
            config={"run_name": "ava-rolling-summarization"},
        )
        new_summary = resp.content.strip()
    except Exception as exc:
        logger.warning(f"Failed to generate batch summary: {exc}")
        new_summary = old_summary

    await _persist_summary_and_trim(
        redis,
        conv_id,
        new_summary=new_summary,
        summarized_head=turns_to_summarize,
        head_len=max_fresh_turns * 2,
    )

    logger.info(
        "Conversation rolling batch summary updated", conversation_id=conv_id
    )
    return new_summary, fresh_turns


async def schedule_summary_refresh(
    redis: Redis,
    conv_id: str,
    *,
    max_fresh_turns: int = settings.max_fresh_turns,
    history_len: int | None = None,
) -> bool:
    """Enqueue an out-of-band summary refresh IF the conversation has overflowed
    the fresh-turn window (C7). Reuses in-memory length from `get_or_summarize_history`
    when available to avoid a duplicate Redis HGET + JSON parse on every turn."""
    if not conv_id:
        return False
    try:
        n = history_len if history_len is not None else _recent_history_len.pop(conv_id, -1)
        if n < 0:
            raw = await redis.hget(_conv_key(conv_id), "history")
            try:
                n = len(json.loads(raw)) if raw else 0
            except (json.JSONDecodeError, TypeError):
                n = 0
        if n // 2 <= max_fresh_turns:
            return False
        lock_key = f"{_SUMMARY_REFRESH_PREFIX}{conv_id}"
        if not await redis.set(lock_key, "1", nx=True, ex=30):
            return False
        from app.worker import summarize_refresh_task

        await summarize_refresh_task.enqueue(conv_id).start(priority="high")
        return True
    except Exception as exc:
        logger.warning(f"Failed to schedule summary refresh: {exc}")
        return False


# ── LTM sync lock (kept as separate STRING key, NOT in the HASH) ────────────
async def acquire_ltm_lock(redis: Redis, conv_id: str) -> bool:
    """SET NX EX 300. Returns True if acquired, False if already held."""
    if not conv_id:
        return False
    try:
        got = await redis.set(_ltm_lock_key(conv_id), "1", nx=True, ex=300)
        return bool(got)
    except Exception:
        return False


async def release_ltm_lock(redis: Redis, conv_id: str) -> None:
    if not conv_id:
        return
    try:
        await redis.delete(_ltm_lock_key(conv_id))
    except Exception:
        pass
