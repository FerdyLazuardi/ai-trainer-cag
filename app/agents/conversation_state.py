"""
Redis schema for conversation state.

Collapses 6 separate keys per conversation into 1 HASH with per-field TTL
via Redis 8's HSETEX / HEXPIRE. Server requirement: Redis 8.0+ for HSETEX,
Redis 7.4+ for HEXPIRE (HSETEX itself is 8.0+).

The transient `rag:ltm:syncing:{id}` lock is kept as a separate STRING
key (not a HASH field) because it must outlive the conversation HASH —
a slow LTM worker can fire after the HASH expires.

HASH schema (`rag:conv:{conversation_id}`) — per-field TTL via HSETEX:
    history      JSON list       12h (conversation_ttl_seconds)
    summary      str             12h (conversation_ttl_seconds)
    last_active  str (epoch)     afk + 3600s
    scheduled    "1"             afk + 600s
    courses      JSON list       afk + 3600s
    owner        str             no field TTL — gated by key-level EXPIRE only

Ownership (B1): the `owner` field stores the authenticated user_id allowed to
read this conversation. It is written WITHOUT a per-field TTL, so within the
HASH it can only ever outlive the history fields, never the reverse. Because
Redis eviction is atomic at the key level (never field-by-field), owner and
history are evicted TOGETHER — closing the prior fail-open window where a
separate `rag:conv_owner:{id}` STRING could be LRU-evicted while the history
HASH survived, letting the next requester re-claim and read it. This reverses
4.3 (which had moved owner OUT to a STRING); the get/set_owner helpers stay
removed — the chat route reads/writes the field inline via HGET/HSETNX.

Note: HSETEX `EX seconds` sets PER-FIELD TTLs, NOT a top-level key TTL.
A HASH carrying ONLY field-level (HEXPIRE) TTLs has no key-level EXPIRE, and
`volatile-lru` (our maxmemory-policy) only evicts keys that have a key-level
expire set — so such a HASH would be NON-evictable and could drive the
instance OOM on write under pressure (C6). To keep every conversation HASH
eviction-eligible, every write path also sets a KEY-LEVEL EXPIRE of
`conversation_ttl_seconds` (12h = the longest field TTL, so it never kills a
still-live field early). The key dies wholesale at 12h; individual fields may
expire sooner via their own HEXPIRE. Re-writing a field refreshes both the
field TTL and the key-level EXPIRE.
"""
from __future__ import annotations

import json
import re
import time

from loguru import logger
# RedisLike (a Protocol typing the async client's methods as pure coroutines)
# is imported under the name `Redis` so the 21 `redis: Redis` param annotations
# below resolve to it WITHOUT touching each line. Fixes the "X is not awaitable"
# checker noise at its root (see app/database/redis_client.py docstring). This
# is annotation-only here — no Redis() instantiation / isinstance — so the
# alias is behavior-neutral.
from app.database.redis_client import RedisLike as Redis
from redis.exceptions import WatchError

from app.config.settings import get_settings

settings = get_settings()

# ── Constants ────────────────────────────────────────────────────────────────
_CONV_KEY_PREFIX = "rag:conv:"
_LTM_LOCK_PREFIX = "rag:ltm:syncing:"
_SUMMARY_REFRESH_PREFIX = "rag:sumref:"   # NX dedup lock for async summary refresh
_MAX_WATCH_RETRIES = 5


def _conv_key(conv_id: str) -> str:
    return f"{_CONV_KEY_PREFIX}{conv_id}"


def _ltm_lock_key(conv_id: str) -> str:
    return f"{_LTM_LOCK_PREFIX}{conv_id}"


def _key_ttl() -> int:
    """Key-level EXPIRE applied on EVERY write (C6).

    Always the longest field TTL (`conversation_ttl_seconds`, 12h) regardless
    of which field is being written, so the key-level EXPIRE never truncates a
    still-live longer field (e.g. a `set_last_active` write with an 11h field
    TTL must NOT pull the 12h `history` field's key down to 11h). This makes
    the HASH eviction-eligible under `volatile-lru` without ever shortening a
    field's effective lifetime.
    """
    return settings.conversation_ttl_seconds


async def _incr_stm_append_failure_metric(redis: Redis) -> None:
    """Fire-and-forget counter for append_to_history failures (C6).

    Mirrors pipeline._incr_parse_failure_metric: UTC-day-bucketed, self-
    expiring, never raises. A rising count means turns are being silently
    dropped from STM (WRONGTYPE legacy keys, WATCH exhaustion, or Redis
    errors) — i.e. users losing conversation memory.
    """
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
    """Single-RTT HMGET for the chat handler's gather() hot path.
    Saves 1 RTT vs separate HGETs for history and summary."""
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
async def append_to_history(
    redis: Redis, conv_id: str,
    user_message: str, assistant_message: str,
    *,
    max_turns: int = 10,
) -> int:
    """Atomic field+TTL set via WATCH+HGET+HSETEX. Replaces the Lua script.

    Returns the new history length. Retries once on WatchError (concurrent
    writer raced us; convergence in <2 RTTs since per-conv writers are
    rare — single user, single tab).

    Per-call TTL is `conversation_ttl_seconds` (12h default).
    """
    if not conv_id:
        return 0
    key = _conv_key(conv_id)
    ttl = settings.conversation_ttl_seconds
    max_msgs = max_turns * 2
    user_msg = {"role": "user", "content": user_message}
    asst_msg = {"role": "assistant", "content": assistant_message}

    # Cap the WATCH/MULTI/EXEC retry loop. In the original `while True` form,
    # a sustained clash (e.g. a buggy other writer that always mutates the
    # key between our HGET and our EXEC) would spin the request handler
    # forever, blocking the event loop and eventually timing out at the
    # uvicorn layer with a 504. 5 retries is generous: WATCH conflicts are
    # rare (single user, single tab is the dominant case) and back-to-back
    # failures strongly suggest a bug elsewhere, not transient contention.
    # Returning 0 on exhaustion matches the existing `except Exception: return 0`
    # contract — the caller treats it as "history not stored" and the request
    # still returns a valid answer; we just lose one turn of memory. The
    # failure is now also surfaced via a metric (C6) so silent turn-loss is
    # observable rather than invisible.
    for _attempt in range(_MAX_WATCH_RETRIES):
        try:
            async with redis.pipeline(transaction=True) as pipe:
                # WATCH on the conv key — fires WatchError on HSETEX
                # commit if another writer changed the key in between.
                await pipe.watch(key)
                # HGET runs immediately on the watched connection
                # (Pipeline.execute_command routes to immediate_execute_command
                # while watching=True), so we can read the current history
                # before queuing the MULTI block.
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
                # HSETEX with the conversation TTL — atomic field+TTL set.
                # redis-py 8.0: hsetex(name, mapping={...}, ex=ttl)
                # (`fields=` is NOT a real keyword — that's a different lib).
                pipe.hsetex(
                    key,
                    mapping={"history": json.dumps(history)},
                    ex=ttl,
                )
                # C6: key-level EXPIRE so the HASH is evictable under
                # volatile-lru. Queued in the SAME MULTI as the HSETEX so the
                # field write + key EXPIRE commit atomically.
                pipe.expire(key, _key_ttl())
                await pipe.execute()
                return len(history)
        except WatchError:
            # Another writer raced us. Retry with fresh state.
            continue
        except Exception:
            await _incr_stm_append_failure_metric(redis)
            return 0
    # Exhausted retries — log + metric and bail rather than spin forever.
    logger.warning(
        "STM append_turn exhausted WATCH retries; dropping turn",
        conv_id=conv_id,
        attempts=_MAX_WATCH_RETRIES,
    )
    await _incr_stm_append_failure_metric(redis)
    return 0


async def set_summary(redis: Redis, conv_id: str, summary: str) -> None:
    """Single-field summary write + key-level EXPIRE (C6). Safe to call
    concurrently with append_to_history — it touches only the `summary`
    field, never `history`, so there is no read-modify-write race on the
    turn list."""
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
    never clobbered (C7).

    Concurrent-append merge: append_to_history only ever pushes to the END of
    `history`. So if the head we summarized is still the current prefix,
    dropping exactly `head_len` messages from the RE-READ history preserves
    both the fresh tail AND any turns appended during the (slow) LLM call:
    trimmed == cur[head_len:].

    If the prefix no longer matches (append's max-turns cap evicted some of
    the summarized messages, or the conv was cleared/rewritten), index-
    trimming would drop NEWER turns — so we persist the summary ONLY and let
    the next refresh re-trim. No data loss either way.
    """
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
                pipe.expire(key, _key_ttl())  # C6 key-level EXPIRE
                await pipe.execute()
                return
        except WatchError:
            # append_to_history raced us between read and EXEC; retry.
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
            pipe.expire(key, _key_ttl())  # C6 key-level EXPIRE
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
            pipe.expire(key, _key_ttl())  # C6 key-level EXPIRE
            await pipe.execute()
    except Exception:
        pass


async def add_courses(
    redis: Redis, conv_id: str, names: set[str]
) -> None:
    """Read-merge-write the courses field. Sequential HGET + HSETEX (2 RTTs
    total) because the HSETEX value depends on the HGET result. The chat
    handler is the only writer, so a non-transactional pipeline is safe.

    Falls back to legacy-safe behavior: if HSETEX fails for any reason
    (e.g. transient Redis error), the courses aren't tracked this turn —
    LTM worker will still see whatever was there before."""
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
            pipe.expire(key, _key_ttl())  # C6 key-level EXPIRE
            await pipe.execute()
    except Exception:
        pass


# ── Compound operations ──────────────────────────────────────────────────────
async def schedule_afk_sync(
    redis: Redis, conv_id: str, user_id: str
) -> bool:
    """Combines: set_last_active + is_scheduled check + (if not scheduled)
    enqueue sync_ltm_task + set_scheduled.

    Returns True if a new sync was enqueued, False if one was already
    pending (idempotent re-schedule is a no-op).
    """
    if not conv_id or not user_id:
        return False
    await set_last_active(redis, conv_id)
    if await is_scheduled(redis, conv_id):
        return False
    afk_seconds = settings.ltm_afk_threshold_seconds
    # Lazy import: avoid circular import with app.worker
    from app.worker import sync_ltm_task

    try:
        await sync_ltm_task.enqueue(conv_id, user_id).start(delay=afk_seconds, priority="high")
    except Exception as e:
        from loguru import logger
        logger.warning(f"Failed to enqueue AFK LTM sync: {e}")
        return False
    await set_scheduled(redis, conv_id)
    return True


async def clear_conversation(redis: Redis, conv_id: str) -> None:
    """DEL the single HASH key. Replaces the 4-DEL legacy pattern."""
    if not conv_id:
        return
    try:
        await redis.delete(_conv_key(conv_id))
    except Exception:
        pass


# ── High-level history operations (ported from legacy memory.py) ────────────
SUMMARY_TRIGGER_TURNS = 6  # summarize after 6 full turns (12 messages)


def extract_follow_up_questions(assistant_message: str) -> list[str]:
    """Extract numbered follow-up questions from an assistant message.

    Pure function (no Redis). Looks for a "penasaran tentang" section first,
    then falls back to scanning the last 10 lines for a numbered list.
    """
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
    """Resolve numeric input (1, 2, 3) to the matching follow-up question.

    If `query` is a single digit 1-3 and the last assistant message contains
    follow-up questions, returns the corresponding full question text.
    Otherwise returns `query` unchanged.
    """
    stripped_query = query.strip()
    if stripped_query not in ['1', '2', '3']:
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
    max_fresh_turns: int = 5,
    *,
    persist: bool = True,
) -> tuple[str, list[dict]]:
    """Return (summary_str, recent_turns_list) using Rolling Batch Summarization.

    HASH-schema port of the legacy memory.py implementation. Reads history +
    summary in a single HMGET round-trip.

    Two modes (C7):
      persist=False (READ-ONLY — request hot path + AFK worker): never calls
        the LLM and never writes. When the conversation exceeds the fresh-turn
        window, returns the CACHED summary + the in-memory fresh slice. This
        keeps the user-facing request off the (slow) summarization LLM call
        and removes the racy write entirely. The summary itself is refreshed
        out-of-band by `summarize_refresh_task` (see schedule_summary_refresh).
      persist=True (REFRESH TASK only): summarizes the oldest
        `max_fresh_turns*2` messages into the persistent summary field and
        atomically trims history to the remaining fresh turns via
        `_persist_summary_and_trim` (WATCH/MULTI — never clobbers a concurrent
        append_to_history).
    """
    history, cached_summary = await get_history_and_summary(redis, conv_id)

    # 1. Within the fresh-turn window: no LLM call, return as-is.
    if len(history) // 2 <= max_fresh_turns:
        return cached_summary or "", history

    # 2. Window exceeded.
    turns_to_summarize = history[:(max_fresh_turns * 2)]
    fresh_turns = history[(max_fresh_turns * 2):]
    old_summary = cached_summary or ""

    # 2a. READ-ONLY caller: return cached summary + fresh slice, no LLM/write.
    #     The async refresh task owns the refine+persist.
    if not persist:
        return old_summary, fresh_turns

    # 3. Recursive refinement: merge existing summary with overflowing turns.
    from langchain_core.messages import HumanMessage as HM
    old_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:300]}"
        for m in turns_to_summarize
    )
    prompt = (
        "Refine the following conversation summary to include the key points from the new dialogue segment. "
        "Maintain a concise, 2-3 sentence overview. "
        "Write the summary in the dominant language of the conversation (English or Indonesian).\n\n"
        f"Existing Summary:\n{old_summary}\n\n"
        f"New context to integrate:\n{old_text}\n\n"
        "Updated Summary:"
    )

    try:
        resp = await llm.ainvoke(
            [HM(content=prompt)],
            config={"run_name": "ava-rolling-summarization"},
        )
        new_summary = resp.content.strip()
    except Exception as exc:
        logger.warning(f"Failed to generate batch summary: {exc}")
        new_summary = old_summary

    # 4. Atomically persist updated summary + trimmed history. WATCH/MULTI
    #    merge preserves any turns appended during the LLM call (C7).
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
    redis: Redis, conv_id: str, *, max_fresh_turns: int = 5
) -> bool:
    """Enqueue an out-of-band summary refresh IF the conversation has overflowed
    the fresh-turn window (C7). Fire-and-forget; deduped via a short NX lock so
    rapid-fire turns don't pile up redundant refresh jobs.

    Returns True if a refresh was enqueued, False otherwise (within window,
    already in flight, or enqueue failed). Never raises — scheduling must never
    break the request path.
    """
    if not conv_id:
        return False
    try:
        raw = await redis.hget(_conv_key(conv_id), "history")
        try:
            n = len(json.loads(raw)) if raw else 0
        except (json.JSONDecodeError, TypeError):
            n = 0
        if n // 2 <= max_fresh_turns:
            return False
        # NX dedup: 30s window covers the enqueue→consume→persist round-trip.
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
