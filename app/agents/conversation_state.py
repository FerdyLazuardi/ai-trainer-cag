"""
Redis schema for conversation state.

Collapses 6 separate keys per conversation into 1 HASH with per-field TTL
via Redis 8's HSETEX / HEXPIRE. Server requirement: Redis 8.0+ for HSETEX,
Redis 7.4+ for HEXPIRE (HSETEX itself is 8.0+).

The transient `rag:ltm:syncing:{id}` lock is kept as a separate STRING
key (not a HASH field) because it must outlive the conversation HASH —
a slow LTM worker can fire after the HASH expires.

HASH schema (`rag:conv:{conversation_id}`) — per-field TTL via HSETEX:
    history      JSON list       24h
    summary      str             24h
    owner        str (user_id)   7d   (set via HSETNX + HEXPIRE)
    last_active  str (epoch)     afk + 3600s
    scheduled    "1"             afk + 600s
    courses      JSON list       afk + 3600s

Note: HSETEX `EX seconds` sets PER-FIELD TTLs, NOT a top-level key TTL.
The HASH key itself has no EXPIRE — fields expire individually, and the
empty HASH keyspace entry lingers (negligible memory). Re-HSETEX'ing a
field that previously expired creates a fresh entry with a fresh TTL.
"""
from __future__ import annotations

import json
import time

from loguru import logger
from redis.asyncio import Redis
from redis.exceptions import WatchError

from app.config.settings import get_settings

settings = get_settings()

# ── Constants ────────────────────────────────────────────────────────────────
_CONV_KEY_PREFIX = "rag:conv:"
_LTM_LOCK_PREFIX = "rag:ltm:syncing:"
_OWNER_TTL_SECONDS = 86400 * 7   # 7 days


def _conv_key(conv_id: str) -> str:
    return f"{_CONV_KEY_PREFIX}{conv_id}"


def _ltm_lock_key(conv_id: str) -> str:
    return f"{_LTM_LOCK_PREFIX}{conv_id}"


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


async def get_owner(redis: Redis, conv_id: str) -> str | None:
    if not conv_id:
        return None
    try:
        val = await redis.hget(_conv_key(conv_id), "owner")
        return val if val else None
    except Exception:
        return None


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

    Per-call TTL is `conversation_ttl_seconds` (24h default).
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
    # still returns a valid answer; we just lose one turn of memory.
    _MAX_WATCH_RETRIES = 5
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
                await pipe.execute()
                return len(history)
        except WatchError:
            # Another writer raced us. Retry with fresh state.
            continue
        except Exception:
            return 0
    # Exhausted retries — log and bail rather than spin forever.
    logger.warning(
        "STM append_turn exhausted WATCH retries; dropping turn",
        conv_id=conv_id,
        attempts=_MAX_WATCH_RETRIES,
    )
    return 0


async def set_summary(redis: Redis, conv_id: str, summary: str) -> None:
    if not conv_id:
        return
    try:
        await redis.hsetex(
            _conv_key(conv_id),
            mapping={"summary": summary},
            ex=settings.conversation_ttl_seconds,
        )
    except Exception:
        pass


async def trim_history(
    redis: Redis, conv_id: str, fresh_turns: list[dict]
) -> None:
    """Called after rolling-batch summarization. HSETEX the trimmed history."""
    if not conv_id:
        return
    try:
        await redis.hsetex(
            _conv_key(conv_id),
            mapping={"history": json.dumps(fresh_turns)},
            ex=settings.conversation_ttl_seconds,
        )
    except Exception:
        pass


async def set_owner(
    redis: Redis, conv_id: str, user_id: str, *, nx: bool = False
) -> bool:
    """If nx=True, uses HSETNX (atomic claim). Returns True if claimed.
    If nx=False, unconditional HSETEX with 7d TTL (used by dev-bypass reclaim)."""
    if not conv_id:
        return False
    key = _conv_key(conv_id)
    try:
        if nx:
            if await redis.hsetnx(key, "owner", user_id):
                await redis.hexpire(key, _OWNER_TTL_SECONDS, "owner")
                return True
            return False
        await redis.hsetex(
            key, mapping={"owner": user_id}, ex=_OWNER_TTL_SECONDS
        )
        return True
    except Exception:
        return False


async def set_last_active(redis: Redis, conv_id: str) -> None:
    if not conv_id:
        return
    try:
        await redis.hsetex(
            _conv_key(conv_id),
            mapping={"last_active": str(time.time())},
            ex=settings.ltm_afk_threshold_seconds + 3600,
        )
    except Exception:
        pass


async def set_scheduled(redis: Redis, conv_id: str) -> None:
    if not conv_id:
        return
    try:
        await redis.hsetex(
            _conv_key(conv_id),
            mapping={"scheduled": "1"},
            ex=settings.ltm_afk_threshold_seconds + 600,
        )
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
        await redis.hsetex(key, mapping={"courses": json.dumps(merged)}, ex=ttl)
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
        await sync_ltm_task.enqueue(conv_id, user_id).start(delay=afk_seconds)
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
