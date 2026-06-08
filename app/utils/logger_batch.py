import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import insert

from app.database.models import AgentLog
from app.database.postgres import AsyncSessionLocal
from app.database.redis_client import get_redis_client
from app.utils.pii import redact_pii


# Columns of AgentLog that may contain user free-form text. These are the
# three fields where a PII (email, phone, NIK, NPWP, credit card) can
# land if the user typed it. We redact at the batch-insert boundary
# (not at call sites) so no caller can forget — this is the single
# chokepoint between the request path and the durable log table.
_PII_COLUMNS = ("query", "rewritten_query", "answer")

# Hard cap on the Redis list length. The list is a buffer between the
# request path and the agent_logs Postgres table; in steady state it's
# flushed every 10s and stays under 50 entries. 1000 is 20x the
# steady-state ceiling — anything beyond means the flush worker is
# broken (DB down, OOM, etc.) and we should drop the oldest rather
# than let the list grow unbounded. LTRIM keeps the most recent
# 1000 entries, which is what an operator debugging a stuck flush
# actually wants.
_LOG_BUFFER_MAX = 1000


def _redact_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of `entry` with PII columns redacted.

    Non-PII columns (ids, scores, ints, bools) pass through unchanged.
    Missing columns are left missing (we don't synthesize empty
    strings; SQLAlchemy would interpret that as 'user typed
    nothing' and break downstream eval queries).
    """
    out = dict(entry)
    for col in _PII_COLUMNS:
        if col in out and isinstance(out[col], str):
            out[col] = redact_pii(out[col])
    return out


class BatchLogger:
    def __init__(self, batch_size: int = 50, flush_interval: float = 10.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.redis_key = "agent_log_buffer"
        self._stop_event = asyncio.Event()
        self._flush_task: Optional[asyncio.Task] = None
        self._flush_lock = asyncio.Lock()
        self._inflight: set[asyncio.Task] = set()

    async def start(self):
        """Start the background flushing task."""
        if self._flush_task is None:
            self._stop_event.clear()
            self._flush_task = asyncio.create_task(self._flush_worker())
            logger.info("BatchLogger background worker started")

    async def stop(self):
        """Stop the background flushing task and perform a final flush."""
        if self._flush_task:
            self._stop_event.set()
            # Wait for the worker to finish the current cycle
            try:
                await asyncio.wait_for(self._flush_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning("BatchLogger worker did not stop gracefully, cancelling")
                self._flush_task.cancel()
            self._flush_task = None
            logger.info("BatchLogger background worker stopped")
        
        # Final flush
        await self.flush()

    async def add_log(self, log_entry: Dict[str, Any]):
        """Buffer a log entry in Redis.

        The entry is run through `_redact_entry()` so PII (emails,
        phones, NIK, NPWP, credit cards) in `query` / `rewritten_query`
        / `answer` is masked before it lands in Redis. This is the
        single chokepoint between the request path and the durable
        agent_logs Postgres table; callers don't need to remember
        to redact.
        """
        redis = get_redis_client()

        # Add timestamp if not present
        if "created_at" not in log_entry:
            log_entry["created_at"] = datetime.now(timezone.utc).isoformat()

        # Redact free-form text fields in-place (a copy) so PII
        # never reaches Redis. We mutate the local dict only; the
        # caller's reference is left alone.
        redacted = _redact_entry(log_entry)

        try:
            # LPUSH + LTRIM in a single pipeline so the cap is applied
            # atomically — there's no window where the list could be
            # LPUSH'd without an LTRIM following it. The transaction
            # wraps both ops, so if either fails, neither lands.
            #
            # LTRIM keeps indices 0.._LOG_BUFFER_MAX-1 (the most
            # recent N entries). If the flush worker is broken and the
            # list would otherwise grow unboundedly, the oldest entries
            # are dropped — which is the right failure mode for a
            # short-lived buffer. The console log on the next flush
            # failure will surface the underlying cause.
            payload = json.dumps(redacted)
            async with redis.pipeline(transaction=True) as pipe:
                pipe.lpush(self.redis_key, payload)
                pipe.ltrim(self.redis_key, 0, _LOG_BUFFER_MAX - 1)
                pipe.llen(self.redis_key)
                results = await pipe.execute()
            list_len = results[2]

            # If we've reached the threshold, we could signal the worker to flush immediately.
            # But the requirement says "when a threshold is reached OR timer expires".
            # For simplicity, we check the length and flush if it's > batch_size.
            if list_len >= self.batch_size:
                # Trigger an immediate background flush without blocking the request.
                # Track the task so Python doesn't GC it mid-flight.
                task = asyncio.create_task(self.flush())
                self._inflight.add(task)
                task.add_done_callback(self._inflight.discard)
        except Exception as e:
            logger.error(f"Failed to buffer log to Redis: {e}")

    async def _flush_worker(self):
        """Background worker that periodically flushes logs."""
        while not self._stop_event.is_set():
            try:
                # Wait for the interval OR for the stop event
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.flush_interval)
            except asyncio.TimeoutError:
                # Interval reached, flush logs
                await self.flush()
            except Exception as e:
                logger.error(f"BatchLogger worker error: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors

    async def flush(self):
        """Flush logs from Redis to PostgreSQL. Serialized via flush_lock."""
        async with self._flush_lock:
            redis = get_redis_client()
            temp_key = f"{self.redis_key}:temp"

            try:
                # check if redis is available
                await redis.ping()

                # Handle previously failed flushes if temp key exists
                if not await redis.exists(temp_key):
                    if await redis.llen(self.redis_key) == 0:
                        return
                    # Atomic rename to prepare for processing
                    await redis.rename(self.redis_key, temp_key)

                # Get all logs from temp list
                raw_logs = await redis.lrange(temp_key, 0, -1)
                if not raw_logs:
                    await redis.delete(temp_key)
                    return

                logger.info(f"Flushing {len(raw_logs)} logs from Redis to PostgreSQL")

                logs_to_insert = []
                for raw_log in raw_logs:
                    try:
                        log_data = json.loads(raw_log)
                        # Convert isoformat string back to datetime
                        if "created_at" in log_data and isinstance(log_data["created_at"], str):
                            log_data["created_at"] = datetime.fromisoformat(log_data["created_at"])
                        logs_to_insert.append(log_data)
                    except Exception as e:
                        logger.error(f"Error parsing log from Redis: {e}")

                if logs_to_insert:
                    # Normalize for bulk insert: drop keys that aren't AgentLog
                    # columns (e.g. ad-hoc debug fields) and give every row the
                    # SAME key set (union, missing → None). SQLAlchemy's
                    # multi-row INSERT requires homogeneous dicts; without this a
                    # single heterogeneous row breaks the whole batch.
                    valid_cols = {c.name for c in AgentLog.__table__.columns}
                    cleaned = [
                        {k: v for k, v in row.items() if k in valid_cols}
                        for row in logs_to_insert
                    ]
                    all_keys: set[str] = set()
                    for row in cleaned:
                        all_keys.update(row.keys())
                    normalized = [
                        {k: row.get(k) for k in all_keys} for row in cleaned
                    ]
                    async with AsyncSessionLocal() as session:
                        await session.execute(insert(AgentLog), normalized)
                        await session.commit()
                    logger.info(f"Successfully flushed {len(normalized)} logs to PostgreSQL")

                # Clean up temp key only after successful DB commit
                await redis.delete(temp_key)

            except Exception as e:
                if "no such key" in str(e).lower():
                    return
                logger.error(f"Error during BatchLogger flush: {e}")

# Singleton instance
batch_logger = BatchLogger()
