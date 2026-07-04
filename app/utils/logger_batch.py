import asyncio
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Dict

from loguru import logger
from sqlalchemy import insert

from app.database.models import AgentLog
from app.database.postgres import AsyncSessionLocal
from app.utils.pii import redact_pii

_PII_COLUMNS = ("query", "rewritten_query", "answer")

def _redact_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(entry)
    for col in _PII_COLUMNS:
        if col in out and isinstance(out[col], str):
            out[col] = redact_pii(out[col])
    return out

async def _do_insert(log_data: Dict[str, Any]):
    try:
        valid_cols = {c.name for c in AgentLog.__table__.columns}
        cleaned = {k: v for k, v in log_data.items() if k in valid_cols}
        
        # Prevent StringDataRightTruncationError for provider strings
        if "or_provider" in cleaned and isinstance(cleaned["or_provider"], str):
            cleaned["or_provider"] = cleaned["or_provider"][:64]
            
        async with AsyncSessionLocal() as session:
            await session.execute(insert(AgentLog).values(**cleaned))
            await session.commit()
    except Exception as e:
        logger.error(f"Failed to insert log directly to DB: {e}")

class BatchLogger:
    def __init__(self):
        self._queue: asyncio.Queue[Dict[str, Any]] | None = None
        self._worker: asyncio.Task | None = None

    async def start(self):
        if self._queue is not None:
            return
        self._queue = asyncio.Queue(maxsize=1000)
        self._worker = asyncio.create_task(self._run())

    async def stop(self):
        if self._queue is None:
            return
        await self._queue.join()
        if self._worker is not None:
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
        self._queue = None
        self._worker = None

    async def _run(self):
        assert self._queue is not None
        while True:
            row = await self._queue.get()
            try:
                await _do_insert(row)
            finally:
                self._queue.task_done()

    async def add_log(self, log_entry: Dict[str, Any]):
        if "created_at" not in log_entry:
            log_entry["created_at"] = datetime.now(timezone.utc).isoformat()
        
        redacted = _redact_entry(log_entry)
        
        if "created_at" in redacted and isinstance(redacted["created_at"], str):
            try:
                redacted["created_at"] = datetime.fromisoformat(redacted["created_at"])
            except ValueError:
                pass

        if self._queue is None:
            await _do_insert(redacted)
            return
        await self._queue.put(redacted)

batch_logger = BatchLogger()
