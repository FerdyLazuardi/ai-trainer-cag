import asyncio
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
        
        async with AsyncSessionLocal() as session:
            await session.execute(insert(AgentLog).values(**cleaned))
            await session.commit()
    except Exception as e:
        logger.error(f"Failed to insert log directly to DB: {e}")

class BatchLogger:
    async def start(self):
        pass

    async def stop(self):
        pass

    async def add_log(self, log_entry: Dict[str, Any]):
        if "created_at" not in log_entry:
            log_entry["created_at"] = datetime.now(timezone.utc).isoformat()
        
        redacted = _redact_entry(log_entry)
        
        if "created_at" in redacted and isinstance(redacted["created_at"], str):
            try:
                redacted["created_at"] = datetime.fromisoformat(redacted["created_at"])
            except ValueError:
                pass
                
        asyncio.create_task(_do_insert(redacted))

batch_logger = BatchLogger()
