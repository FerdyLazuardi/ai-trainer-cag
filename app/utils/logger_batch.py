import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import insert

from app.database.models import AgentLog
from app.database.postgres import AsyncSessionLocal
from app.database.redis_client import get_redis_client


class BatchLogger:
    def __init__(self, batch_size: int = 50, flush_interval: float = 10.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.redis_key = "agent_log_buffer"
        self._stop_event = asyncio.Event()
        self._flush_task: Optional[asyncio.Task] = None

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
        """Buffer a log entry in Redis."""
        redis = get_redis_client()
        
        # Add timestamp if not present
        if "created_at" not in log_entry:
            log_entry["created_at"] = datetime.now(timezone.utc).isoformat()
        
        try:
            # LPUSH is O(1).
            await redis.lpush(self.redis_key, json.dumps(log_entry))
            
            # If we've reached the threshold, we could signal the worker to flush immediately.
            # But the requirement says "when a threshold is reached OR timer expires".
            # For simplicity, we check the length and flush if it's > batch_size.
            list_len = await redis.llen(self.redis_key)
            if list_len >= self.batch_size:
                # Trigger an immediate background flush without blocking the request
                asyncio.create_task(self.flush())
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
        """Flush logs from Redis to PostgreSQL."""
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
                async with AsyncSessionLocal() as session:
                    await session.execute(insert(AgentLog), logs_to_insert)
                    await session.commit()
                logger.info(f"Successfully flushed {len(logs_to_insert)} logs to PostgreSQL")
            
            # Clean up temp key only after successful DB commit
            await redis.delete(temp_key)
                
        except Exception as e:
            if "no such key" in str(e).lower():
                return
            logger.error(f"Error during BatchLogger flush: {e}")

# Singleton instance
batch_logger = BatchLogger()
