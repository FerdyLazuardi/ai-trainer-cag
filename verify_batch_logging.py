import asyncio
import json
import uuid
from datetime import datetime, timezone
from sqlalchemy import select, func
from app.utils.logger_batch import batch_logger
from app.database.models import AgentLog
from app.database.postgres import AsyncSessionLocal
from app.database.redis_client import get_redis_client

async def verify():
    print("--- Starting Verification ---")
    redis = get_redis_client()
    redis_key = "agent_log_buffer"
    
    # 1. Clear existing logs in Redis buffer for clean test
    await redis.delete(redis_key)
    print("Cleared Redis buffer.")

    # 2. Add some logs
    num_logs = 5
    print(f"Adding {num_logs} logs to buffer...")
    for i in range(num_logs):
        log_entry = {
            "conversation_id": f"test-conv-{uuid.uuid4()}",
            "query": f"Test query {i}",
            "rewritten_query": f"Rewritten test query {i}",
            "answer": f"Test answer {i}",
            "chunks_retrieved": i,
            "latency_ms": 100.0 + i,
            "cache_hit": False,
        }
        await batch_logger.add_log(log_entry)
    
    # 3. Check Redis buffer length
    list_len = await redis.llen(redis_key)
    print(f"Redis buffer length: {list_len}")
    if list_len != num_logs:
        print(f"ERROR: Expected {num_logs} logs in Redis, found {list_len}")
    else:
        print("SUCCESS: Logs correctly buffered in Redis.")

    # 4. Get initial count in DB
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(func.count()).select_from(AgentLog))
        initial_count = result.scalar()
    print(f"Initial DB log count: {initial_count}")

    # 5. Trigger manual flush
    print("Triggering manual flush...")
    await batch_logger.flush()

    # 6. Check Redis buffer length again (should be 0)
    list_len = await redis.llen(redis_key)
    print(f"Redis buffer length after flush: {list_len}")

    # 7. Check DB count again
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(func.count()).select_from(AgentLog))
        final_count = result.scalar()
    print(f"Final DB log count: {final_count}")

    if final_count == initial_count + num_logs:
        print("SUCCESS: Logs correctly flushed to PostgreSQL.")
    else:
        print(f"ERROR: Expected {initial_count + num_logs} logs in DB, found {final_count}")

    print("--- Verification Finished ---")

if __name__ == "__main__":
    asyncio.run(verify())
