import asyncio
import os
from arq import create_pool
from arq.connections import RedisSettings

async def main():
    redis_port = int(os.environ.get("REDIS_PORT", 6381))
    print(f"Connecting to Redis on port {redis_port}...")
    redis = await create_pool(RedisSettings(host='localhost', port=redis_port))
    job = await redis.enqueue_job('dummy_task', name='VerificationTester')
    print(f"Enqueued job: {job.job_id}")
    await redis.close()

if __name__ == "__main__":
    asyncio.run(main())
