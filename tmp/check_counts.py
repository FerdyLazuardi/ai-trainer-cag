import asyncio
from qdrant_client import AsyncQdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check_all_collections():
    host = os.getenv("QDRANT_HOST", "localhost")
    # For host check, we use 6335 if 6333 fails, or just try both
    for port in [6335, 6333]:
        print(f"Trying Qdrant at {host}:{port}...")
        client = AsyncQdrantClient(host=host, port=port)
        try:
            collections = await client.get_collections()
            print(f"Connected to Qdrant at port {port}")
            for c in collections.collections:
                info = await client.get_collection(c.name)
                print(f"Collection: {c.name}")
                print(f"  - Points count: {info.points_count}")
                print(f"  - Status: {info.status}")
            return
        except Exception as e:
            print(f"Failed at port {port}: {e}")
        finally:
            await client.close()

if __name__ == "__main__":
    asyncio.run(check_all_collections())
