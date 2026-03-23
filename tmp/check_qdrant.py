import asyncio
from qdrant_client import AsyncQdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

async def check_qdrant():
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", 6333))
    collection = "Knowledge_Base"
    
    print(f"Connecting to Qdrant at {host}:{port}...")
    client = AsyncQdrantClient(host=host, port=port)
    
    try:
        collections = await client.get_collections()
        print(f"Available collections: {[c.name for c in collections.collections]}")
        
        if collection not in [c.name for c in collections.collections]:
            print(f"Collection '{collection}' NOT FOUND.")
            return

        info = await client.get_collection(collection_name=collection)
        print(f"\nCollection Status: {info.status}")
        print(f"Points Count: {info.points_count}")
        print(f"Indexed Vectors: {info.indexed_vectors_count}")
        
        # Try to retrieve 1 point to see if data exists
        points = await client.scroll(collection_name=collection, limit=1)
        if points[0]:
            print("\nFound at least one point:")
            print(points[0][0])
        else:
            print("\nNo points found via scroll.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(check_qdrant())
