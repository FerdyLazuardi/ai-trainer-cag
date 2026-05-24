import asyncio
from app.config.settings import get_settings
from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.reranker import rerank

async def main():
    s = get_settings()
    for q in ["How many users have you reached in total?", "kamu pernah ngajar berapa orang?"]:
        print("Q:", q)
        docs = await hybrid_search(query=q, top_k=8, collection=s.qdrant_personal_collection)
        print("  HYBRID TOP 8 (raw):")
        for d in docs:
            m = d.metadata or {}
            print(f"    doc_type={m.get('doc_type')} slug={m.get('knowledge_slug') or m.get('project_slug') or '-'} hybrid={round(d.score,3)}")
        reranked = await rerank(query=q, chunks=docs, top_k=3)
        print("  RERANKED TOP 3:")
        for d in reranked:
            m = d.metadata or {}
            print(f"    doc_type={m.get('doc_type')} slug={m.get('knowledge_slug') or m.get('project_slug') or '-'} cohere={round(d.score,3)}")
        print()

asyncio.run(main())

