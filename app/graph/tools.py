"""
Retrieval Tool for the Agentic RAG graph.
Exposes the hybrid search and reranker logic to the LLM as a callable tool.
"""
import json

from langchain_core.tools import tool

from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.reranker import rerank


@tool
async def search_company_knowledge(query: str) -> str:
    """
    Search the company knowledge base for context. 
    Use this tool when you need information about internal processes, products, or technical documentation.
    
    Args:
        query (str): The specific question or search terms to look up.
        
    Returns:
        str: A JSON-formatted string of relevant document snippets and their sources.
    """
    # 1. Hybrid Search
    docs = await hybrid_search(query=query)
    
    if not docs:
        return json.dumps([{"text": "No relevant documents found.", "source": "None"}])
        
    # 2. Rerank and deduplicate
    reranked = rerank(docs)
    
    # 3. Format MINIMAL output for the LLM — only what it needs to answer.
    #    Fewer fields = fewer tokens = less hallucination surface.
    results = []
    for d in reranked:
        m = d.metadata or {}
        results.append({
            "text": d.text,
            "course_id": m.get("course_id", ""),
            "course_name": m.get("course_name", d.title),
            "score": round(d.score, 4) if getattr(d, 'score', None) is not None else 0.0,
            "source": getattr(d, 'source', None) or m.get("source", "Unknown"),
            "document_id": getattr(d, 'document_id', None) or m.get("document_id", "Unknown"),
        })
        
    return json.dumps(results, ensure_ascii=False)
