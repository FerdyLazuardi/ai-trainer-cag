import asyncio, tiktoken
from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.reranker import rerank
from app.config.settings import get_settings
from app.llm.askfer_prompts import ASKFER_SYSTEM_PROMPT

enc = tiktoken.get_encoding("cl100k_base")
def count(t): return len(enc.encode(t or ""))

async def main():
    s = get_settings()
    q = "What did you build in the Agent Network project?"
    docs = await hybrid_search(query=q, top_k=s.askfer_retrieval_top_k, collection=s.qdrant_personal_collection)
    reranked = await rerank(query=q, chunks=docs, top_k=s.askfer_reranked_top_k)
    chunks = [{"doc_type": (d.metadata or {}).get("doc_type",""), "title": d.title, "project_slug": (d.metadata or {}).get("project_slug",""), "source": d.source, "text": d.text} for d in reranked]
    max_c = s.askfer_chunk_text_max_chars
    ctx_lines = []
    for i, c in enumerate(chunks, 1):
        label = c.get("title") or c.get("project_slug") or c.get("source") or c.get("doc_type")
        text = (c.get("text") or "")[:max_c]
        ctx_lines.append("[" + str(i) + "] (" + c.get("doc_type","") + ") " + label + "\n" + text)
    ctx = "\n\n---\n\n".join(ctx_lines)
    full = ASKFER_SYSTEM_PROMPT + "\n\n<retrieved_context>\n" + ctx + "\n</retrieved_context>"
    print("===TOKENBUDGET===")
    print("system_prompt_tokens=" + str(count(ASKFER_SYSTEM_PROMPT)))
    print("context_tokens=" + str(count(ctx)))
    print("full_system_tokens=" + str(count(full)))
    print("query_tokens=" + str(count(q)))
    print("total_input_tokens=" + str(count(full) + count(q)))
    print("chunks_used=" + str(len(chunks)))

asyncio.run(main())
