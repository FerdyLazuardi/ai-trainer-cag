"""
Askfer pipeline — public portfolio chat assistant.

Parallel to A-Pedi's app/graph/pipeline.py. Stateless: no conversation history,
no LTM, no rolling summary, no user preferences. Hits Qdrant collection
`Personal_Portfolio` (homepage + project pages + CV chunks).

Topology:
    pre_processor → route by intent
        ├─ GREETING   → handle_greeting    → END
        ├─ OFF_SCOPE  → handle_off_scope   → END
        ├─ MALICIOUS  → handle_malicious   → END
        └─ KNOWLEDGE  → rag_node → generate_node → END
"""
from functools import lru_cache
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.graph.state import RAGState
from app.llm.askfer_prompts import (
    ASKFER_PERSONA,
    ASKFER_SYSTEM_PROMPT,
    PRE_PROCESSOR_PROMPT,
)
from app.llm.client import get_llm

_settings = get_settings()


class AskferPreProcessorResult(BaseModel):
    intent: Literal["GREETING", "OFF_SCOPE", "MALICIOUS", "KNOWLEDGE"] = Field(
        description="Intent of the user's message."
    )
    rewritten_query: str = Field(
        description="Echo the user's query verbatim (Askfer is stateless — no rewrite)."
    )


# ─── Nodes ───────────────────────────────────────────────────────────────────

async def _pre_processor(state: RAGState, config: RunnableConfig):
    """Classify intent. No history rewrite (stateless)."""
    user_msg = state["messages"][-1].content
    low_msg = user_msg.lower().strip()

    GREETING_PREFIXES = (
        "halo", "hai", "hi", "hey", "hello",
        "pagi", "siang", "sore", "malam",
        "good morning", "good afternoon", "good evening",
        "selamat", "test", "siapa kamu", "siapa lo",
        "who are you", "introduce yourself",
    )
    if low_msg.startswith(GREETING_PREFIXES) and len(low_msg) < 60:
        return {"intent": "GREETING", "rewritten_query": user_msg}

    llm = get_llm()
    structured = llm.with_structured_output(AskferPreProcessorResult)
    try:
        result = await structured.ainvoke(
            [
                SystemMessage(content=PRE_PROCESSOR_PROMPT),
                HumanMessage(content=f"User query: {user_msg}"),
            ],
            config=config,
        )
        intent = result.intent
        rewritten = result.rewritten_query.strip() or user_msg
    except Exception as exc:
        logger.warning(f"Askfer pre-processor failed, defaulting to KNOWLEDGE: {exc}")
        intent = "KNOWLEDGE"
        rewritten = user_msg

    logger.info(f"Askfer pre-processor: intent={intent}")
    return {"intent": intent, "rewritten_query": rewritten}


async def _handle_greeting(state: RAGState, config: RunnableConfig):
    """Friendly bilingual self-introduction."""
    llm = get_llm()
    sys = (
        f"{ASKFER_PERSONA}\n"
        "Greet the visitor warmly and briefly introduce yourself as Ferdy. "
        "Match the user's language (English default, switch to Indonesian if "
        "they wrote in Indonesian). Keep it under 3 sentences. Mention you can "
        "answer questions about your projects, tech stack, and experience."
    )
    response = await llm.ainvoke(
        [SystemMessage(content=sys)] + list(state["messages"]),
        config=config,
    )
    return {"messages": [response]}


async def _handle_off_scope(state: RAGState, config: RunnableConfig):
    """Polite scope redirect, bilingual auto."""
    llm = get_llm()
    sys = (
        f"{ASKFER_PERSONA}\n"
        "The user asked something OUTSIDE your scope (not about projects, tech "
        "stack, experience, skills, education, or contact). Politely decline "
        "in 1-2 sentences and steer them back to professional topics. "
        "Match the user's language (EN default, ID if they wrote in Indonesian).\n"
        "EN example: \"I keep this chat focused on my professional work. For other things, reach out directly via my homepage.\"\n"
        "ID example: \"Aku fokus bahas kerjaan profesional aja di sini. Buat hal lain, kontak aku langsung ya.\""
    )
    response = await llm.ainvoke(
        [SystemMessage(content=sys)] + list(state["messages"]),
        config=config,
    )
    return {"messages": [response]}


async def _handle_malicious(state: RAGState, config: RunnableConfig):
    """Hard refusal — bilingual fixed responses, no LLM call."""
    user_msg = state["messages"][-1].content if state["messages"] else ""
    # Crude language detect — Indonesian common words
    is_id = any(t in user_msg.lower() for t in ("kamu", "saya", "aku", "siapa", "apa", "bagaimana"))
    if is_id:
        msg = "Maaf, aku gak bisa bantu permintaan itu. Aku khusus jawab pertanyaan tentang project, skill, dan pengalaman profesional aku."
    else:
        msg = "Sorry, I can't help with that. I only answer questions about my projects, skills, and professional background."
    return {"messages": [AIMessage(content=msg)]}


async def _rag_node(state: RAGState, config: RunnableConfig):
    """Retrieve from Personal_Portfolio collection only."""
    from app.retrieval.hybrid_retriever import hybrid_search
    from app.retrieval.reranker import rerank

    query_to_search = state.get("rewritten_query") or state["messages"][-1].content
    try:
        docs = await hybrid_search(
            query=query_to_search,
            top_k=_settings.askfer_retrieval_top_k,
            collection=_settings.qdrant_personal_collection,
        )
        reranked = await rerank(
            query=query_to_search,
            chunks=docs,
            top_k=_settings.askfer_reranked_top_k,
        )

        chunks = []
        for d in reranked:
            m = d.metadata or {}
            chunks.append({
                "text": d.text,
                "doc_type": m.get("doc_type", ""),
                "project_slug": m.get("project_slug", ""),
                "project_url": m.get("project_url", ""),
                "title": d.title or m.get("title", ""),
                "score": round(d.score, 4) if d.score is not None else 0.0,
                "hybrid_score": round(d.hybrid_score, 4) if d.hybrid_score is not None else 0.0,
                "source": d.source or m.get("source", "Unknown"),
                "document_id": d.document_id or m.get("document_id", "Unknown"),
            })
        logger.info(f"Askfer rag_node retrieved {len(chunks)} chunks for: {query_to_search[:60]}")
        return {"retrieved_context": chunks}
    except Exception as exc:
        logger.error(f"Askfer rag_node failed: {exc}")
        raise RuntimeError(f"Askfer retrieval failed: {exc}") from exc


async def _generate_node(state: RAGState, config: RunnableConfig):
    """Generate from retrieved context. No history/LTM/preferences interpolation."""
    chunks = state.get("retrieved_context") or []

    if chunks:
        max_chars = _settings.askfer_chunk_text_max_chars
        context_lines = []
        for i, c in enumerate(chunks, 1):
            doc_type = c.get("doc_type") or "doc"
            label = c.get("title") or c.get("project_slug") or c.get("source") or doc_type
            text = c.get("text") or ""
            # Don't truncate the overview doc — it's the complete project index
            # and missing items from it would defeat its purpose.
            if doc_type != "overview":
                text = text[:max_chars]
            context_lines.append(f"[{i}] ({doc_type}) {label}\n{text}")
        context_str = "\n\n---\n\n".join(context_lines)
    else:
        context_str = "No relevant documents found."

    full_system = (
        f"{ASKFER_SYSTEM_PROMPT}"
        f"\n\n<retrieved_context>\n{context_str}\n</retrieved_context>"
    )

    llm = get_llm()
    messages = [SystemMessage(content=full_system)] + list(state["messages"])
    response = await llm.ainvoke(messages, config=config)
    return {"messages": [response]}


# ─── Routing & assembly ─────────────────────────────────────────────────────

def _route_by_intent(state: RAGState) -> str:
    return state.get("intent", "KNOWLEDGE")


def _build_askfer_graph():
    builder = StateGraph(RAGState)
    builder.add_node("pre_processor", _pre_processor)
    builder.add_node("greeting", _handle_greeting)
    builder.add_node("off_scope", _handle_off_scope)
    builder.add_node("malicious", _handle_malicious)
    builder.add_node("rag_node", _rag_node)
    builder.add_node("generate_node", _generate_node)

    builder.add_edge(START, "pre_processor")
    builder.add_conditional_edges(
        "pre_processor",
        _route_by_intent,
        {
            "GREETING": "greeting",
            "OFF_SCOPE": "off_scope",
            "MALICIOUS": "malicious",
            "KNOWLEDGE": "rag_node",
        },
    )
    builder.add_edge("greeting", END)
    builder.add_edge("off_scope", END)
    builder.add_edge("malicious", END)
    builder.add_edge("rag_node", "generate_node")
    builder.add_edge("generate_node", END)
    return builder.compile()


@lru_cache(maxsize=1)
def get_askfer_graph():
    """Singleton compiled Askfer graph."""
    return _build_askfer_graph()
