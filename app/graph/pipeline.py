"""
Optimized Agentic RAG pipeline - Retrieve-then-Generate pattern.

Architecture change vs prior ReAct pattern:
  BEFORE: classifier → agent(LLM decides tool) → ToolNode → agent(LLM answers)  = 3 LLM calls for KNOWLEDGE
  AFTER:  classifier → rag_node(pure retrieval) → generate_node(LLM answers)    = 2 LLM calls for KNOWLEDGE

Savings: ~700 tokens per KNOWLEDGE query (the first "decide to call tool" agent call is eliminated).
"""
import json
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger

from app.config.settings import get_settings
from app.graph.state import RAGState
from app.llm.client import get_llm
from app.llm.prompts import PERSONA

_settings = get_settings()
_MOODLE_BASE = _settings.moodle_api_url.rstrip("/")

# ─── System Prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""<role>
{PERSONA}
</role>

<instructions>
1. Answer ONLY by paraphrasing sentences from the "text" field in the retrieved context below.
   Do NOT add facts, numbers, or details not found in the context.
2. If the retrieved context is empty or not relevant, say: "Maaf, aku tidak menemukan informasi tentang itu. Coba tanya dengan kata kunci lain ya!"
3. Do NOT cite filenames like [Client Protection.md]. Instead, end with a Moodle link when relevant:
   "Pelajari lebih lanjut: [course_name]({_MOODLE_BASE}/course/view.php?id=COURSE_ID)" using `course_id` and `course_name` from the context.
</instructions>

<follow_up>
After EVERY answer, suggest 2-3 related follow-up questions based on the context topics. Format:

**Apa kamu penasaran tentang:**
1. [follow-up question 1]
2. [follow-up question 2]
3. [follow-up question 3]
</follow_up>"""

ROUTER_PROMPT = """Classify intent into exactly one word:
GREETING - salutations, introductions, small talk
AMBIGUOUS - vague/short input needing clarification
KNOWLEDGE - clear question about Amartha facts/policies/training
Reply with one word only."""


# ─── Nodes ───────────────────────────────────────────────────────────────────

def _classify_intent(state: RAGState, config: RunnableConfig):
    """Classify user intent. Fast heuristic first, then LLM if needed."""
    user_msg = state["messages"][-1].content
    low_msg = user_msg.lower().strip()

    # Fast heuristic for common greetings — zero LLM cost
    if low_msg in ["halo", "hi", "hey", "pagi", "siang", "sore", "malam", "test", "siapa", "siapa kamu"]:
        return {"intent": "GREETING"}

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=user_msg)
    ], config=config)
    intent = response.content.upper().strip()

    if "GREETING" in intent:
        return {"intent": "GREETING"}
    if "AMBIGUOUS" in intent:
        return {"intent": "AMBIGUOUS"}
    return {"intent": "KNOWLEDGE"}


def _handle_greeting(state: RAGState, config: RunnableConfig):
    """Simple friendly response for sapaan/perkenalan."""
    llm = get_llm()
    greet_sys = f"{PERSONA} Greet warmly in 'aku/kamu' style. Keep it brief and friendly."
    response = llm.invoke([SystemMessage(content=greet_sys)] + state["messages"], config=config)
    return {"messages": [response]}


def _handle_ambiguity(state: RAGState, config: RunnableConfig):
    """Politely ask for clarification when query is too vague."""
    llm = get_llm()
    ambiguity_sys = f"{PERSONA} The user's message is unclear. Ask casually (aku/kamu) what they need help with."
    response = llm.invoke([SystemMessage(content=ambiguity_sys)] + state["messages"], config=config)
    return {"messages": [response]}


async def _rag_node(state: RAGState, config: RunnableConfig):
    """
    Pure retrieval node — calls hybrid_search + rerank without any LLM call.
    Stores formatted context chunks into state['retrieved_context'].
    This replaces the first ReAct 'agent' call that previously just decided to use a tool.
    """
    from app.retrieval.hybrid_retriever import hybrid_search
    from app.retrieval.reranker import rerank
    from app.config.settings import get_settings

    settings = get_settings()
    user_msg = state["messages"][-1].content

    try:
        docs = await hybrid_search(query=user_msg)
        reranked = rerank(docs)

        chunks = []
        for d in reranked:
            m = d.metadata or {}
            chunks.append({
                "text": d.text,
                "course_id": m.get("course_id", ""),
                "course_name": m.get("course_name", d.title),
                "score": round(d.score, 4) if d.score is not None else 0.0,
                "source": d.source or m.get("source", "Unknown"),
                "document_id": d.document_id or m.get("document_id", "Unknown"),
            })

        logger.info(f"RAG node retrieved {len(chunks)} chunks for query: {user_msg[:60]}")
        return {"retrieved_context": chunks}

    except Exception as e:
        logger.warning(f"RAG node retrieval failed: {e}")
        return {"retrieved_context": []}


def _generate_node(state: RAGState, config: RunnableConfig):
    """
    Generate node — receives already-retrieved context from state and calls LLM once.
    This replaces the second ReAct 'agent' call that synthesized the tool result.
    """
    chunks = state.get("retrieved_context") or []
    user_msg = state["messages"][-1].content

    # Format context for the LLM prompt
    if chunks:
        context_lines = []
        for i, c in enumerate(chunks, 1):
            context_lines.append(
                f"[{i}] (score: {c.get('score', '?')}, source: {c.get('source', '?')}, "
                f"course: {c.get('course_name', '?')} [id:{c.get('course_id', '?')}])\n"
                f"{c.get('text', '')}"
            )
        context_str = "\n\n---\n\n".join(context_lines)
    else:
        context_str = "No relevant documents found."

    full_system = f"{SYSTEM_PROMPT}\n\n<retrieved_context>\n{context_str}\n</retrieved_context>"

    llm = get_llm()
    messages = [SystemMessage(content=full_system)] + list(state["messages"])
    response = llm.invoke(messages, config=config)
    return {"messages": [response]}


# ─── Routing ─────────────────────────────────────────────────────────────────

def _route_by_intent(state: RAGState) -> str:
    return state.get("intent", "KNOWLEDGE")


# ─── Graph Assembly ───────────────────────────────────────────────────────────

def _build_agent_graph():
    """Build and compile the optimized RAG StateGraph."""
    builder = StateGraph(RAGState)

    # Nodes
    builder.add_node("classifier", _classify_intent)
    builder.add_node("greeting", _handle_greeting)
    builder.add_node("ambiguity", _handle_ambiguity)
    builder.add_node("rag_node", _rag_node)
    builder.add_node("generate_node", _generate_node)

    # Edges
    builder.add_edge(START, "classifier")

    builder.add_conditional_edges(
        "classifier",
        _route_by_intent,
        {
            "GREETING": "greeting",
            "AMBIGUOUS": "ambiguity",
            "KNOWLEDGE": "rag_node",
        }
    )

    builder.add_edge("greeting", END)
    builder.add_edge("ambiguity", END)
    builder.add_edge("rag_node", "generate_node")
    builder.add_edge("generate_node", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_rag_graph():
    """Return the singleton compiled RAG graph."""
    return _build_agent_graph()
