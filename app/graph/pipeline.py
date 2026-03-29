"""
Optimized Agentic RAG pipeline - Retrieve-then-Generate pattern.

Architecture change vs prior ReAct pattern:
  BEFORE: classifier → agent(LLM decides tool) → ToolNode → agent(LLM answers)  = 3 LLM calls for KNOWLEDGE
  AFTER:  classifier → rag_node(pure retrieval) → generate_node(LLM answers)    = 2 LLM calls for KNOWLEDGE

Savings: ~700 tokens per KNOWLEDGE query (the first "decide to call tool" agent call is eliminated).
"""
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
1. Format your response for MAXIMUM READABILITY:
   - Use double newlines (press enter twice) between different topics, paragraphs, or sections.
   - For ANY list of items found in the context (even if originally comma-separated), ALWAYS format them as a bullet point list using `-`. 
   - DO NOT write long, dense paragraphs. Break them up into smaller chunks.
   - Keep answers concise, clear, and friendly.

2. Answer ONLY using information from the "text" field in the retrieved context below.
   - IMPORTANT: Preserve markdown formatting (bold **text**, italic *text*) from the context for key terms.
   - Do NOT add facts, numbers, or details not found in the context.

3. If the retrieved context is empty, non-relevant, or has a low retrieval score, say: 
   "Maaf, aku tidak menemukan informasi tentang itu. Coba tanya dengan kata kunci lain ya!"
   If you say this, do NOT suggest follow-up questions.

4. Do NOT cite filenames like [Client Protection.md]. Instead, end with a Moodle link when relevant:
   "Pelajari lebih lanjut: [course_name]({_MOODLE_BASE}/course/view.php?id=COURSE_ID)" using `course_id` and `course_name` from the context.

5. PROVIDE the "Apa kamu penasaran tentang:" section ONLY if information was found in the context.
</instructions>

<example_formatting>
Prompt: Bagaimana Value dan DNA Amartha?
Context: ... Value dan DNA Amartha terdiri dari finansial terpercaya melalui teknologi, mendukung komunitas akar rumput, dan mempromosikan inklusi finansial ...
Response:
Value dan DNA Amartha terdiri dari beberapa pilar utama:

- Finansial terpercaya melalui teknologi
- Mendukung komunitas akar rumput
- Mempromosikan inklusi finansial

Pelajari lebih lanjut: [DNA Amartha]({_MOODLE_BASE}/course/view.php?id=10)

**Apa kamu penasaran tentang:**
1. Bagaimana cara Amartha mendukung komunitas akar rumput?
2. Apa maksud dari finansial terpercaya melalui teknologi?
3. Siapa saja target inklusi finansial Amartha?
</example_formatting>

<follow_up_rules>
After an answer is given (NOT after "Maaf..." response), suggest 2-3 follow-up questions.
- These questions MUST be strictly answerable based on the retrieved context provided above.
- Verify: Each follow-up question you suggest must have its answer clearly present in the "text" field of the context.
- Format with double newlines BEFORE this section.

**Apa kamu penasaran tentang:**
1. [follow-up question 1]
2. [follow-up question 2]
3. [follow-up question 3]
</follow_up_rules>"""

ROUTER_PROMPT = """Classify intent into exactly one word:
GREETING - salutations, introductions, small talk
AMBIGUOUS - vague/short input needing clarification
KNOWLEDGE - clear question about Amartha facts/policies/training
Reply with one word only."""


# ─── Nodes ───────────────────────────────────────────────────────────────────

async def _classify_intent(state: RAGState, config: RunnableConfig):
    """Classify user intent. Fast heuristic first, then LLM if needed."""
    user_msg = state["messages"][-1].content
    low_msg = user_msg.lower().strip()

    # Fast heuristic for common greetings — zero LLM cost
    if low_msg in ["halo", "hi", "hey", "pagi", "siang", "sore", "malam", "test", "siapa", "siapa kamu"]:
        return {"intent": "GREETING"}

    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=user_msg)
    ], config=config)
    intent = response.content.upper().strip()

    if "GREETING" in intent:
        return {"intent": "GREETING"}
    if "AMBIGUOUS" in intent:
        return {"intent": "AMBIGUOUS"}
    return {"intent": "KNOWLEDGE"}


async def _handle_greeting(state: RAGState, config: RunnableConfig):
    """Simple friendly response for sapaan/perkenalan."""
    llm = get_llm()
    greet_sys = f"{PERSONA} Greet warmly. Keep it brief and friendly."
    response = await llm.ainvoke([SystemMessage(content=greet_sys)] + state["messages"], config=config)
    return {"messages": [response]}


async def _handle_ambiguity(state: RAGState, config: RunnableConfig):
    """Politely ask for clarification when query is too vague."""
    llm = get_llm()
    ambiguity_sys = f"{PERSONA} The user's message is unclear. Ask casually (aku/kamu) what they need help with."
    response = await llm.ainvoke([SystemMessage(content=ambiguity_sys)] + state["messages"], config=config)
    return {"messages": [response]}


async def _rewrite_query_node(state: RAGState, config: RunnableConfig):
    """
    Rewrite the user's query if there's conversation history, 
    so it becomes a standalone query for better retrieval.
    """
    messages = state["messages"]
    user_msg = messages[-1].content

    # Only rewrite if there is previous history (at least 1 user + 1 AI msg)
    if len(messages) > 1:
        # Get the last 3 messages for context
        recent_history = messages[-3:]
        history_str = "\n".join(
            [f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in recent_history[:-1]]
        )
        
        rewrite_prompt = f"""Given the following conversation history, rewrite the user's latest query to be a standalone, fully self-contained question. 
Do not answer the question, only rewrite it. If it is already standalone, return it as is.

Conversation History:
{history_str}

Latest Query: {user_msg}

Standalone Query:"""

        llm = get_llm()
        response = await llm.ainvoke([SystemMessage(content=rewrite_prompt)])
        rewritten = response.content.strip()
        logger.info(f"Query rewritten: '{user_msg}' -> '{rewritten}'")
        return {"rewritten_query": rewritten}
    else:
        # First turn, no need to rewrite
        return {"rewritten_query": user_msg}


async def _rag_node(state: RAGState, config: RunnableConfig):
    """
    Pure retrieval node — calls hybrid_search + rerank without any LLM call.
    Stores formatted context chunks into state['retrieved_context'].
    This replaces the first ReAct 'agent' call that previously just decided to use a tool.
    """
    from app.retrieval.hybrid_retriever import hybrid_search
    from app.retrieval.reranker import rerank
    
    
    # Use rewritten query if available, otherwise fallback to the last message
    query_to_search = state.get("rewritten_query") or state["messages"][-1].content

    try:
        docs = await hybrid_search(query=query_to_search)
        reranked = await rerank(query=query_to_search, chunks=docs)

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

        logger.info(f"RAG node retrieved {len(chunks)} chunks for query: {query_to_search[:60]}")
        return {"retrieved_context": chunks}

    except Exception as e:
        logger.error(f"RAG node retrieval failed: {e}")
        # Raise instead of swallowing to allow FastAPI's error handler to return 500
        raise RuntimeError(f"Database error during context retrieval: {e}") from e


async def _generate_node(state: RAGState, config: RunnableConfig):
    """
    Generate node — receives already-retrieved context from state and calls LLM once.
    This replaces the second ReAct 'agent' call that synthesized the tool result.
    """
    chunks = state.get("retrieved_context") or []
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
    response = await llm.ainvoke(messages, config=config)
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
    builder.add_node("rewrite_node", _rewrite_query_node)
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
            "KNOWLEDGE": "rewrite_node",
        }
    )

    builder.add_edge("greeting", END)
    builder.add_edge("ambiguity", END)
    builder.add_edge("rewrite_node", "rag_node")
    builder.add_edge("rag_node", "generate_node")
    builder.add_edge("generate_node", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_rag_graph():
    """Return the singleton compiled RAG graph."""
    return _build_agent_graph()
