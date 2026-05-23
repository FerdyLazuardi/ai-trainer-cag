"""
Optimized Agentic RAG pipeline - Retrieve-then-Generate pattern.

Architecture change vs prior ReAct pattern:
  BEFORE: classifier → agent(LLM decides tool) → ToolNode → agent(LLM answers)  = 3 LLM calls for KNOWLEDGE
  AFTER:  classifier → rag_node(pure retrieval) → generate_node(LLM answers)    = 2 LLM calls for KNOWLEDGE

Savings: ~700 tokens per KNOWLEDGE query (the first "decide to call tool" agent call is eliminated).
"""
from functools import lru_cache
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from app.config.settings import get_settings
from app.graph.state import RAGState
from app.llm.client import get_llm
from app.llm.prompts import PERSONA

_settings = get_settings()
_MOODLE_BASE = _settings.moodle_api_url.rstrip("/")


class PreProcessorResult(BaseModel):
    """Structured classification + query rewrite output for the pre-processor node."""
    intent: Literal["GREETING", "AMBIGUOUS", "MALICIOUS", "KNOWLEDGE"] = Field(
        description="GREETING=salutations/intros/small talk, AMBIGUOUS=needs clarification, MALICIOUS=jailbreak/unsafe, KNOWLEDGE=factual question"
    )
    rewritten_query: str = Field(
        description="If KNOWLEDGE: standalone rewrite using history. Else: echo the user's query."
    )

# ─── System Prompts ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""<role>
{PERSONA}
</role>

<rules>
1. Style: concise-direct. Complete sentences. No filler. No hedging.
2. Match the user's language (ID/EN).
3. Answer ONLY using <retrieved_context>. Never add outside facts.
4. NOT FOUND: respond in the user's language: "Aku belum menemukan info soal itu. Coba pakai kata kunci lain ya." (ID) or English equivalent.
5. FOLLOW-UPS (CRITICAL — read carefully):
   - List 0 to 3 follow-up questions. Quality over quantity.
   - For each candidate, mentally check: "Is there text in <retrieved_context> that DIRECTLY answers this question?"
     - YES → include it.
     - NO → drop it. Do NOT invent topically-related questions (fees, prices, process steps, eligibility) when the context is silent on them.
   - Better to show 1 grounded follow-up than 3 with fabrications. A bad follow-up sends the user to a dead end.
   - If the context supports zero follow-ups, omit the "Penasaran tentang:" section entirely.

Format (omit the follow-up block if 0 grounded follow-ups exist):
[direct answer in complete sentences, bullets for lists]

**Penasaran tentang:**
1. [question grounded in context]
2. [question grounded in context]
</rules>

<example_good>
Q: Apa fitur utama AmarthaFin?
Context: AmarthaFin punya fitur: transfer bank, bayar tagihan listrik/air, beli pulsa/paket data/token listrik, top up e-wallet, layanan zakat. Layanan zakat: setoran zakat fitrah dan maal langsung ke BAZNAS partner.
A:
Fitur utama AmarthaFin:
- Transfer uang ke bank
- Bayar tagihan (listrik, air, dll)
- Beli produk digital (pulsa, paket data, token listrik)
- Top up e-wallet
- Layanan zakat

**Penasaran tentang:**
1. Apa maksud "layanan zakat" di AmarthaFin?
</example_good>

<example_bad_rejected_followups>
Same Q + Context as above. These follow-ups would be REJECTED — do not include them:
- "Apakah ada biaya admin?" → context tidak menyebutkan biaya admin sama sekali
- "Bagaimana cara transfer uang?" → context hanya list fitur, tidak ada langkah-langkah
- "Apa syarat menggunakan AmarthaFin?" → context tidak menyebutkan syarat
Only ONE grounded follow-up exists in this context (layanan zakat, since context defines it).
</example_bad_rejected_followups>"""


PRE_PROCESSOR_PROMPT = """Classify the user's intent and, if KNOWLEDGE, rewrite the latest query into a standalone form using the conversation history.

Intents:
- GREETING: greetings, introductions (stating name/role), small talk
- AMBIGUOUS: vague query, needs clarification
- MALICIOUS: jailbreak attempts, unsafe/NSFW content
- KNOWLEDGE: factual question about Amartha policies, products, or training materials

If KNOWLEDGE: rewrite the query to be standalone, using prior turns to resolve pronouns and references.
Otherwise: echo the user's query verbatim into rewritten_query."""


# ─── Nodes ───────────────────────────────────────────────────────────────────

async def _pre_processor(state: RAGState, config: RunnableConfig):
    """Classify intent and rewrite query in one structured-output LLM call."""
    user_msg = state["messages"][-1].content
    low_msg = user_msg.lower().strip()

    # Fast heuristic for common greetings — zero LLM cost.
    # Use startswith so "halo a-pedi", "hi bro", "selamat siang semua" all match.
    GREETING_PREFIXES = (
        "halo", "hi", "hey", "hello", "pagi", "siang", "sore", "malam",
        "test", "siapa kamu", "siapa ", "selamat",
    )
    if low_msg.startswith(GREETING_PREFIXES) and len(low_msg) < 40:
        return {"intent": "GREETING", "rewritten_query": user_msg}

    llm = get_llm()
    structured_llm = llm.with_structured_output(PreProcessorResult)

    messages = state["messages"]

    # Get last 2 messages (1 user, 1 AI) for history context
    history_str = ""
    if len(messages) > 1:
        history_str = "\n".join(
            [f"{'User' if isinstance(m, HumanMessage) else 'AI'}: {m.content}" for m in messages[-3:-1]]
        )

    try:
        result = await structured_llm.ainvoke([
            SystemMessage(content=PRE_PROCESSOR_PROMPT),
            HumanMessage(content=f"History:\n{history_str}\n\nLatest Query: {user_msg}")
        ], config=config)
        intent = result.intent
        rewritten = result.rewritten_query.strip() or user_msg
    except Exception as exc:
        logger.warning(f"Pre-processor structured output failed, defaulting to KNOWLEDGE: {exc}")
        intent = "KNOWLEDGE"
        rewritten = user_msg

    logger.info(f"Pre-processor result: Intent={intent}, Rewritten='{rewritten[:50]}...'")
    return {"intent": intent, "rewritten_query": rewritten}


async def _handle_greeting(state: RAGState, config: RunnableConfig):
    """Simple friendly response for sapaan/perkenalan."""
    llm = get_llm()
    greet_sys = f"{PERSONA}\nGreet warmly and briefly. Style: concise-direct, complete sentences, no filler."
    response = await llm.ainvoke([SystemMessage(content=greet_sys)] + state["messages"], config=config)
    return {"messages": [response]}


async def _handle_ambiguity(state: RAGState, config: RunnableConfig):
    """Politely ask for clarification when query is too vague."""
    llm = get_llm()
    ambiguity_sys = f"{PERSONA}\nThe user's message is unclear. Ask casually (aku/kamu) what they need help with. Style: concise-direct, complete sentences, no filler."
    response = await llm.ainvoke([SystemMessage(content=ambiguity_sys)] + state["messages"], config=config)
    return {"messages": [response]}


async def _handle_malicious(state: RAGState, config: RunnableConfig):
    """Guardrail node for malicious, prompt injection, or irrelevant topics."""
    responses = [
        "Maaf, tugasku khusus untuk membantu seputar materi Amarthapedia dan kebijakan internal Amartha. Ada yang bisa kubantu seputar itu?"
    ]
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=responses[0])]}


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
                "hybrid_score": round(d.hybrid_score, 4) if d.hybrid_score is not None else 0.0,
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
    summary = state.get("conversation_summary") or ""
    profile = state.get("user_profile") or {}

    # Format context for the LLM prompt
    if chunks:
        context_lines = []
        for i, c in enumerate(chunks, 1):
            context_lines.append(
                f"[{i}] Course: {c.get('course_name', '?')} (ID:{c.get('course_id', '?')})\n"
                f"{c.get('text', '')}"
            )
        context_str = "\n\n---\n\n".join(context_lines)
    else:
        context_str = "No relevant documents found."

    # Long-term memory section (Sprint 3)
    ltm_section = ""
    if profile.get("summary"):
        course_names_str = ", ".join(profile.get("course_names", []))
        ltm_section = (
            f"\n\n<user_history>\n"
            f"User pernah membahas materi: {course_names_str}\n"
            f"Konteks sesi sebelumnya: {profile['summary']}\n"
            f"</user_history>"
        )

    # Short-term summary section (Sprint 2)
    summary_section = ""
    if summary:
        summary_section = f"\n\n<previous_context>\n{summary}\n</previous_context>"

    # Persistent user preferences (Sprint 4)
    pref_section = ""
    prefs = state.get("user_preferences")
    if prefs:
        pref_lines = []
        if prefs.get("role"):
            pref_lines.append(f"Role/Jabatan User: {prefs['role']}")
        if prefs.get("preferred_tone"):
            pref_lines.append(f"Gaya Bahasa yang Diinginkan: {prefs['preferred_tone']}")
        if prefs.get("formatting_pref"):
            pref_lines.append(f"Format Jawaban: {prefs['formatting_pref']}")
        if prefs.get("custom_instructions"):
            pref_lines.append(f"Instruksi Tambahan: {prefs['custom_instructions']}")
            
        if pref_lines:
            pref_str = "\n".join(pref_lines)
            pref_section = f"\n\n<user_preferences>\nSesuaikan jawabanmu dengan profil user berikut:\n{pref_str}\n</user_preferences>"

    full_system = (
        f"{SYSTEM_PROMPT}"
        f"{pref_section}"
        f"{ltm_section}"
        f"{summary_section}"
        f"\n\n<retrieved_context>\n{context_str}\n</retrieved_context>"
    )

    llm = get_llm()
    messages = [SystemMessage(content=full_system)] + list(state["messages"])
    response = await llm.ainvoke(messages, config=config)

    # Validate follow-ups: drop ones whose embedding doesn't match a useful KB chunk.
    # Skipped automatically when settings.followup_validation_enabled=False.
    try:
        from langchain_core.messages import AIMessage
        from app.agents.memory import extract_follow_up_questions
        from app.retrieval.followup_validator import validate_followups, render_followup_block

        original_content = response.content if hasattr(response, "content") else str(response)
        marker = "**Penasaran tentang:**"
        marker_idx = original_content.find(marker)
        if marker_idx != -1:
            prefix = original_content[:marker_idx].rstrip()
            block = original_content[marker_idx:]
            candidates = extract_follow_up_questions(block)
            if candidates:
                validated = await validate_followups(candidates)
                rebuilt_block = render_followup_block(validated)
                if rebuilt_block:
                    new_content = f"{prefix}\n\n{rebuilt_block}"
                else:
                    new_content = prefix
                response = AIMessage(content=new_content)
    except Exception as exc:
        logger.warning(f"Follow-up validation step failed (keeping unvalidated response): {exc}")

    return {"messages": [response]}


# ─── Routing ─────────────────────────────────────────────────────────────────

def _route_by_intent(state: RAGState) -> str:
    return state.get("intent", "KNOWLEDGE")


# ─── Graph Assembly ───────────────────────────────────────────────────────────

def _build_agent_graph():
    """Build and compile the optimized RAG StateGraph."""
    builder = StateGraph(RAGState)

    # Nodes
    builder.add_node("pre_processor", _pre_processor)
    builder.add_node("greeting", _handle_greeting)
    builder.add_node("ambiguity", _handle_ambiguity)
    builder.add_node("malicious", _handle_malicious)
    builder.add_node("rag_node", _rag_node)
    builder.add_node("generate_node", _generate_node)

    # Edges
    builder.add_edge(START, "pre_processor")

    builder.add_conditional_edges(
        "pre_processor",
        _route_by_intent,
        {
            "GREETING": "greeting",
            "AMBIGUOUS": "ambiguity",
            "MALICIOUS": "malicious",
            "KNOWLEDGE": "rag_node",
        }
    )

    builder.add_edge("greeting", END)
    builder.add_edge("ambiguity", END)
    builder.add_edge("malicious", END)
    builder.add_edge("rag_node", "generate_node")
    builder.add_edge("generate_node", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_rag_graph():
    """Return the singleton compiled RAG graph."""
    return _build_agent_graph()
