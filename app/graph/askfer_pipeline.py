"""
Askfer pipeline — public portfolio chat assistant.

Parallel to Ava's app/graph/pipeline.py. Stateless: no conversation history,
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
from typing import Literal, cast

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
from app.llm.client import get_llm, get_preprocessor_llm

_settings = get_settings()


@lru_cache(maxsize=1)
def _load_profile_block() -> str:
    """Read data/personal/profile.md ONCE per process and cache the result.

    profile.md is a small static single-source-of-truth file. The previous
    inline `open()` inside the async greeting handler did blocking file I/O on
    the event loop on EVERY greeting. Caching at module level means the read
    happens at most once per worker process; every subsequent greeting is a
    pure in-memory return — no blocking, no redundant syscalls. Returns the
    pre-wrapped `<profile>...</profile>` block (or "" if missing/empty/unreadable).
    """
    import os

    profile_path = "data/personal/profile.md"
    if not os.path.exists(profile_path):
        return ""
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile_text = f.read().strip()
        if profile_text:
            return f"\n\n<profile>\n{profile_text}\n</profile>"
    except Exception as exc:
        logger.warning(f"Greeting: failed to load profile.md: {exc}")
    return ""


def _detect_user_language(text: str) -> str:
    """Cheap heuristic language detector for the user's last message.

    Returns 'Indonesian' if any common ID marker word is found, else 'English'.
    Used to inject an explicit language directive at the end of the system
    prompt so the LLM doesn't get pulled into Indonesian by bilingual context
    (the overview doc + scraped pages mix both languages).
    """
    if not text:
        return "English"
    padded = " " + text.lower() + " "
    id_markers = (
        " kamu ", " saya ", " aku ", " apa ", " apakah ", " siapa ", " gimana ",
        " bagaimana ", " kenapa ", " mengapa ", " dimana ", " di mana ",
        " kapan ", " jelasin ", " jelaskan ", " ceritain ", " ceritakan ",
        " yang ", " ini ", " itu ", " adalah ", " untuk ", " dengan ",
        " dong ", " sih ", " nih ", " banget ", " sebagai ", " punya ",
        " buat ", " bikin ", " kerja ", " dimana ", " gimana ",
    )
    if any(m in padded for m in id_markers):
        return "Indonesian"
    return "English"


class AskferPreProcessorResult(BaseModel):
    intent: Literal["GREETING", "OFF_SCOPE", "MALICIOUS", "KNOWLEDGE"] = Field(
        description="Intent of the user's message."
    )
    rewritten_query: str = Field(
        description=(
            "For KNOWLEDGE intent: rewrite the user's query into a clear, "
            "retrieval-friendly form using formal vocabulary in the SAME "
            "language as the user. Remove slang/colloquial verbs, fillers, "
            "and contractions. "
            "Examples: "
            "'kamu pernah ngajar berapa orang?' → 'berapa banyak peserta yang "
            "sudah belajar dari kursus saya?'. "
            "'modal itu apa sih?' → 'apa itu project Modal Cycle Zero?'. "
            "'ceritain dong project bts' → 'ceritakan tentang project Belajar "
            "Tulang Skuy (BTS)'. "
            "'gw udah kerja brp lama?' → 'sudah berapa lama saya bekerja "
            "sebagai Learning Designer?'. "
            "For non-KNOWLEDGE intents (GREETING / OFF_SCOPE / MALICIOUS): "
            "echo the user's query verbatim."
        )
    )


# ─── Nodes ───────────────────────────────────────────────────────────────────

async def _pre_processor(state: RAGState, config: RunnableConfig):
    """Classify intent. No history rewrite (stateless)."""
    user_msg = state["messages"][-1].content
    low_msg = user_msg.lower().strip()  # type: ignore[union-attr]  # langchain message.content is str at runtime

    GREETING_PREFIXES = (
        "halo", "hai", "hi", "hey", "hello",
        "pagi", "siang", "sore", "malam",
        "good morning", "good afternoon", "good evening",
        "selamat", "test", "siapa kamu", "siapa lo",
        "who are you", "introduce yourself",
    )
    if low_msg.startswith(GREETING_PREFIXES) and len(low_msg) < 60:
        return {"intent": "GREETING", "rewritten_query": user_msg}

    llm = get_preprocessor_llm()
    structured = llm.with_structured_output(AskferPreProcessorResult)
    try:
        result = await structured.ainvoke(
            [
                SystemMessage(content=PRE_PROCESSOR_PROMPT),
                HumanMessage(content=f"User query: {user_msg}"),
            ],
            config=config,
        )
        result = cast(AskferPreProcessorResult, result)  # structured output returns model at runtime
        intent = result.intent
        rewritten = result.rewritten_query.strip() or user_msg
    except Exception as exc:
        logger.warning(f"Askfer pre-processor failed, defaulting to KNOWLEDGE: {exc}")
        intent = "KNOWLEDGE"
        rewritten = user_msg

    logger.info(f"Askfer pre-processor: intent={intent}")
    return {"intent": intent, "rewritten_query": rewritten}


async def _handle_greeting(state: RAGState, config: RunnableConfig):
    """Friendly bilingual self-introduction.

    Loads `data/personal/profile.md` directly so the intro is grounded in the
    same single-source-of-truth as KNOWLEDGE answers — without going through
    retrieval (no embedding/Qdrant call needed for a greeting).

    CRITICAL: this handler must NEVER make factual claims about specific
    metrics, project names, dates, or numbers. The greeting path has no
    retrieved <retrieved_context>, so any specific claim would be a
    hallucination. Stay generic and pivot to inviting the user to ask.
    """
    profile_block = _load_profile_block()

    llm = get_llm()
    sys = (
        f"{ASKFER_PERSONA}\n"
        "GREETING-MODE — strict rules:\n"
        "1. Warmly introduce yourself as Ferdy, leading with your role focus "
        "(Learning Designer) from the <profile> block — never vague labels "
        "like 'professional in technology'. Match the user's language "
        "(English default), under 3 sentences, then invite them to ask about "
        "projects, tech stack, or experience.\n"
        "2. CRITICAL — do NOT answer factual questions here (specific "
        "projects, metrics, scores, dates, counts) and NEVER fabricate names "
        "or numbers. Only high-level role/team/location from <profile> is "
        "allowed. If asked something factual, redirect: 'Happy to dig into "
        "that — feel free to ask directly!' / 'Boleh banget, silakan tanya "
        "langsung ya!'"
        f"{profile_block}"
    )
    response = await llm.ainvoke(
        [SystemMessage(content=sys)] + list(state["messages"]),
        config=config,
    )
    return {"messages": [response]}


async def _handle_off_scope(state: RAGState, config: RunnableConfig):
    """Polite scope redirect — canned bilingual response (no LLM call).

    Bypassing the LLM here prevents two failure modes we kept seeing:
      1. LLM "halu" — attempts to answer the off-scope question before declining.
      2. Cost / latency for what is a fixed string anyway.
    Bahasa picked from heuristic on the user's last message.
    """
    user_msg = state["messages"][-1].content if state["messages"] else ""
    lang = _detect_user_language(user_msg)  # type: ignore[arg-type]  # langchain message.content is str at runtime
    if lang == "Indonesian":
        msg = (
            "Aku fokus bahas kerjaan profesional aja di sini — proyek, tech "
            "stack, dan pengalaman. Buat hal lain, kontak aku langsung via "
            "LinkedIn atau email ya."
        )
    else:
        msg = (
            "I keep this chat focused on my professional work — projects, "
            "tech stack, and experience. For other things, reach out "
            "directly via LinkedIn or email."
        )
    return {"messages": [AIMessage(content=msg)]}


async def _handle_malicious(state: RAGState, config: RunnableConfig):
    """Hard refusal — bilingual fixed responses, no LLM call."""
    user_msg = state["messages"][-1].content if state["messages"] else ""
    # Crude language detect — Indonesian common words
    is_id = any(t in user_msg.lower() for t in ("kamu", "saya", "aku", "siapa", "apa", "bagaimana"))  # type: ignore[union-attr]  # langchain message.content is str at runtime
    if is_id:
        msg = "Maaf, aku gak bisa bantu permintaan itu. Aku khusus jawab pertanyaan tentang project, skill, dan pengalaman profesional aku."
    else:
        msg = "Sorry, I can't help with that. I only answer questions about my projects, skills, and professional background."
    return {"messages": [AIMessage(content=msg)]}


async def _rag_node(state: RAGState, config: RunnableConfig):
    """Retrieve from Personal_Portfolio collection only.

    Always prepends the `doc_type=profile` chunk (if it exists) to the
    retrieved context. Profile is the single-source-of-truth for "who is
    Ferdy" answers — without this prepend, BM25 ranking on a name-heavy
    query like "ceritain soal Ferdy" gets dominated by CV chunks which
    repeat the name in every page header.
    """
    from app.retrieval.hybrid_retriever import hybrid_search
    from app.database.qdrant_client import get_qdrant_client
    from qdrant_client import models as qm

    query_to_search = state.get("rewritten_query") or state["messages"][-1].content
    try:
        result = await hybrid_search(
            query=query_to_search,  # type: ignore[arg-type]  # langchain message.content is str at runtime
            top_k=_settings.askfer_final_top_k,
            fetch_k=_settings.askfer_retrieval_top_k,
            collection=_settings.qdrant_personal_collection,
        )
        docs = result.chunks

        chunks = []
        for d in docs:
            m = d.metadata or {}
            chunks.append({
                "text": d.text,
                "doc_type": m.get("doc_type", ""),
                "project_slug": m.get("project_slug", ""),
                "project_url": m.get("project_url", ""),
                "title": d.title or m.get("title", ""),
                "score": round(d.score, 4) if d.score is not None else 0.0,
                "hybrid_score": round(d.hybrid_score, 4) if d.hybrid_score is not None else 0.0,
                "dense_score": round(d.dense_score, 4) if d.dense_score is not None else 0.0,
                "source": d.source or m.get("source", "Unknown"),
                "document_id": d.document_id or m.get("document_id", "Unknown"),
            })

        # Always-prepend the profile chunk if it isn't already in the top-K.
        if not any(c.get("doc_type") == "profile" for c in chunks):
            try:
                qdrant = get_qdrant_client()
                profile_pts, _ = await qdrant.client.scroll(
                    collection_name=_settings.qdrant_personal_collection,
                    scroll_filter=qm.Filter(
                        must=[qm.FieldCondition(key="doc_type", match=qm.MatchValue(value="profile"))]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False,
                )
                if profile_pts:
                    import json as _json
                    payload = profile_pts[0].payload or {}
                    nc_raw = payload.get("_node_content")
                    text = ""
                    if nc_raw:
                        try:
                            nc = _json.loads(nc_raw)
                            text = nc.get("text", "") or ""
                        except Exception:
                            text = ""
                    if text:
                        chunks.insert(0, {
                            "text": text,
                            "doc_type": "profile",
                            "project_slug": "",
                            "project_url": "",
                            "title": "About Ferdy",
                            "score": 1.0,
                            "hybrid_score": 1.0,
                            "source": "portfolio://profile",
                            "document_id": str(profile_pts[0].id),
                        })
            except Exception as exc:
                logger.warning(f"Profile prepend failed (continuing without): {exc}")

        logger.info(f"Askfer rag_node retrieved {len(chunks)} chunks for: {query_to_search[:60]}")
        return {"retrieved_context": chunks}
    except Exception as exc:
        logger.error(f"Askfer rag_node failed: {exc}")
        raise RuntimeError(f"Askfer retrieval failed: {exc}") from exc


async def _generate_node(state: RAGState, config: RunnableConfig):
    """Generate from retrieved context. No history/LTM/preferences interpolation."""
    chunks = state.get("retrieved_context") or []

    if chunks:
        # Differential per-doc-type caps. Knowledge files can be long (1.5k+
        # tokens each); without a cap a single Bloom or ADDIE query can blow
        # past 3k tokens of context. Overview is the project index — needs to
        # be mostly intact. Profile is short already so soft cap is safe.
        DOC_TYPE_CAPS: dict[str, int] = {
            "knowledge": 1200,
            "overview": 1500,
            "profile": 1500,
        }
        default_cap = _settings.askfer_chunk_text_max_chars
        context_lines = []
        for i, c in enumerate(chunks, 1):
            doc_type = c.get("doc_type") or "doc"
            label = c.get("title") or c.get("project_slug") or c.get("source") or doc_type
            text = c.get("text") or ""
            cap = DOC_TYPE_CAPS.get(doc_type, default_cap)
            if len(text) > cap:
                text = text[:cap].rstrip() + "..."
            context_lines.append(f"[{i}] ({doc_type}) {label}\n{text}")
        context_str = "\n\n---\n\n".join(context_lines)
    else:
        context_str = "No relevant documents found."

    full_system = (
        f"{ASKFER_SYSTEM_PROMPT}"
        f"\n\n<retrieved_context>\n{context_str}\n</retrieved_context>"
    )

    # Inject explicit language directive based on the user's last message —
    # the persona prompt's STRICT MIRROR rule isn't reliably honored when the
    # retrieved context mixes ID/EN, so we pin the language deterministically.
    user_msg = state["messages"][-1].content if state["messages"] else ""
    lang = _detect_user_language(user_msg)  # type: ignore[arg-type]  # langchain message.content is str at runtime
    full_system += (
        f"\n\n<language_directive>\n"
        f"The user's message is in {lang}. Respond in {lang} ONLY. "
        f"Ignore the language(s) used in <retrieved_context>.\n"
        f"</language_directive>"
    )

    llm = get_llm()
    messages = [SystemMessage(content=full_system)] + list(state["messages"])
    response = await llm.ainvoke(messages, config=config)
    return {"messages": [response]}


# ─── Routing & assembly ─────────────────────────────────────────────────────

def _route_by_intent(state: RAGState) -> str:
    return state.get("intent") or "KNOWLEDGE"


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
