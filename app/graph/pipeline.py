"""
Optimized CAG pipeline - classify then generate over the active KB pack.

Architecture change vs prior RAG pattern:
  BEFORE: classifier → retrieval → generate_node
  AFTER:  classifier → generate_node with the stable CAG KB prefix
"""
import asyncio
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from sqlalchemy import update

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger

from app.config.settings import get_settings
from app.graph.state import CAGState
from app.llm.cag_client import OpenRouterUsage
from app.llm.client import _provider_extra_body, _shared_http_client, get_generate_llm, get_generate_llm_nostream
from app.llm.prompts import CHIT_CHAT_PROMPT, CONVERSATIONAL_PROMPT, SOCRATIC_PROMPT
from app.knowledge.kb_pack import extract_kb_topics, extract_kb_sections, extract_kb_h2_headings

_settings = get_settings()
_MOODLE_BASE = _settings.moodle_api_url.rstrip("/")


# ─── System Prompts ──────────────────────────────────────────────────────────

# ─── Nodes ───────────────────────────────────────────────────────────────────

from app.graph.sanitizer import (
    StreamLeakGuard,
    _AMARTHA_GLOSSARY,
    _COURSE_NUM_RE,
    _DIRECTIVE_LINE_RE,
    _GLOSSARY_PATTERN,
    _GLOSSARY_RE,
    _INLINE_CITE_RE,
    _LEAK_BLOCK_RE,
    _LEAK_CITATION_HEAD_RE,
    _LEAK_OPEN_TAG_RE,
    _MD_HEADING_RE,
    _META_CONTEXT_LINE_RE,
    _META_CONVO_RE,
    _OFFSCOPE_PARTIAL_RE,
    _OFFSCOPE_RE,
    _apply_glossary,
    _normalize_dashes,
    _sanitize_answer,
)


async def _incr_parse_failure_metric() -> None:
    """Fire-and-forget counter for pre-processor JSON parse failures (C2).

    Bucketed by UTC date so the key self-expires (7-day retention) and gives a
    per-day failure rate that ops can scrape with a single SCAN/GET. Never
    raises — a metrics write must never break the request path. A rising count
    here means the pre-processor is failing to classify cleanly often enough to
    fall back to a default intent, i.e. silent quality decay.
    """
    try:
        from datetime import datetime, timezone

        from app.database.redis_client import get_redis_client

        redis = get_redis_client()
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        key = f"rag:metrics:preprocess_parse_failure:{day}"
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 7 * 24 * 3600)
        await pipe.execute()
    except Exception:
        pass







async def _log_cache_usage(response: Any, call_name: str, turn_id=None, started_at=None) -> None:
    """Log OpenRouter/Gemini prompt-cache hit info for ONE LLM call.

    The chat path previously logged NOTHING about cache effectiveness, so a
    cache regression (a prompt dropping below the provider's cache-min token
    threshold, or a cache_control breakpoint silently not honored) was
    invisible — only inferrable from the OpenRouter dashboard. This surfaces
    cached-prompt-token counts per call so we can SEE whether the cache is hit.
    Best-effort: never raises.

    LangChain and the raw gateway expose this differently, so check both:
      - usage_metadata.input_token_details.cache_read  (LangChain-normalized)
      - response_metadata.token_usage.prompt_tokens_details.cached_tokens (raw)
    """
    try:
        cached = 0
        prompt = 0
        um = getattr(response, "usage_metadata", None) or {}
        if um:
            prompt = um.get("input_tokens", 0) or 0
            details = um.get("input_token_details") or {}
            cached = details.get("cache_read", 0) or 0
        rm = getattr(response, "response_metadata", None) or {}
        tu = rm.get("token_usage") or {}
        if not cached or not prompt:
            prompt = prompt or (tu.get("prompt_tokens", 0) or 0)
            ptd = tu.get("prompt_tokens_details") or {}
            cached = cached or (ptd.get("cached_tokens", 0) or 0)
        completion = int((um or {}).get("output_tokens", 0) or 0)
        if not completion:
            completion = int(tu.get("completion_tokens", 0) or 0)
        model = rm.get("model_name") or rm.get("model") or "unknown"
        provider = rm.get("provider_name") or rm.get("provider") or _infer_provider(model)

        pct = (cached / prompt * 100) if prompt else 0.0
        duration_s = round(time.monotonic() - started_at, 4) if started_at else None

        logger.info(
            "LLM cache usage [{}]: cached={}/{} prompt tok ({:.0f}%) completion={} "
            "model={} provider={} duration={}s turn={}",
            call_name, cached, prompt, pct, completion,
            model, provider, duration_s, (turn_id or "-")[:8],
        )
        cost = float(tu.get("cost", 0.0) or 0.0)
        if turn_id:
            try:
                await _persist_or_cache_metrics(
                    turn_id=turn_id,
                    prompt=int(prompt),
                    cached=int(cached),
                    completion=completion,
                    provider=provider,
                    duration_s=duration_s,
                    cost=cost,
                )
            except Exception as e:
                logger.warning("_persist_or_cache_metrics failed for turn={}: {}", (turn_id or "-")[:8], e)
    except Exception as e:
        logger.warning("_log_cache_usage failed [{}]: {}", call_name, e)


async def _persist_or_cache_metrics(
    *,
    turn_id: str,
    prompt: int,
    cached: int,
    completion: int,
    provider: str,
    duration_s: float | None,
    cost: float | None,
) -> None:
    """UPDATE agent_logs row matching turn_id with OpenRouter cache metrics.

    Used by the Streamlit dashboard to show OR cache hit/miss + cached
    prompt-token counts (replacing the old Redis semantic-cache hit-rate).
    """
    try:
        from app.database.postgres import AsyncSessionLocal
        from app.database.models import AgentLog

        async with AsyncSessionLocal() as s:
            await s.execute(
                update(AgentLog)
                .where(AgentLog.turn_id == turn_id)
                .values(
                    or_prompt_tokens=prompt,
                    or_cached_tokens=cached,
                    or_completion_tokens=completion,
                    or_provider=provider,
                    or_duration_s=duration_s,
                    or_cost=cost,
                )
            )
            await s.commit()
    except Exception as e:
        logger.warning("_persist_or_cache_metrics failed for turn={}: {}", turn_id[:8], e)


def _infer_provider(model: str) -> str:
    """Best-effort provider inference from model id, e.g. 'google/gemini-2.5-flash' -> 'google'."""
    if "/" in model:
        return model.split("/", 1)[0]
    return "openrouter"


async def _pre_processor(state: CAGState, config: RunnableConfig):
    """Lightweight pre-step — NO LLM call. Decides retrieval vs no-retrieval.

    Ava is one conversational LLM call (see _generate_node + CONVERSATIONAL_PROMPT).
    This node uses the deterministic regex Tier-1 classifier ONLY to route — it
    never emits a canned reply (that was the old "yang benerlah → identity intro"
    misroute). Three buckets:
      - MALICIOUS (injection/jailbreak) → canned refusal, no retrieval, no LLM.
      - CHIT-CHAT (GREETING / AMBIGUOUS / OFF_SCOPE / TOPIC_LIST): a salutation,
        identity Q, vague filler, off-topic, or "what topics exist" — these need
        NO knowledge-base lookup, so we SKIP retrieval and go straight to the
        conversational generate node with NO <context>. That prevents an
        irrelevant chunk from being dumped into a greeting/vague turn, and lets
        the prompt ask a clarifying question on ambiguous input instead of
        guessing. Cheaper too (no retrieval round-trip).
      - KNOWLEDGE (regex returns None — a real question): retrieve, then generate.

    `intent` carries the regex label so chat.py's existing cache/eval gates
    (which already exclude GREETING/AMBIGUOUS/etc.) keep working. `intent_scores`
    stays a vestigial derived dict for the DB/logging schema.
    """
    from app.graph.intent_rules import classify as rule_classify

    messages = state["messages"]
    user_msg = messages[-1].content
    user_msg_str = user_msg if isinstance(user_msg, str) else str(user_msg)

    rule_intent = rule_classify(user_msg_str)

    # ── Injection / jailbreak guard ─────────────────────────────────────────
    if rule_intent == "MALICIOUS":
        logger.info("Pre-processor: injection detected → MALICIOUS")
        return {
            "intent": "MALICIOUS",
            "rewritten_query": user_msg_str,
            "retrieval_query": user_msg_str,
            "intent_scores": {"needs_lookup": 0.0, "needs_reasoning": 0.0, "needs_empathy": 0.0, "needs_safety_escalation": 0.0, "learning_context": 0.0},
            "gate_score": None,
        }

    # NOTE: "apa aja di <section>" text-detection was REMOVED — structured
    # navigation (which section, which item) now lives in the UI: a topic-list
    # button opens a section/item picker, and clicking an item sends a normal
    # KNOWLEDGE query ("jelaskan tentang <item>"). Free-text section parsing was
    # fragile (cross-language, content-noun collisions) and is no longer needed.
    # The full topic list ("topik apa aja") still routes via the regex/semantic
    # TOPIC_LIST path below.

    # ── SECTION_DRILLDOWN ───────────────────────────────────────────────────
    # Refinement (Jun 2026): "topic apa aja" -> TOPIC_LIST,
    # "product amartha apaan" -> SECTION_DRILLDOWN. If the shape matches AND we
    # can resolve the section from query (token match) OR history (deictic ordinal),
    # route to SECTION_DRILLDOWN immediately, regardless of the initial rule_intent.
    if _is_section_drilldown_shape(user_msg_str) and _extract_topic_list_from_history(messages):
        try:
            resolved_role = resolve_user_role(state.get("user_context"))
            _sm = await _load_section_map(resolved_role)
        except Exception:
            _sm = {}
        _resolved, _respath = _resolve_drilldown_section(user_msg_str, messages, _sm)
        if _resolved:
            logger.info(
                f"Pre-processor: intent refined -> SECTION_DRILLDOWN "
                f"(section={_resolved!r}, via {_respath!r})"
            )
            state["drilldown_section"] = _resolved
            state["drilldown_resolution"] = _respath
            return {
                "intent": "SECTION_DRILLDOWN",
                "rewritten_query": user_msg_str,
                "retrieval_query": user_msg_str,
                "intent_scores": {"needs_lookup": 0.0, "needs_reasoning": 0.0, "needs_empathy": 0.0, "needs_safety_escalation": 0.0, "learning_context": 0.0},
                "gate_score": None,
                "drilldown_section": _resolved,
                "drilldown_resolution": _respath,
            }
        logger.info(
            "Pre-processor: drilldown shape matched but no section resolved - falling back"
        )

    # ── Chit-chat / no-lookup intents → skip retrieval entirely ─────────────
    if rule_intent in ("GREETING", "AMBIGUOUS", "OFF_SCOPE", "TOPIC_LIST"):
        logger.info(f"Pre-processor: {rule_intent} → no retrieval, straight to generate")
        return {
            "intent": rule_intent,
            "rewritten_query": user_msg_str,
            "retrieval_query": user_msg_str,
            "intent_scores": {"needs_lookup": 0.0, "needs_reasoning": 0.0, "needs_empathy": 0.0, "needs_safety_escalation": 0.0, "learning_context": 0.0},
            "gate_score": None,
        }

    # ── KNOWLEDGE: a real question → retrieve, then generate ────────────────
    # Query Expansion (HyDE): We rewrite conversational queries (up to 250 chars)
    # into focused, keyword-rich search queries. This ensures queries like "terlambat bayar 15 hari"
    # match documents like "Definisi DPD PAR 3" without needing hardcoded Moodle keywords.
    # It also handles coreference resolution for short follow-ups.
    retrieval_query = user_msg_str
    _msg_stripped = user_msg_str.strip()

    # Reuse pre-computed query rewrite if passed from chat.py to avoid duplicate LLM calls
    precomputed_queries = state.get("rewritten_queries")
    if precomputed_queries:
        if isinstance(precomputed_queries, list):
            retrieval_query = " | ".join(precomputed_queries)
        else:
            retrieval_query = precomputed_queries
        logger.info(f"Pre-processor: reusing pre-computed query rewrite: {retrieval_query[:60]}")


    # ── Coaching (Socratic) promotion ───────────────────────────────────────
    # When the user has the coaching toggle ON (state.coaching_mode), a real
    # question becomes a COACHING turn instead of KNOWLEDGE. generate_node then
    # uses SOCRATIC_PROMPT — which opens diagnostic/reasoning asks with ONE
    # grounded guiding question, but still answers pure factual lookups directly
    # (that fact-vs-diagnostic split is an LLM judgment in the prompt, not a
    # fragile regex here). Retrieval runs either way: a guiding question must be
    # grounded in the KB, not invented.
    intent = "COACHING" if state.get("coaching_mode") else "KNOWLEDGE"
    logger.info(f"Pre-processor: intent={intent} retrieval='{retrieval_query[:60]}...'")

    # Ensure rewritten_queries list and retrieval_query are structured correctly in state.
    # Supports both newline-separated (rewrite LLM output per REWRITE_PROMPT rule #9)
    # and pipe-separated (precomputed join from chat.py " | ".join).
    if not retrieval_query:
        queries_list = [user_msg_str]
    elif "\n" in retrieval_query:
        queries_list = [q.strip() for q in retrieval_query.replace("\r\n", "\n").split("\n") if q.strip()]
    else:
        queries_list = [q.strip() for q in retrieval_query.split(" | ") if q.strip()]
    primary_query = queries_list[0] if queries_list else retrieval_query

    return {
        "intent": intent,
        "rewritten_query": retrieval_query,
        "retrieval_query": primary_query,
        "rewritten_queries": queries_list,
        "intent_scores": {
            "needs_lookup": 1.0,
            "needs_reasoning": 1.0 if intent == "COACHING" else 0.0,
            "needs_empathy": 0.0,
            "needs_safety_escalation": 0.0,
            "learning_context": 0.0,
        },
        "gate_score": None,
    }


async def _handle_malicious(state: CAGState, config: RunnableConfig):
    """Canned refusal for jailbreak/prompt-injection (deterministic guard).

    The only canned handler kept after the conversational collapse. _is_injection
    in _pre_processor routes here BEFORE any retrieval/LLM, so an injection attempt
    never reaches the conversational prompt. No LLM.
    """
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=(
        "Maaf, tugasku khusus untuk membantu seputar materi Amarthapedia dan "
        "kebijakan internal Amartha. Ada yang bisa kubantu seputar itu?"
    ))]}



def _window_generate_history(messages: list, max_fresh_turns: int, max_ai_chars: int) -> list:
    """Trim the message history fed to generate_node.

    chat.py hands generate_node the current query (always the LAST message)
    preceded by up to `get_or_summarize_history`'s window of completed turns;
    everything older is already folded into the rolling summary
    (<previous_context>). So feeding the full turn list here double-pays:
    the summary covers the old turns AND the raw turns are still attached.

    Two cuts:
      1. Keep only the last `max_fresh_turns` completed turns (= 2*N messages)
         before the current query, then re-append the current query.
      2. Cap each AIMessage's content to `max_ai_chars` — prior AI replies can
         be long, and only their gist (entity names, the topic in play) matters
         for follow-up resolution. User turns are left intact (short + carry the
         actual intent).

    Returns a NEW list with NEW capped AIMessage objects, so state["messages"]
    (consumed downstream for history/cache persistence) is never mutated.
    """
    if not messages:
        return messages
    current = messages[-1]
    prior = messages[:-1]
    if max_fresh_turns > 0 and len(prior) > max_fresh_turns * 2:
        prior = prior[-(max_fresh_turns * 2):]

    windowed: list = []
    for m in prior:
        if isinstance(m, AIMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            if max_ai_chars and len(content) > max_ai_chars:
                windowed.append(AIMessage(content=content[:max_ai_chars].rstrip() + "…"))
                continue
        windowed.append(m)
    windowed.append(current)
    return windowed


_COURSE_CACHE_TTL_SECONDS = 600  # 10 minutes
_course_cache: dict[str, Any] = {"courses": [], "expires_at": 0.0}
_course_cache_lock: asyncio.Lock | None = None


def _get_course_cache_lock() -> asyncio.Lock:
    """Lazy-init the cache lock (must be created inside a running event loop)."""
    global _course_cache_lock
    if _course_cache_lock is None:
        _course_cache_lock = asyncio.Lock()
    return _course_cache_lock


async def _load_course_names() -> list[str]:
    """Distinct TOPIC labels from the active CAG KB pack, TTL-cached (10min)."""
    import time as _time

    now = _time.time()
    if now < _course_cache["expires_at"] and _course_cache["courses"]:
        return _course_cache["courses"]

    lock = _get_course_cache_lock()
    async with lock:
        now = _time.time()
        if now < _course_cache["expires_at"] and _course_cache["courses"]:
            return _course_cache["courses"]

        try:
            cag_kb_text = await _load_active_cag_kb_text()
            courses = extract_kb_topics(cag_kb_text) if cag_kb_text else []
        except Exception as exc:
            logger.warning(f"Topic-name load failed: {exc}")
            return []

        _course_cache["courses"] = courses
        _course_cache["expires_at"] = now + _COURSE_CACHE_TTL_SECONDS
        return courses


_doc_cache: dict[str, Any] = {"titles": [], "expires_at": 0.0}


async def _load_doc_titles() -> list[str]:
    """Distinct document filenames from the active CAG KB pack, TTL-cached (10min)."""
    import time as _time

    now = _time.time()
    if now < _doc_cache["expires_at"] and _doc_cache["titles"]:
        return _doc_cache["titles"]

    try:
        cag_kb_text = await _load_active_cag_kb_text()
        from app.knowledge.kb_pack import extract_kb_filenames
        titles = extract_kb_filenames(cag_kb_text) if cag_kb_text else []
    except Exception as exc:
        logger.warning(f"Doc-title load failed: {exc}")
        return []

    _doc_cache["titles"] = titles
    _doc_cache["expires_at"] = now + 600.0
    return titles


# ── Section → items map (for "apa aja di <section>" questions) ────────────────
_section_map_cache: dict[str, Any] = {"map": {}, "expires_at": 0.0}
_section_map_lock: asyncio.Lock | None = None


def _get_section_map_lock() -> asyncio.Lock:
    global _section_map_lock
    if _section_map_lock is None:
        _section_map_lock = asyncio.Lock()
    return _section_map_lock


from app.graph.drilldown import (
    _ORDINAL_TO_INT,
    _SECTION_DRILLDOWN_PHRASES,
    _SECTION_NAME_STOPWORDS,
    _detect_section_from_query,
    _extract_sections_from_topic_list,
    _extract_topic_list_from_history,
    _flatten_message_content,
    _fuzzy_token_match,
    _has_topic_list_marker,
    _is_section_drilldown_shape,
    _levenshtein,
    _normalize_section_tokens,
    _resolve_drilldown_section,
    _resolve_section_ordinal,
    _score_query_against_section,
)

_h2_topics_cache: dict[str, Any] = {"map": {}, "expires_at": 0.0}


async def _load_role_kb_cache(
    cache_store: dict[str, Any],
    user_role: str,
    extractor_fn,
    default_factory,
    label: str,
):
    """Generic role-filtered KB metadata loader with 10-minute TTL cache."""
    import time as _time

    role = user_role.upper().strip()
    now = _time.time()

    if now >= cache_store["expires_at"]:
        cache_store["map"] = {}
        cache_store["expires_at"] = 0.0

    if role in cache_store["map"]:
        return cache_store["map"][role]

    lock = _get_section_map_lock()
    async with lock:
        now = _time.time()
        if now >= cache_store["expires_at"]:
            cache_store["map"] = {}
            cache_store["expires_at"] = 0.0

        if role in cache_store["map"]:
            return cache_store["map"][role]

        try:
            full_kb = await _load_active_cag_kb_text()
            cag_kb_text = _filter_kb_by_role(full_kb, role) if full_kb else ""
            result = extractor_fn(cag_kb_text) if cag_kb_text else default_factory()
        except Exception as exc:
            logger.warning(f"{label} load failed for role {role}: {exc}")
            return default_factory()

        cache_store["map"][role] = result
        cache_store["expires_at"] = now + _COURSE_CACHE_TTL_SECONDS
        return result


async def _load_section_map(user_role: str = "ALL") -> dict[str, list[str]]:
    """Map each Moodle SECTION -> its item list, filtered by role, TTL-cached (10min)."""
    return await _load_role_kb_cache(_section_map_cache, user_role, extract_kb_sections, dict, "Section-map")


async def _load_h2_topics(user_role: str = "ALL") -> list[str]:
    """Return distinct H2 (##) headings from active KB filtered by role, TTL-cached (10min)."""
    return await _load_role_kb_cache(_h2_topics_cache, user_role, extract_kb_h2_headings, list, "H2-topics")


_active_kb_cache: dict[str, Any] = {"hash": "", "content": "", "expires_at": 0.0}
_role_kb_cache: dict[tuple[int, str], str] = {}
_kb_cache_lock: asyncio.Lock | None = None


def _get_kb_cache_lock() -> asyncio.Lock:
    global _kb_cache_lock
    if _kb_cache_lock is None:
        _kb_cache_lock = asyncio.Lock()
    return _kb_cache_lock


def clear_cag_kb_cache() -> None:
    _active_kb_cache.update({"hash": "", "content": "", "expires_at": 0.0})
    _role_kb_cache.clear()
    _course_cache.update({"courses": [], "expires_at": 0.0})
    _section_map_cache.update({"map": {}, "expires_at": 0.0})
    _h2_topics_cache.update({"map": {}, "expires_at": 0.0})


_KB_HASH_CHECK_TTL_SECONDS = 15.0


async def _load_active_cag_kb_text() -> str:
    from app.database.postgres import AsyncSessionLocal
    from app.knowledge.store import get_active_kb_hash, get_active_kb_pack

    now = time.time()
    cached_hash = _active_kb_cache.get("hash")
    cached_content = _active_kb_cache.get("content")
    if cached_hash and cached_content and now < _active_kb_cache.get("expires_at", 0.0):
        return cached_content

    async with _get_kb_cache_lock():
        now = time.time()
        cached_hash = _active_kb_cache.get("hash")
        cached_content = _active_kb_cache.get("content")
        if cached_hash and cached_content and now < _active_kb_cache.get("expires_at", 0.0):
            return cached_content

        try:
            async with AsyncSessionLocal() as session:
                if cached_hash and cached_content:
                    current_hash = await get_active_kb_hash(session, source=_settings.cag_kb_source)
                    if current_hash == cached_hash:
                        _active_kb_cache["expires_at"] = now + _KB_HASH_CHECK_TTL_SECONDS
                        return cached_content

                active = await get_active_kb_pack(session, source=_settings.cag_kb_source)
                if active:
                    if _active_kb_cache.get("hash") != active.kb_hash:
                        clear_cag_kb_cache()
                        _active_kb_cache.update({
                            "hash": active.kb_hash,
                            "content": active.content,
                            "expires_at": now + _KB_HASH_CHECK_TTL_SECONDS,
                        })
                    else:
                        _active_kb_cache["expires_at"] = now + _KB_HASH_CHECK_TTL_SECONDS
                    return _active_kb_cache["content"]
        except Exception as exc:
            logger.warning(f"Failed to load active cag kb text: {exc}")
        return _active_kb_cache.get("content", "")


@lru_cache(maxsize=64)
def _openrouter_prompt_session_id(*parts: str) -> str:
    stable_prefix = "\n".join(part for part in parts if part)
    if not stable_prefix:
        return ""
    return "ava-prefix-" + hashlib.sha256(stable_prefix.encode()).hexdigest()[:32]


def _with_openrouter_session(llm, session_id: str | None):
    session_id = str(session_id or "").strip()[:256]
    if not session_id:
        return llm
    if not hasattr(llm, "bind"):
        return llm
    extra_body = dict(getattr(llm, "extra_body", None) or {})
    return llm.bind(extra_body={**extra_body, "session_id": session_id})


def _openrouter_messages(messages: list) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            role = "system"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            role = "user"
        content = getattr(msg, "content", msg)
        if isinstance(content, list):
            content = "".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
        out.append({"role": role, "content": str(content)})
    return out


def resolve_user_role(user_context: dict | None) -> str:
    if not user_context:
        return "ALL"
        
    location = str(user_context.get("location") or "").strip().upper()
    grade = str(user_context.get("grade") or "").strip().upper()
    user_role = str(user_context.get("role") or "").strip().upper()

    # Management Trainee or adapted role check (BP, BM, RM, AM, HMB)
    if user_role in ("RM", "AM", "HMB", "BM", "BP"):
        return user_role

    # 1. HO (Head Office) -> HO (Head Office users get full access)
    if location == "HO":
        return "HO"
        
    # 2. FO Location: Extract first token prefix for Field Officers (BP, BM, RM, AM, HMB)
    tokens = re.split(r'[\s\-_/]+', grade)
    grade_prefix = tokens[0] if tokens else ""
    if grade_prefix in ("RM", "AM", "HMB", "BM", "BP"):
        return grade_prefix
        
    # Fallback to location
    if location == "FO":
        return "FO"
        
    return "ALL"


_ROLES_ATTR_Q_RE = re.compile(r'roles=["\']([^"\']*)["\']', re.IGNORECASE)
_ROLE_BLOCK_RE = re.compile(r'<role_block\s+([^>]*?)>(.*?)</role_block>', re.DOTALL | re.IGNORECASE)
_COMMENT_ROLE_RE = re.compile(r'<!--\s*role:\s*([^>]*?)\s*-->(.*?)<!--\s*/role\s*-->', re.DOTALL | re.IGNORECASE)
_DOC_BLOCK_RE = re.compile(r'(<doc\s+([^>]*?)>)(.*?)(</doc>)', re.DOTALL)
_ROLES_ATTR_DQ_RE = re.compile(r'roles="([^"]*)"')
_ID_ATTR_DQ_RE = re.compile(r'id="([^"]*)"')
_KB_INDEX_RE = re.compile(r'<kb_index>(.*?)</kb_index>', re.DOTALL)
_KB_INDEX_ENTRY_RE = re.compile(r'(-\s+\[(DOC-\d+)\](?:(?!-\s+\[DOC-).)*)', re.DOTALL)
_KB_VERSION_RE = re.compile(r'<knowledge_base\s+([^>]*?)>')


def _filter_inner_role_blocks(text: str, role: str) -> str:
    """Filter inner <role_block roles="..."> and <!-- role: ... --> blocks inside text."""
    if not text or not role:
        return text
        
    role = role.upper().strip()

    # 1. Handle <role_block roles="...">...</role_block>
    def _replace_role_block(m: re.Match) -> str:
        attrs = m.group(1)
        content = m.group(2)
        match = _ROLES_ATTR_Q_RE.search(attrs)
        if match:
            block_roles = [r.strip().upper() for r in match.group(1).split(",")]
            if role in ("ALL", "HO") or "ALL" in block_roles or role in block_roles:
                return content.strip()
            return ""
        return content.strip()

    text = _ROLE_BLOCK_RE.sub(_replace_role_block, text)

    # 2. Handle <!-- role: BP,BM --> ... <!-- /role -->
    def _replace_comment_block(m: re.Match) -> str:
        roles_str = m.group(1)
        content = m.group(2)
        block_roles = [r.strip().upper() for r in roles_str.split(",")]
        if role in ("ALL", "HO") or "ALL" in block_roles or role in block_roles:
            return content.strip()
        return ""

    text = _COMMENT_ROLE_RE.sub(_replace_comment_block, text)
    return text


def _filter_kb_by_role(kb_text: str, user_role: str) -> str:
    if not kb_text or not user_role:
        return kb_text
    
    role = user_role.upper().strip()
    cache_key = (hash(kb_text), role)
    cached_filtered = _role_kb_cache.get(cache_key)
    if cached_filtered is not None:
        return cached_filtered
    
    # 1. Find and filter the <doc> blocks first to collect allowed doc IDs
    docs_found = _DOC_BLOCK_RE.findall(kb_text)
    if not docs_found:
        res = _filter_inner_role_blocks(kb_text, role)
        _role_kb_cache[cache_key] = res
        return res

    filtered_docs = []
    allowed_doc_ids = set()
    for header, attrs, content, footer in docs_found:
        id_match = _ID_ATTR_DQ_RE.search(attrs)
        doc_id = id_match.group(1) if id_match else ""
        
        match = _ROLES_ATTR_DQ_RE.search(attrs)
        if match:
            doc_roles = [r.strip().upper() for r in match.group(1).split(",")]
            if "ALL" in doc_roles or role in doc_roles:
                filtered_content = _filter_inner_role_blocks(content, role)
                filtered_docs.append(f"{header}{filtered_content}{footer}")
                if doc_id:
                    allowed_doc_ids.add(doc_id)
        else:
            filtered_content = _filter_inner_role_blocks(content, role)
            filtered_docs.append(f"{header}{filtered_content}{footer}")
            if doc_id:
                allowed_doc_ids.add(doc_id)
                
    # 2. Extract and filter the <kb_index> block based on allowed_doc_ids
    kb_index_match = _KB_INDEX_RE.search(kb_text)
    kb_index = ""
    if kb_index_match:
        index_content = kb_index_match.group(1)
        filtered_entries = []
        for entry, doc_id in _KB_INDEX_ENTRY_RE.findall(index_content):
            if doc_id in allowed_doc_ids:
                filtered_entries.append(entry.strip())
        
        if filtered_entries:
            kb_index = "<kb_index>\n" + "\n".join(filtered_entries) + "\n</kb_index>"
            
    # 3. Get the version attribute if present to reconstruct the root tag
    version_match = _KB_VERSION_RE.search(kb_text)
    root_attrs = version_match.group(1) if version_match else ""
    
    # Reassemble
    out = [f"<knowledge_base {root_attrs}>".strip()]
    if kb_index:
        out.append(kb_index)
    out.extend(filtered_docs)
    out.append("</knowledge_base>")
    
    res = "\n".join(out)
    if len(_role_kb_cache) >= 64:
        _role_kb_cache.pop(next(iter(_role_kb_cache)), None)
    _role_kb_cache[cache_key] = res
    return res


def _format_user_context_block(uctx: dict) -> str:
    """Format user context into structured profile & performance/KPI metrics block."""
    if not uctx:
        return ""
    
    profile_lines = []
    regional_data = {}
    other_kpis = []

    standard_keys = {
        "name", "gender", "dept", "position", "grade", "location", "point", 
        "area", "regional", "pulau", "username", "full_name", "jabatan", "cakupan"
    }

    # Extract profile keys case-insensitively while preserving insertion order
    profile_dict = {}
    for k, v in uctx.items():
        k_norm = str(k).lower().strip()
        if v is not None and str(v).strip() != "" and k_norm in standard_keys:
            profile_dict[k_norm] = str(v).strip()

    for k, v in profile_dict.items():
        label = k.replace("_", " ").title()
        profile_lines.append(f"- {label}: {v}")

    for k, v in uctx.items():
        k_str = str(k).strip()
        k_norm = k_str.lower()
        if k_norm in standard_keys or k_norm in profile_dict or k_norm == "role" or v is None or str(v).strip() == "":
            continue
            
        if " - " in k_str:
            region, metric = k_str.split(" - ", 1)
            region_name = region.strip()
            metric_name = metric.strip()
            # Clean KPI acronym in metric name
            metric_name = re.sub(r'(?i)\bkpi\b', 'KPI', metric_name)
            if region_name not in regional_data:
                regional_data[region_name] = []
            regional_data[region_name].append(f"  • {metric_name}: {v}")
        else:
            label = k_str.replace("_", " ").strip()
            if label.lower().startswith("kpi "):
                label = label[4:].strip()
            elif label.lower().startswith("kpi_"):
                label = label[4:].strip()
            elif label.lower() == "kpi":
                label = "KPI"
            label = label[0].upper() + label[1:] if label else k_str
            other_kpis.append(f"- {label}: {v}")

    if not profile_lines and not regional_data and not other_kpis:
        return ""

    content_parts = []
    if profile_lines:
        content_parts.append("Profile:\n" + "\n".join(profile_lines))

    if regional_data:
        reg_lines = ["[Metrik Performa & KPI Terstruktur/Tim]:"]
        for region, metrics in regional_data.items():
            reg_lines.append(f"- {region}:\n" + "\n".join(metrics))
        content_parts.append("\n".join(reg_lines))

    if other_kpis:
        content_parts.append("[Metrik Performa & KPI Lainnya]:\n" + "\n".join(other_kpis))

    ctx_body = "\n\n".join(content_parts)
    return (
        "\n\n<user_context>\nYou are speaking with the following user. "
        "Adapt your answers to their context, but DO NOT call or greet them by their first name "
        "repeatedly at the beginning of sentences or transitions, and DO NOT abbreviate or shorten "
        "their Position/jabatan title (always write out the full position title as specified below):\n"
        + ctx_body
        + "\n</user_context>"
    )


async def _build_generate_messages(state: CAGState) -> tuple[list, str]:
    """Build the exact prompt messages and OpenRouter session_id for generation."""
    summary = state.get("conversation_summary") or ""
    profile = state.get("user_profile") or {}
    intent = state.get("intent") or "KNOWLEDGE"
    user_context = state.get("user_context") or {}

    cag_kb_text = ""
    if intent in ("KNOWLEDGE", "COACHING"):
        try:
            full_kb = await _load_active_cag_kb_text()
            resolved_role = resolve_user_role(user_context)
            cag_kb_text = _filter_kb_by_role(full_kb, resolved_role)
            logger.info(f"_generate_node: Filtered KB for role={resolved_role}: {len(cag_kb_text)} chars (out of {len(full_kb)})")
        except Exception as exc:
            logger.warning(f"CAG KB pack load failed: {exc}")

    context_section = ""
    if intent in ("KNOWLEDGE", "COACHING") and not cag_kb_text:
        context_section = (
            "\n\n<knowledge_base_missing>\n"
            "No active CAG knowledge base pack is available. Ask an admin to run Moodle KB sync first."
            "\n</knowledge_base_missing>"
        )

    topics_section = ""
    if intent == "TOPIC_LIST":
        try:
            course_names = await _load_course_names()
        except Exception:
            course_names = []
        topics_section = (
            "\n\n<available_topics>\n"
            + ("\n".join(f"- {c}" for c in course_names) if course_names else "(could not load topic list right now)")
            + "\n</available_topics>"
        )

    section_section = ""
    drilldown_sec = state.get("drilldown_section")
    if drilldown_sec:
        try:
            resolved_role = resolve_user_role(user_context)
            items = (await _load_section_map(resolved_role)).get(drilldown_sec, [])
        except Exception:
            items = []
        if items:
            section_section = (
                f'\n\n<section_materials section="{drilldown_sec}">\n'
                + "\n".join(f"- {it}" for it in items)
                + "\n</section_materials>"
            )
            logger.info(
                f"SECTION_DRILLDOWN inject: section={drilldown_sec!r}, "
                f"{len(items)} items, via={state.get('drilldown_resolution')!r}"
            )
        else:
            logger.warning(
                f"SECTION_DRILLDOWN resolved section={drilldown_sec!r} but section_map has no items"
            )

    ltm_section = ""
    learning_summary = (profile.get("learning_summary") or "").strip()
    last_topics = profile.get("last_topics") or []
    if learning_summary or last_topics:
        history_lines = []
        if learning_summary:
            history_lines.append(f"Ringkasan progres & konteks belajar user: {learning_summary}")
        if last_topics:
            topics_str = ", ".join(last_topics) if isinstance(last_topics, list) else str(last_topics)
            history_lines.append(f"Topik utama yang pernah dibahas: {topics_str}")
        ltm_section = "\n\n<user_history>\n" + "\n".join(history_lines) + "\n</user_history>"

    summary_section = f"\n\n<previous_context>\n{summary}\n</previous_context>" if summary else ""
    user_ctx_section = _format_user_context_block(user_context)
    dynamic_tail = f"{user_ctx_section}{ltm_section}{summary_section}{topics_section}{section_section}{context_section}".strip()

    windowed_messages = _window_generate_history(
        list(state["messages"]),
        max_fresh_turns=_settings.max_fresh_turns,
        max_ai_chars=_settings.max_history_ai_chars,
    )

    if intent == "COACHING":
        system_prompt_text = SOCRATIC_PROMPT
    elif intent in ("GREETING", "AMBIGUOUS", "OFF_SCOPE"):
        system_prompt_text = CHIT_CHAT_PROMPT
    else:
        system_prompt_text = CONVERSATIONAL_PROMPT

    msgs: list = [SystemMessage(content=system_prompt_text)]
    if cag_kb_text:
        msgs.append(SystemMessage(content=cag_kb_text))
    if dynamic_tail:
        msgs.append(HumanMessage(content=dynamic_tail))
    msgs += windowed_messages
    return msgs, _openrouter_prompt_session_id(system_prompt_text, cag_kb_text)


async def stream_openrouter_generate(state: CAGState, config: RunnableConfig | None = None):
    """Stream generate directly from OpenRouter so final usage.cost is preserved."""
    messages, session_id = await _build_generate_messages(state)
    extra_body = _provider_extra_body(_settings.llm_model)
    if session_id:
        extra_body = {**extra_body, "session_id": session_id}
    body = {
        "model": _settings.llm_model,
        "messages": _openrouter_messages(messages),
        "temperature": _settings.generate_llm_temperature,
        "max_tokens": _settings.llm_max_tokens,
        "stream": True,
        **extra_body,
    }
    headers = {
        "Authorization": f"Bearer {_settings.openrouter_api_key}",
        "HTTP-Referer": "https://github.com/FerdyLazuardi/ai-trainer-cag",
        "X-Title": "CAG AI TRAINER (Generate)",
    }

    generation_id = None
    model = None
    provider = None
    sent_usage = False
    url = _settings.openrouter_base_url.rstrip("/") + "/chat/completions"
    async with _shared_http_client().stream("POST", url, headers=headers, json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            if line.startswith(":"):
                yield {"type": "ping"}
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                if not sent_usage and generation_id:
                    yield {
                        "type": "usage",
                        "usage": OpenRouterUsage(
                            provider=model or provider,
                            generation_id=generation_id,
                        ),
                    }
                break
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            generation_id = data.get("id") or generation_id
            model = data.get("model") or model
            provider = data.get("provider") or data.get("provider_name") or provider
            for choice in data.get("choices") or []:
                delta = choice.get("delta") or {}
                token = delta.get("content")
                if token:
                    yield {"type": "token", "text": token}
            usage = data.get("usage") or {}
            if usage:
                sent_usage = True
                details = usage.get("prompt_tokens_details") or {}
                yield {
                    "type": "usage",
                    "usage": OpenRouterUsage(
                        prompt_tokens=int(usage.get("prompt_tokens") or 0),
                        cached_tokens=int(details.get("cached_tokens") or 0),
                        completion_tokens=int(usage.get("completion_tokens") or 0),
                        provider=model or provider,
                        cost=float(usage.get("cost") or 0.0),
                        generation_id=generation_id,
                    ),
                }


async def _generate_node(state: CAGState, config: RunnableConfig):
    """Single conversational LLM call for the non-stream graph execution path."""
    msgs, openrouter_session_id = await _build_generate_messages(state)
    llm = _with_openrouter_session(
        get_generate_llm_nostream(),
        openrouter_session_id,
    )

    _t0 = time.monotonic()
    response = await llm.ainvoke(msgs, config=config)
    await _log_cache_usage(
        response,
        "generate",
        turn_id=state.get("turn_id") if isinstance(state, dict) else None,
        started_at=_t0,
    )

    raw = response.content if hasattr(response, "content") else str(response)
    intent = state.get("intent") or "KNOWLEDGE"
    off_scope_detected = False
    if isinstance(raw, str):
        if intent == "OFF_SCOPE" or _OFFSCOPE_RE.search(raw):
            off_scope_detected = True
        cleaned = _sanitize_answer(raw)
        if cleaned != raw:
            logger.warning(
                "generate_node: stripped leaked instruction block from LLM output "
                f"(orig_len={len(raw)} clean_len={len(cleaned)})"
            )
            response.content = cleaned

    return {"messages": [response], "off_scope_detected": off_scope_detected}


# ─── Routing ─────────────────────────────────────────────────────────────────

def _route_by_intent(state: CAGState) -> str:
    return state.get("intent") or "KNOWLEDGE"



# ─── Graph Assembly ───────────────────────────────────────────────────────────

def _build_agent_graph():
    """Build and compile the minimal conversational CAG StateGraph.

    Collapsed from the old retrieval router to a CAG graph. Routing by the
    regex Tier-1 label set in _pre_processor (no LLM):
        START → pre_processor → MALICIOUS                    → malicious      → END
                              → GREETING/AMBIGUOUS/OFF_SCOPE/TOPIC_LIST
                                                              → generate_node → END  (no retrieval)
                              → KNOWLEDGE/COACHING            → generate_node → END

    Chit-chat / no-lookup intents skip retrieval entirely and go straight to the
    conversational generate node with NO <context> — so a greeting or a vague
    "info dong" never gets an irrelevant chunk dumped on it, and the prompt asks
    a clarifying question instead of guessing. Knowledge turns answer from the
    full active CAG KB pack in generate_node. The canned
    handlers (greeting/ambiguity/off_scope/topic_list/low_relevance) are gone —
    their behavior lives in CONVERSATIONAL_PROMPT.
    """
    builder = StateGraph(CAGState)

    # Nodes
    builder.add_node("pre_processor", _pre_processor)
    builder.add_node("malicious", _handle_malicious)
    builder.add_node("generate_node", _generate_node)

    # Edges
    builder.add_edge(START, "pre_processor")
    builder.add_conditional_edges(
        "pre_processor",
        _route_by_intent,
        {
            "MALICIOUS": "malicious",
            # No-lookup intents → straight to the conversational LLM, no retrieval.
            "GREETING": "generate_node",
            "AMBIGUOUS": "generate_node",
            "OFF_SCOPE": "generate_node",
            "TOPIC_LIST": "generate_node",
            # Jun 2026: SECTION_DRILLDOWN resolves to one specific section
            # from query/history and injects its canonical items via
            # `<section_materials>` — no KB retrieval needed, straight to generate.
            "SECTION_DRILLDOWN": "generate_node",
            # CAG answers from the full active KB pack in generate_node.
            "KNOWLEDGE": "generate_node",
            "COACHING": "generate_node",
        },
    )
    builder.add_edge("malicious", END)
    builder.add_edge("generate_node", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_cag_graph():
    """Return the singleton compiled CAG graph."""
    return _build_agent_graph()


get_rag_graph = get_cag_graph
