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
from app.llm.client import _provider_extra_body, _shared_http_client, get_generate_llm_nostream
from app.llm.prompts import CHIT_CHAT_PROMPT, CONVERSATIONAL_PROMPT, SOCRATIC_PROMPT
from app.knowledge.kb_pack import extract_kb_topics, extract_kb_sections

_settings = get_settings()
_MOODLE_BASE = _settings.moodle_api_url.rstrip("/")


# ─── System Prompts ──────────────────────────────────────────────────────────

# ─── Nodes ───────────────────────────────────────────────────────────────────

# Strips leaked instruction blocks from the LLM response. Some models
# (Gemini Flash Lite especially) occasionally echo the literal contents of
# <retrieved_context> / <user_history> / etc. as part of their output —
# leading to giant <h1>-rendered context dumps in the UI. We catch that
# server-side as a defensive net even after prompt-level guards.
_LEAK_BLOCK_RE = re.compile(
    r"<(retrieved_context|user_history|previous_context|user_preferences|user_context|response_shape|conversation_signals|capabilities|mode|output_contract|role|rules|how_to_talk|length|grounding|disambiguate|no_context|when_to_ask_vs_answer|how_to_ask|during_the_loop|wrap_up|scope|available_topics)>"
    r".*?"
    r"</\1>\s*",
    re.DOTALL | re.IGNORECASE,
)
_LEAK_OPEN_TAG_RE = re.compile(
    r"</?(retrieved_context|user_history|previous_context|user_preferences|user_context|response_shape|conversation_signals|capabilities|mode|output_contract|role|rules|how_to_talk|length|grounding|disambiguate|no_context|when_to_ask_vs_answer|how_to_ask|during_the_loop|wrap_up|scope|available_topics)>",
    re.IGNORECASE,
)
_OFFSCOPE_RE = re.compile(r"\[OFFSCOPE\]", re.IGNORECASE)
_OFFSCOPE_PARTIAL_RE = re.compile(
    r"\[(?:O(?:F(?:F(?:S(?:C(?:O(?:P(?:E\]?)?)?)?)?)?)?)?)?$",
    re.IGNORECASE
)
_COURSE_NUM_RE = re.compile(r"\bCourse\s+\d+(?:\s*:\s*|\s+)?", re.IGNORECASE)
# Citation header from context formatter — "[N] Course: <name> (ID:<id>)".
# Distinctive pattern; never appears in legitimate prose.
_LEAK_CITATION_HEAD_RE = re.compile(
    r"^\s*(?:[>\-*]\s*)?(?:\d+[.)]\s*)?(?:\[\d+\]\s*)?Course:\s*[^\n]*",
    re.MULTILINE | re.IGNORECASE,
)
_META_CONTEXT_LINE_RE = re.compile(
    r"^\s*>?\s*\*\*\[Meta-(?:Context|Konteks)\]\*\*[^\n]*",
    re.MULTILINE | re.IGNORECASE,
)
# ATX markdown headings — "# Foo", "## Bar". Stripping these from chunk text
# before sending to the LLM prevents the giant-font rendering disaster if
# the LLM later echoes chunk content verbatim.
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+", re.MULTILINE)
# Inline source citations like "[[1]]" or "[[1]][[2]]" that the LLM
# sometimes emits from the persona's old example format. Sources are
# rendered separately in the UI — never inline in the user-facing reply.
_INLINE_CITE_RE = re.compile(r"\[\[\d+\]\]")
# Layer-4 leak: lines that look like LITERAL prompt directives (the LLM
# drifts into reciting its conditioning when it has no good answer). They
# start with rule-list words ("Default:", "Go LONGER", "EXCEPTION", "NEVER
# ...", "Open with", "End with", "Talk like", etc.) and are NOT natural
# prose. This catches the case where the LLM echoes block CONTENTS without
# the wrapping tags (Layers 1-3 only catch tagged leaks).
_DIRECTIVE_LINE_RE = re.compile(
    r"^[ \t]*(?:"
    r"Default\s*:\s*SHORT|"
    r"Go LONGER and more structured|"
    r"EXCEPTION\s*[—–-]|"
    r"NEVER\s+(?:echo|pull|use|close|start|emit|start|open)|"
    r"ALWAYS\s+(?:open|close|preserve|use|emit|start)|"
    r"Open with the answer|"
    r"End with substance|"
    r"No hedging|"
    r"Use complete sentences|"
    r"Use bullets for lists|"
    r"Mirror the user's language|"
    r"If <context> is absent|"
    r"When the context (?:IS|is) relevant|"
    r"When the user asks about (?:a SET|the set)|"
    r"CRITICAL\s*[—–-]|"
    r"Talk like a senior|"
    r"Answer factual (?:lookups|questions)|"
    r"Format examples \(Indonesian\)|"
    r"STYLE\s*[—–-]|"
    r"MENTOR MINDSET|"
    r"In COACHING mode|"
    r"FRUSTRATION OVERRIDE|"
    r"COACHING CONDUCT|"
    r"First check RELEVANCE|"
    r"When the context IS relevant|"
    # Leaked <available_topics> instruction + <disambiguate> prose (Flash Lite
    # recites these when the block is output-shaped). Whole-line strip.
    r"(?:The )?[Uu]ser asked what topics|"
    r"List ONLY the topics|"
    r"Runs before answering|"
    r"Check if the turn is UNDERSPECIFIED|"
    r"Ask ONE short clarifying question|"
    r"Irrelevant with the user question|"
    r"\(\d\)\s+A (?:broad|bare|reference|BARE)"
    r")"
    # Eat the rest of the line (often continues with quoted examples / em-dash rules)
    r"[^\n]*",
    re.MULTILINE | re.IGNORECASE,
)

# Meta-conversation recall questions ("udah bahas apa aja", "yang kita bahas",
# "emng itu aja yang kita bahas", "what did we discuss"). The answer is the
# conversation history, NOT the knowledge base — so _pre_processor routes these
# to the no-retrieval path. Without this, the question gets embedded + retrieved,
# random chunks cross the dense floor, and the model describes THOSE as "what we
# discussed" (the fabrication bug). Deliberately biased toward catching meta
# questions (a false positive merely answers from history; a false negative
# brings back the fabrication). A missed phrasing falls through to KNOWLEDGE,
# where the prompt's relevance gate + the wider history window are the backstop.
_META_CONVO_RE = re.compile(
    r"(?:udah|sudah|udh|tadi|barusan|kita|kami)\b[^.?!\n]{0,30}"
    r"(?:bahas|dibahas|ngomong|omongin|diskusi|obrol)"
    r"|(?:yang|apa)\b[^.?!\n]{0,20}(?:di)?(?:bahas|omongin|diskusi)"
    r"|itu aja[^.?!\n]{0,25}(?:bahas|omongin)"
    r"|what (?:did|have|were) we (?:discuss|talk|cover|go over|chat)"
    # Short deictic follow-ups: "which one?" / "the earlier one?" /
    # "how do I do it?" need clarification, not KB retrieval.
    r"|(?:yg|yang)\s+(?:mana|tadi|yg\s+tadi|sebelumnya|sebelum|yg\s+sebelumnya)\b"
    r"|(?:yg|yang)\s+(?:mana|tadi|sebelumnya)\s*[?.!\s]*$"
    r"|(?:gimana|gmana|gmn|how)\s+(?:caranya|carany)(?:\s+(?:ya|yaa|dong|donk|sih))?\s*[?.!\s]*$"
    r"|(?:terus|trus|lanjut|next)\s+(?:gimana|gmn|apa|apanya)\b",
    re.IGNORECASE,
)
_AMARTHA_GLOSSARY = {
    "BM": "Business Manager",
    "BP": "Business Partner",
    "PAR": "Portfolio at Risk",
    "OS": "Outstanding",
    "BTC": "Back to Current",
    "DPD": "Days Past Due",
    "NPL": "Non-Performing Loan",
    "RR": "Repayment Rate",
    "PJ": "Penanggung Jawab"
}

_GLOSSARY_PATTERN = r'\b(' + '|'.join(_AMARTHA_GLOSSARY.keys()) + r')\b'
_GLOSSARY_RE = re.compile(_GLOSSARY_PATTERN, flags=re.IGNORECASE)

def _apply_glossary(text: str) -> str:
    """Replaces Amartha acronyms with their full terms using exact word boundaries."""
    if not text:
        return text
    return _GLOSSARY_RE.sub(lambda m: _AMARTHA_GLOSSARY[m.group(0).upper()], text)


def _normalize_dashes(text: str) -> str:
    # Em-dash reads as AI-generated. After a bold label it's a colon
    # ("**Listen** — x" → "**Listen**: x"); elsewhere a comma. En-dash just
    # becomes a hyphen so numeric/day ranges ("0–7", "Senin–Sabtu") survive.
    text = re.sub(r"\*\*\s*—\s*", "**: ", text)
    text = re.sub(r"\s*—\s*", ", ", text)
    return text.replace("–", "-")


def _sanitize_answer(text: str) -> str:
    """Strip any leaked instruction-block content / tags from an LLM reply.

    Layer 1: balanced XML wrappers (when LLM echoes the whole tag block).
    Layer 2: orphan tags (when only one half leaked).
    Layer 3: bare context dump (when LLM dropped XML wrapper but kept the
             "[N] Course: <name> (ID:<id>)" citation headers and chunk text).
             Heuristic: if any citation header is present, assume everything
             before the LAST one is leak. Take the tail after the last header
             and drop the first paragraph (chunk text) — keep the rest as
             the actual answer. Falls back to a generic retry message if
             nothing usable remains.
    """
    if not text:
        return text
    # Adjust specific closing reference to the general reference
    cleaned = re.sub(
        r"(?i)Untuk\s+detail\s+[^.!?]+silakan\s+cek\s+langsung\s+di\s+modul\s+Business\s+Process(?:[^.!?]*Amarthapedia)?\.?",
        "Kamu bisa pelajari lebih lanjut di Amarthapedia atau bertanya langsung denganku.",
        text
    )
    cleaned = _LEAK_BLOCK_RE.sub("", cleaned)
    cleaned = _LEAK_OPEN_TAG_RE.sub("", cleaned)
    cleaned = _INLINE_CITE_RE.sub("", cleaned)
    cleaned = _META_CONTEXT_LINE_RE.sub("", cleaned)
    # Layer 4: strip prompt-directive echoes (untagged prompt content the
    # LLM recites when it has no good answer — e.g. "Default: SHORT — 2-4
    # sentences..." from the <length> block, leaked without its wrapper).
    cleaned = _DIRECTIVE_LINE_RE.sub("", cleaned)
    cleaned = _OFFSCOPE_RE.sub("", cleaned)
    cleaned = _COURSE_NUM_RE.sub("", cleaned)
    # Collapse 3+ consecutive blank lines that the stripping may leave behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    matches = list(_LEAK_CITATION_HEAD_RE.finditer(cleaned))
    if matches:
        last_end = matches[-1].end()
        tail = cleaned[last_end:].strip()
        # The chunk text right after the last citation header is still leak.
        # Drop the first paragraph; keep whatever follows.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", tail) if p.strip()]
        if len(paragraphs) >= 2:
            cleaned = "\n\n".join(paragraphs[1:])
        elif paragraphs:
            # Only one paragraph — could be the answer OR could be just the
            # chunk text. Heuristic: if it's longer than 80 chars and doesn't
            # start with a bullet/dash, assume it's the answer.
            only = paragraphs[0]
            if len(only) > 80 and not only.startswith(("-", "*", "•")):
                cleaned = only
            else:
                cleaned = "Maaf, ada kendala merangkum jawaban. Coba tanya ulang ya."
        else:
            cleaned = "Maaf, ada kendala merangkum jawaban. Coba tanya ulang ya."

    cleaned = _normalize_dashes(cleaned.lstrip())
    if text.strip() and not cleaned.strip():
        if _OFFSCOPE_RE.search(text):
            return cleaned
        return "Maaf, ada kendala merangkum jawaban. Coba tanya ulang ya."
    return cleaned


class StreamLeakGuard:
    """Stream-time leak detector. Buffers the generated reply only when a leak
    signature (like "[1] Course: X" or "<retrieved_context>") is detected,
    otherwise passes clean tokens through immediately to eliminate startup latency.
    """

    _LEAK_PATTERNS = (
        _LEAK_CITATION_HEAD_RE,
        _LEAK_OPEN_TAG_RE,
        _INLINE_CITE_RE,
        _DIRECTIVE_LINE_RE,
        _OFFSCOPE_PARTIAL_RE,
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._mode = "passthrough"

    def feed(self, token: str) -> str:
        """Push a streamed token. Returns the safe text to emit (may be "")."""
        self._buffer += token
        if any(p.search(self._buffer) for p in self._LEAK_PATTERNS):
            self._mode = "buffered"
            return ""
        
        # If we were buffered but the buffer no longer matches any leak pattern,
        # we can safely flush it and return to passthrough mode.
        self._mode = "passthrough"
        out = self._buffer
        self._buffer = ""
        return out

    def flush(self) -> str:
        """Called at end-of-stream. Returns sanitized trailing text."""
        if self._mode == "buffered":
            cleaned = _sanitize_answer(self._buffer)
            self._buffer = ""
            return cleaned
        out = self._buffer
        self._buffer = ""
        return out

    @property
    def leak_detected(self) -> bool:
        return self._mode == "buffered"


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
        guessing. Cheaper too (no embed + no Qdrant round-trip).
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

    # ── Meta-conversation question → answer from HISTORY, never the KB ───────
    # "kita udah bahas apa aja", "tadi ngomongin apa", "what did we discuss" —
    # the answer is the conversation itself, NOT a knowledge-base lookup. If we
    # retrieved, random chunks crossing the dense floor would be described as
    # "what we discussed" (the fabrication bug). Route to the no-retrieval path
    # so generate_node answers purely from the windowed message history.
    if _META_CONVO_RE.search(user_msg_str):
        logger.info("Pre-processor: meta-conversation question → no retrieval (answer from history)")
        return {
            "intent": "AMBIGUOUS",  # no-retrieval bucket; excluded from cache/eval in chat.py
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
    if _is_section_drilldown_shape(user_msg_str):
        try:
            _sm = await _load_section_map()
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


# ── Section → items map (for "apa aja di <section>" questions) ────────────────
_section_map_cache: dict[str, Any] = {"map": {}, "expires_at": 0.0}
_section_map_lock: asyncio.Lock | None = None


def _get_section_map_lock() -> asyncio.Lock:
    global _section_map_lock
    if _section_map_lock is None:
        _section_map_lock = asyncio.Lock()
    return _section_map_lock


# ════════════════════════════════════════════════════════════════════════════════
# Section Drilldown helpers (Jun 2026)
# ════════════════════════════════════════════════════════════════════════════════
# "topic apa aja"         → TOPIC_LIST       (handled in `_pre_processor`).
# "bisnis proses ada apa" → SECTION_DRILLDOWN: resolve WHICH section from query,
#                             then list ALL items inside that section from
#                             `section_map` (Postgres-cached).
#
# Design constraints:
#   - 100% dynamic: section_map comes from Postgres, no hardcoded alias dict.
#   - Zero new LLM call. Zero new embedding call. Pure regex + dict lookup.
#   - Graceful deictic resolution: "yang kedua" / "topik B" / "yang tadi"
#     resolve from the most recent TOPIC_LIST response in conversation history.
# ════════════════════════════════════════════════════════════════════════════════

_SECTION_NAME_STOPWORDS = frozenset({
    "ada", "apa", "aja", "saja", "di", "dari", "ke", "yang", "itu",
    "ini", "tadi", "tuh", "nih", "kan", "ya", "ga", "gak", "nggak",
    "kok", "sih", "dong", "kak", "bang", "mas", "mbak", "bu", "pak",
    "tolong", "mau", "ingin", "bisa", "dapat", "lihat", "tampil",
    "list", "daftar", "show", "tampilkan", "lihatin",
    "materi", "materinya", "dokumen", "dokumennya", "judul", "judulnya",
    "file", "filenya", "topik", "topiknya", "topic", "section",
    "course", "kursus", "pelajaran", "ajar", "nya", "aja",
})

try:
    import yaml as _yaml_drilldown
    _DRILLDOWN_PATTERNS_PATH = Path(__file__).parent / "intent_patterns.yaml"
    _DRILLDOWN_PATTERNS = _yaml_drilldown.safe_load(
        _DRILLDOWN_PATTERNS_PATH.read_text(encoding="utf-8")
    ) or {}
    _SECTION_DRILLDOWN_PHRASES = tuple(_DRILLDOWN_PATTERNS.get("section_drilldown_phrases", []))
except Exception:
    _SECTION_DRILLDOWN_PHRASES = (
        "ada apa aja", "ada apa", "apa aja", "apa saja", "apa isinya",
        "isinya apa", "di dalamnya apa", "dalamnya apa", "materinya apa",
        "materi apa", "dokumennya apa", "judulnya apa", "list materi",
        "list dokumen", "list judul", "daftar materi",
        "tolong lihat", "lihat materi", "tampilkan materi",
        "tampilkan dokumen",
    )

_ORDINAL_TO_INT = {
    "1": 1, "satu": 1, "pertama": 1, "kesatu": 1, "a": 1,
    "2": 2, "dua": 2, "kedua": 2, "kedu": 2, "b": 2,
    "3": 3, "tiga": 3, "ketiga": 3, "c": 3,
    "4": 4, "empat": 4, "keempat": 4, "d": 4,
    "5": 5, "lima": 5, "kelima": 5, "e": 5,
    "6": 6, "enam": 6, "keenam": 6, "f": 6,
    "7": 7, "tujuh": 7, "ketujuh": 7, "g": 7,
    "8": 8, "delapan": 8, "kedelapan": 8, "h": 8,
}


def _normalize_section_tokens(name: str) -> list[str]:
    """Lowercase + strip punctuation + remove stopwords. Returns significant tokens."""
    import re as _re
    s = _re.sub(r"[^\w\s]", " ", (name or "").lower())
    toks = [t for t in s.split() if t and t not in _SECTION_NAME_STOPWORDS and len(t) > 1]
    return toks


def _levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein edit distance. O(len(a)*len(b)). For short tokens only."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i, bc in enumerate(b, 1):
        cur = [i]
        for j, ac in enumerate(a, 1):
            cur.append(min(
                cur[-1] + 1,        # insertion
                prev[j] + 1,        # deletion
                prev[j-1] + (ac != bc),  # substitution
            ))
        prev = cur
    return prev[-1]


def _fuzzy_token_match(qt: str, st: str) -> bool:
    """Token match with edit-distance fallback for cross-language stem variants.

    Examples: "bisnis" vs "business" (dist=3, ratio≈0.57), "ajar" vs "learning"
    (dist=6, ratio≈0.18 — too far; rejected).
    Threshold: edit distance <= max(2, 30% of max_len).
    """
    if not qt or not st:
        return False
    if qt == st:
        return True
    # Only fuzzy-match on tokens of similar length to avoid spurious matches
    ratio = min(len(qt), len(st)) / max(len(qt), len(st))
    if ratio < 0.55:
        return False
    d = _levenshtein(qt, st)
    max_edits = max(2, int(max(len(qt), len(st)) * 0.30))
    return d <= max_edits


def _score_query_against_section(query: str, section_name: str) -> float:
    """Score how well `query` matches `section_name`. 0.0 = no match, 1.0 = perfect."""
    q_toks = _normalize_section_tokens(query)
    s_toks = _normalize_section_tokens(section_name)
    if not q_toks or not s_toks:
        return 0.0
    overlap = 0
    for qt in q_toks:
        for st in s_toks:
            # 1) Substring containment (handles "anti" in "anti harassment")
            if qt in st or st in qt:
                overlap += 1
                break
            # 2) 4-char prefix match (handles "produk" vs "product", "klien" vs "client")
            if len(qt) >= 4 and len(st) >= 4 and qt[:4] == st[:4]:
                overlap += 1
                break
            # 3) Fuzzy edit-distance match (handles "bisnis" vs "business" — ID↔EN)
            if _fuzzy_token_match(qt, st):
                overlap += 1
                break
    token_score = overlap / max(1, len(s_toks))
    q_full = " ".join(q_toks)
    s_full = " ".join(s_toks)
    if q_full and s_full and (q_full in s_full or s_full in q_full):
        return 1.0
    return min(1.0, token_score)


def _detect_section_from_query(query: str, section_map: dict[str, list[str]]) -> str | None:
    """Match query -> canonical section name via token containment."""
    if not section_map:
        return None
    best_section, best_score = None, 0.0
    for section in section_map.keys():
        score = _score_query_against_section(query, section)
        if score > best_score:
            best_score, best_section = score, section
    return best_section if best_score >= 0.30 else None


def _flatten_message_content(content) -> str:
    """LangChain message content can be str OR list[{type:text}]. Flatten to str."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict):
                txt = blk.get("text") or blk.get("content") or ""
                if txt:
                    parts.append(str(txt))
            elif isinstance(blk, str):
                parts.append(blk)
        return " ".join(parts)
    return str(content) if content else ""


def _has_topic_list_marker(content: str) -> bool:
    """Detect if a previous AI message was a TOPIC_LIST response."""
    low = (content or "").lower()
    markers = (
        "berikut topik", "topik-topik", "daftar topik", "berikut daftar",
        "ini dia topik", "topik yang tersedia", "berikut beberapa topik",
        "kamu bisa belajar", "kamu bisa pelajari", "materi yang tersedia",
        "available topics", "topics available",
    )
    return any(m in low for m in markers)


def _extract_sections_from_topic_list(content: str) -> list[str]:
    """Parse a TOPIC_LIST AI response to recover the section list."""
    import re as _re
    if not content:
        return []
    text = content

    numbered = _re.findall(
        r"(?:^|\n)\s*(?:\d+|[A-Ha-h])[\.\)]\s+([^\n]{2,80})", text
    )
    if numbered:
        cleaned = []
        for s in numbered:
            s = s.strip().rstrip(",;.")
            s = _re.sub(r"^[\*_\-`]+|[\*_\-`]+$", "", s).strip()
            if 2 <= len(s) <= 80:
                cleaned.append(s)
        if cleaned:
            return cleaned

    bullets = _re.findall(r"(?:^|\n)\s*[-*•·]\s+([^\n]{2,80})", text)
    if bullets:
        cleaned = []
        for s in bullets:
            s = s.strip().rstrip(",;.")
            s = _re.sub(r"^[\*_\-`]+|[\*_\-`]+$", "", s).strip()
            if 2 <= len(s) <= 80:
                cleaned.append(s)
        if cleaned:
            return cleaned

    bolds = _re.findall(r"\*\*([^*\n]{2,60})\*\*", text)
    if bolds:
        cleaned = [s.strip().rstrip(",;.") for s in bolds if 2 <= len(s.strip()) <= 60]
        if cleaned:
            return cleaned

    return []


def _resolve_section_ordinal(query: str, sections: list[str]) -> str | None:
    """Resolve 'yang kedua', 'topik B', 'nomor 3' against a section list."""
    import re as _re
    q = (query or "").lower().strip()
    if not q or not sections:
        return None
    m = _re.search(
        r"(?:yang|topi[ck]|no(?:mor)?|pilihan?)\s*"
        r"(?:ke-?|nomor\s*)?\s*"
        r"(satu|dua|tiga|empat|lima|enam|tujuh|delapan|"
        r"pertama|kedua|ketiga|keempat|kelima|keenam|ketujuh|kedelapan|"
        r"[1-8]|[a-h])\b",
        q,
    )
    if m:
        word = m.group(1).lower()
        idx = _ORDINAL_TO_INT.get(word)
        if idx and 1 <= idx <= len(sections):
            return sections[idx - 1]
    if _re.fullmatch(
        r"(?:yang\s+(?:itu|tadi|barusan|sebelumnya|maksud|disebut|dibahas))+|"
        r"(?:yang)|(?:itu)|(?:tadi)|(?:yang\s+aja)|(?:pilih\s+itu)",
        q.strip(),
    ):
        # Only use fallback if there's exactly one section in recent context
        if len(sections) == 1:
            return sections[0]
        # With multiple sections, "yang itu" is ambiguous — return None to avoid guessing
        return None
    return None


def _extract_topic_list_from_history(messages: list) -> list[str]:
    """Walk messages backwards, find last AI TOPIC_LIST response, return section list."""
    if not messages:
        return []
    for m in reversed(messages[:-1]):
        role = getattr(m, "type", None) or getattr(m, "role", "")
        if role and role not in ("ai", "assistant"):
            continue
        content = _flatten_message_content(getattr(m, "content", ""))
        if not _has_topic_list_marker(content):
            continue
        sections = _extract_sections_from_topic_list(content)
        if sections:
            return sections
    return []


def _resolve_drilldown_section(
    query: str,
    messages: list,
    section_map: dict[str, list[str]],
) -> tuple[str | None, str | None]:
    """Resolve drilldown query -> canonical section name.

    A drilldown is only valid when the user is picking from a TOPIC_LIST the
    assistant just showed. A fresh content question ("produk amartha apa aja")
    or a clarifying-question reply ("pencegahan", after the AI asked "pencegahan
    atau pengamanan?") must NOT route here — those go to KNOWLEDGE so the actual
    content gets retrieved. So every path is gated on a recent TOPIC_LIST in
    history (via _extract_topic_list_from_history → _has_topic_list_marker);
    with no such context we return None and let the query fall through to
    retrieval instead of force-routing to a section file-list view.

    Resolution order (all gated on a recent TOPIC_LIST in history):
      1. Ordinal/deictic pick against the history section list.
      2. Token match against the history section list (threshold 0.50).
      3. Direct token match against section_map keys (last resort, same gate).

    Returns (section_name | None, resolution_path | None).
    """
    if not query or not section_map:
        return None, None

    sections_in_history = _extract_topic_list_from_history(messages or [])
    if not sections_in_history:
        # No TOPIC_LIST the user is picking from → not a drilldown.
        return None, None

    ordinal = _resolve_section_ordinal(query, sections_in_history)
    if ordinal:
        for sec in section_map.keys():
            if _score_query_against_section(ordinal, sec) >= 0.50:
                return sec, "history_ordinal"

    best, best_score = None, 0.0
    for sec in sections_in_history:
        score = _score_query_against_section(query, sec)
        if score > best_score:
            best_score, best = score, sec
    if best and best_score >= 0.50:
        for sec in section_map.keys():
            if _score_query_against_section(best, sec) >= 0.50:
                return sec, "history"

    direct = _detect_section_from_query(query, section_map)
    if direct:
        return direct, "query"

    return None, None


def _is_section_drilldown_shape(query: str) -> bool:
    """Quick shape check: does the query LOOK like 'what's inside topic X'?"""
    if not query or len(query) > 150:
        return False
    low = query.lower().strip()
    return any(p in low for p in _SECTION_DRILLDOWN_PHRASES)


async def _load_section_map() -> dict[str, list[str]]:
    """Map each Moodle SECTION → its item list, TTL-cached (10min)."""
    import time as _time

    now = _time.time()
    if now < _section_map_cache["expires_at"] and _section_map_cache["map"]:
        return _section_map_cache["map"]

    lock = _get_section_map_lock()
    async with lock:
        now = _time.time()
        if now < _section_map_cache["expires_at"] and _section_map_cache["map"]:
            return _section_map_cache["map"]

        try:
            cag_kb_text = await _load_active_cag_kb_text()
            section_map = extract_kb_sections(cag_kb_text) if cag_kb_text else {}
        except Exception as exc:
            logger.warning(f"Section-map load failed: {exc}")
            return {}

        _section_map_cache["map"] = section_map
        _section_map_cache["expires_at"] = now + _COURSE_CACHE_TTL_SECONDS
        return section_map


_active_kb_cache: dict[str, Any] = {"hash": "", "content": ""}


def clear_cag_kb_cache() -> None:
    _active_kb_cache.update({"hash": "", "content": ""})
    _course_cache.update({"courses": [], "expires_at": 0.0})
    _section_map_cache.update({"map": {}, "expires_at": 0.0})


async def _load_active_cag_kb_text() -> str:
    from app.database.postgres import AsyncSessionLocal
    from app.knowledge.store import get_active_kb_pack

    try:
        async with AsyncSessionLocal() as session:
            active = await get_active_kb_pack(session, source=_settings.cag_kb_source)
            if active:
                if _active_kb_cache.get("hash") != active.kb_hash:
                    clear_cag_kb_cache()
                    _active_kb_cache.update({
                        "hash": active.kb_hash,
                        "content": active.content,
                    })
                return _active_kb_cache["content"]
    except Exception as exc:
        logger.warning(f"Failed to load active cag kb text: {exc}")
    return _active_kb_cache.get("content", "")


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
    position = str(user_context.get("position") or "").strip().lower()
    
    # 1. HO (Head Office) -> HO
    if location == "HO":
        return "HO"
        
    # 2. FO (Field Office) or general fallback
    if "regional" in position or "rm" in position:
        return "RM"
    elif "area" in position or "am" in position:
        return "AM"
    elif "hub" in position or "hmb" in position:
        return "HMB"
    elif "business manager" in position or "bm" in position:
        return "BM"
    elif "business partner" in position or "bp" in position:
        return "BP"
        
    if location == "FO":
        return "FO"
        
    return "ALL"


def _filter_kb_by_role(kb_text: str, user_role: str) -> str:
    if not kb_text or not user_role:
        return kb_text
    
    role = user_role.upper().strip()
    
    # 1. Find and filter the <doc> blocks first to collect allowed doc IDs
    doc_pattern = re.compile(r'(<doc\s+([^>]*?)>)(.*?)(</doc>)', re.DOTALL)
    roles_attr_pattern = re.compile(r'roles="([^"]*)"')
    id_attr_pattern = re.compile(r'id="([^"]*)"')
    
    filtered_docs = []
    allowed_doc_ids = set()
    for header, attrs, content, footer in doc_pattern.findall(kb_text):
        id_match = id_attr_pattern.search(attrs)
        doc_id = id_match.group(1) if id_match else ""
        
        match = roles_attr_pattern.search(attrs)
        if match:
            doc_roles = [r.strip().upper() for r in match.group(1).split(",")]
            if "ALL" in doc_roles or role in doc_roles:
                filtered_docs.append(f"{header}{content}{footer}")
                if doc_id:
                    allowed_doc_ids.add(doc_id)
        else:
            filtered_docs.append(f"{header}{content}{footer}")
            if doc_id:
                allowed_doc_ids.add(doc_id)
                
    # 2. Extract and filter the <kb_index> block based on allowed_doc_ids
    kb_index_match = re.search(r'<kb_index>(.*?)</kb_index>', kb_text, re.DOTALL)
    kb_index = ""
    if kb_index_match:
        index_content = kb_index_match.group(1)
        entry_pattern = re.compile(
            r'(-\s+\[(DOC-\d+)\](?:(?!-\s+\[DOC-).)*)',
            re.DOTALL
        )
        filtered_entries = []
        for entry, doc_id in entry_pattern.findall(index_content):
            if doc_id in allowed_doc_ids:
                filtered_entries.append(entry.strip())
        
        if filtered_entries:
            kb_index = "<kb_index>\n" + "\n".join(filtered_entries) + "\n</kb_index>"
            
    # 3. Get the version attribute if present to reconstruct the root tag
    version_match = re.search(r'<knowledge_base\s+([^>]*?)>', kb_text)
    root_attrs = version_match.group(1) if version_match else ""
    
    # Reassemble
    out = [f"<knowledge_base {root_attrs}>".strip()]
    if kb_index:
        out.append(kb_index)
    out.extend(filtered_docs)
    out.append("</knowledge_base>")
    
    return "\n".join(out)


async def _build_generate_messages(state: CAGState) -> tuple[list, str]:
    """Build the exact prompt used by generate_node."""
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
        except Exception as exc:
            logger.warning(f"CAG KB pack load failed: {exc}")

    has_kb_context = bool(cag_kb_text)
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
            items = (await _load_section_map()).get(drilldown_sec, [])
        except Exception:
            items = []
        if items:
            section_section = (
                f'\n\n<section_materials section="{drilldown_sec}">\n'
                + "\n".join(f"- {it}" for it in items)
                + "\n</section_materials>"
            )

    ltm_section = ""
    if profile.get("summary"):
        course_names_str = ", ".join(profile.get("course_names", []))
        unanswered = profile.get("unanswered_questions") or []
        history_lines = [
            f"User pernah membahas materi: {course_names_str}",
            f"Konteks sesi sebelumnya: {profile['summary']}",
        ]
        if unanswered:
            history_lines.append(
                "Pertanyaan user yang belum sempat terjawab di sesi lalu: "
                + "; ".join(unanswered)
            )
        ltm_section = "\n\n<user_history>\n" + "\n".join(history_lines) + "\n</user_history>"

    summary_section = f"\n\n<previous_context>\n{summary}\n</previous_context>" if summary else ""

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
            ci = re.sub(r"<[^>]+>", "", prefs["custom_instructions"])
            ci = re.sub(
                r"(?i)(?:ignore|forget|disregard|override)\s+(?:all\s+)?(?:previous|above|prior|system)\s+(?:instructions?|rules?|prompts?)",
                "[filtered]",
                ci,
            )[:500]
            pref_lines.append(f"Instruksi Tambahan: {ci}")
        if pref_lines:
            pref_section = (
                "\n\n<user_preferences>\nSesuaikan jawabanmu dengan profil user berikut:\n"
                + "\n".join(pref_lines)
                + "\n</user_preferences>"
            )

    user_ctx_section = ""
    uctx = state.get("user_context") or {}
    if uctx:
        ctx_lines = []
        if uctx.get("name"):
            ctx_lines.append(f"Nama: {uctx['name']}")
        if uctx.get("dept"):
            ctx_lines.append(f"Departemen: {uctx['dept']}")
        if uctx.get("position"):
            ctx_lines.append(f"Posisi: {uctx['position']}")
        if uctx.get("grade"):
            ctx_lines.append(f"Grade: {uctx['grade']}")
        if uctx.get("location"):
            ctx_lines.append(f"Lokasi: {uctx['location']}")
        if uctx.get("point"):
            ctx_lines.append(f"Point: {uctx['point']}")
        if ctx_lines:
            user_ctx_section = (
                "\n\n<user_context>\nKamu sedang berbicara dengan user berikut. "
                "Sesuaikan jawaban dengan konteksnya, tetapi JANGAN memanggil atau "
                "menyapa nama depannya secara berulang-ulang di setiap awal kalimat atau transisi:\n"
                + "\n".join(ctx_lines)
                + "\n</user_context>"
            )

    dynamic_tail = f"{user_ctx_section}{pref_section}{ltm_section}{summary_section}{topics_section}{section_section}{context_section}".strip()
    is_coaching = intent == "COACHING"
    windowed_messages = _window_generate_history(
        list(state["messages"]),
        max_fresh_turns=_settings.max_fresh_turns,
        max_ai_chars=_settings.max_history_ai_chars,
    )

    if is_coaching:
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
    """Single conversational LLM call — the only answer-generating node.

    One CONVERSATIONAL_PROMPT handles everything: greetings, identity, meta-turns
    ("kok gini", "ga nyambung"), chit-chat, and grounded KB answers. Retrieved
    context is injected ONLY when it's actually relevant; for a greeting / off-scope
    / no-match turn we inject NO context, so the model never gets irrelevant chunks
    forced into a casual reply — it just answers conversationally or says it doesn't
    have that info. Memory (STM summary, LTM profile, user prefs) is always injected
    when present. Conciseness + the detail/teach escalation live in the prompt.
    """
    summary = state.get("conversation_summary") or ""
    profile = state.get("user_profile") or {}
    intent = state.get("intent") or "KNOWLEDGE"

    cag_kb_text = ""
    user_context = state.get("user_context") or {}
    if intent in ("KNOWLEDGE", "COACHING"):
        try:
            full_kb = await _load_active_cag_kb_text()
            resolved_role = resolve_user_role(user_context)
            cag_kb_text = _filter_kb_by_role(full_kb, resolved_role)
            logger.info(f"_generate_node: Filtered KB for role={resolved_role}: {len(cag_kb_text)} chars (out of {len(full_kb)})")
        except Exception as exc:
            logger.warning(f"CAG KB pack load failed: {exc}")

    has_kb_context = bool(cag_kb_text)
    context_section = ""
    if intent in ("KNOWLEDGE", "COACHING") and not cag_kb_text:
        context_section = (
            "\n\n<knowledge_base_missing>\n"
            "No active CAG knowledge base pack is available. Ask an admin to run Moodle KB sync first."
            "\n</knowledge_base_missing>"
        )

    # TOPIC_LIST: the user asked what materials/topics exist ("ada materi apa aja").
    topics_section = ""
    if intent == "TOPIC_LIST":
        try:
            course_names = await _load_course_names()
        except Exception:
            course_names = []
        if course_names:
            topics_section = (
                "\n\n<available_topics>\n"
                + "\n".join(f"- {c}" for c in course_names)
                + "\n</available_topics>"
            )
        else:
            topics_section = (
                "\n\n<available_topics>\n(could not load topic list right now)\n"
                "</available_topics>"
            )

    # Section drill-down.
    section_section = ""
    drilldown_sec = state.get("drilldown_section")
    if drilldown_sec:
        try:
            items = (await _load_section_map()).get(drilldown_sec, [])
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
                f"SECTION_DRILLDOWN resolved section={drilldown_sec!r} but "
                f"section_map has no items"
            )

    # Long-term memory (LTM profile)
    ltm_section = ""
    if profile.get("summary"):
        course_names_str = ", ".join(profile.get("course_names", []))
        unanswered = profile.get("unanswered_questions") or []
        history_lines = [
            f"User pernah membahas materi: {course_names_str}",
            f"Konteks sesi sebelumnya: {profile['summary']}",
        ]
        if unanswered:
            history_lines.append(
                "Pertanyaan user yang belum sempat terjawab di sesi lalu: "
                + "; ".join(unanswered)
            )
        ltm_section = "\n\n<user_history>\n" + "\n".join(history_lines) + "\n</user_history>"

    # Short-term rolling summary
    summary_section = ""
    if summary:
        summary_section = f"\n\n<previous_context>\n{summary}\n</previous_context>"

    # Persistent user preferences
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
            ci = prefs["custom_instructions"]
            import re as _re2
            ci = _re2.sub(r"<[^>]+>", "", ci)
            ci = _re2.sub(
                r"(?i)(?:ignore|forget|disregard|override)\s+(?:all\s+)?(?:previous|above|prior|system)\s+(?:instructions?|rules?|prompts?)",
                "[filtered]",
                ci,
            )
            ci = ci[:500]
            pref_lines.append(f"Instruksi Tambahan: {ci}")
        if pref_lines:
            pref_section = (
                "\n\n<user_preferences>\nSesuaikan jawabanmu dengan profil user berikut:\n"
                + "\n".join(pref_lines)
                + "\n</user_preferences>"
            )

    # Live Moodle profile of the person asking (firstname + custom fields).
    user_ctx_section = ""
    uctx = state.get("user_context") or {}
    if uctx:
        ctx_lines = []
        if uctx.get("name"):
            ctx_lines.append(f"Nama: {uctx['name']}")
        if uctx.get("dept"):
            ctx_lines.append(f"Departemen: {uctx['dept']}")
        if uctx.get("position"):
            ctx_lines.append(f"Posisi: {uctx['position']}")
        if uctx.get("grade"):
            ctx_lines.append(f"Grade: {uctx['grade']}")
        if uctx.get("location"):
            ctx_lines.append(f"Lokasi: {uctx['location']}")
        if uctx.get("point"):
            ctx_lines.append(f"Point: {uctx['point']}")
        if ctx_lines:
            user_ctx_section = (
                "\n\n<user_context>\nKamu sedang berbicara dengan user berikut. "
                "Sesuaikan jawaban dengan konteksnya, tetapi JANGAN memanggil atau "
                "menyapa nama depannya secara berulang-ulang di setiap awal kalimat atau transisi:\n"
                + "\n".join(ctx_lines)
                + "\n</user_context>"
            )

    dynamic_tail = f"{user_ctx_section}{pref_section}{ltm_section}{summary_section}{topics_section}{section_section}{context_section}".strip()

    is_coaching = intent == "COACHING"
    _is_grounded = (has_kb_context or bool(topics_section) or bool(section_section)) and not is_coaching
    windowed_messages = _window_generate_history(
        list(state["messages"]),
        max_fresh_turns=_settings.max_fresh_turns,
        max_ai_chars=_settings.max_history_ai_chars,
    )

    if is_coaching:
        system_prompt_text = SOCRATIC_PROMPT
    elif intent in ("GREETING", "AMBIGUOUS", "OFF_SCOPE"):
        system_prompt_text = CHIT_CHAT_PROMPT
    else:
        # KNOWLEDGE, TOPIC_LIST, SECTION_DRILLDOWN
        system_prompt_text = CONVERSATIONAL_PROMPT

    openrouter_session_id = _openrouter_prompt_session_id(system_prompt_text, cag_kb_text)
    llm = _with_openrouter_session(
        get_generate_llm_nostream(),
        openrouter_session_id,
    )

    system_msg = SystemMessage(content=system_prompt_text)
    msgs: list = [system_msg]
    if cag_kb_text:
        msgs.append(SystemMessage(content=cag_kb_text))
    if dynamic_tail:
        msgs.append(HumanMessage(content=dynamic_tail))
    msgs += windowed_messages

    _t0 = time.monotonic()
    try:
        response = await llm.ainvoke(msgs, config=config)
    except Exception as gen_exc:
        logger.warning(
            f"generate_node ainvoke failed ({type(gen_exc).__name__}): {gen_exc}"
        )
        raise
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
