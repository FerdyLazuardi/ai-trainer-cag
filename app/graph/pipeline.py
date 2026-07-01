"""
Optimized Agentic RAG pipeline - Retrieve-then-Generate pattern.

Architecture change vs prior ReAct pattern:
  BEFORE: classifier → agent(LLM decides tool) → ToolNode → agent(LLM answers)  = 3 LLM calls for KNOWLEDGE
  AFTER:  classifier → rag_node(pure retrieval) → generate_node(LLM answers)    = 2 LLM calls for KNOWLEDGE

Savings: ~700 tokens per KNOWLEDGE query (the first "decide to call tool" agent call is eliminated).
"""
import asyncio
from functools import lru_cache
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
from app.graph.state import RAGState
from app.llm.client import get_chat_llm, get_generate_llm, get_generate_llm_nostream
from app.llm.prompts import PERSONA, OUTPUT_CONTRACT, CONVERSATIONAL_PROMPT, CHIT_CHAT_PROMPT, SOCRATIC_PROMPT
from app.utils.token_counter import truncate_to_tokens

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

# Follow-up condense-question rewrite (LLM-based; replaces the old anaphoric-
# regex prepend). A SHORT turn after prior history ("boleh", "iya yang itu",
# "yang kedua tadi", "dampaknya apa") carries no standalone meaning: embedding
# it directly drifts retrieval onto the wrong topic, and the model then
# fabricates an answer from training data (the multi-turn hallucination). We
# condense it into a self-contained query using the last few turns — this is
# phrasing/typo/language-agnostic (no word-list to maintain). Scoped to short
# turns only, so the rewrite call is paid only where coreference resolution is
# actually needed; a long self-contained question skips it. A short query that
# already names its topic is returned unchanged by the prompt, and any
# failure/timeout degrades to the raw message.
_REWRITE_MAX_CHARS = 1000
_REWRITE_TIMEOUT_S = 8.0
_FOLLOWUP_HISTORY_TURNS = 6  # last N messages fed as context (≈3 turns)

_AMARTHA_GLOSSARY = {
    "BM": "Business Manager",
    "BP": "Business Partner",
    "PAR": "Portfolio at Risk",
    "OS": "Outstanding",
    "TR": "Tanggung Renteng",
    "BTC": "Back to Current",
    "DPD": "Days Past Due",
    "NPL": "Non-Performing Loan",
    "RR": "Repayment Rate",
    "PJ": "Penanggung Jawab"
}

import re
_GLOSSARY_PATTERN = r'\b(' + '|'.join(_AMARTHA_GLOSSARY.keys()) + r')\b'
_GLOSSARY_RE = re.compile(_GLOSSARY_PATTERN, flags=re.IGNORECASE)

def _apply_glossary(text: str) -> str:
    """Replaces Amartha acronyms with their full terms using exact word boundaries."""
    if not text:
        return text
    return _GLOSSARY_RE.sub(lambda m: _AMARTHA_GLOSSARY[m.group(0).upper()], text)

from app.llm.prompts import REWRITE_PROMPT

_REWRITE_SPLIT_RE = re.compile(r"\s*(?:\r?\n|\s+\|\s+|[;?]+|,\s*)\s*")
_REWRITE_NOISE_RE = re.compile(
    r"\b(?:aku|saya|gw|gue|gua|kan|ini|itu|lagi|parah|harusn?|gapian|"
    r"ngapain|ngpian|apa|apaan|gimana|bagaimana|gmn|gmna|biar|agar|angka|"
    r"juga|tolong|dong|ya|deh|sih|kayak|kek|gitu)\b",
    flags=re.IGNORECASE,
)
_POINT_LOCATION_RE = re.compile(
    r"\bpoint\s+(?!(?:fraud|business|manager|operasional|amartha|mitra|"
    r"pembiayaan|par|npl|dpd|surprise|visit|portofolio|portfolio|client|"
    r"protection)\b)[\w-]+",
    flags=re.IGNORECASE,
)
_CONTEXT_ONLY_RE = re.compile(
    r"^(?:business manager|point|business manager point|point business manager)$",
    flags=re.IGNORECASE,
)
def _clean_rewrite_line(text: str) -> str:
    text = _apply_glossary(text)
    text = _POINT_LOCATION_RE.sub("point", text)
    text = _REWRITE_NOISE_RE.sub(" ", text)
    text = re.sub(r"[^\w\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _rewrite_context_terms(lines: list[str]) -> list[str]:
    joined = "\n".join(lines).lower()
    terms = []
    if "business manager" in joined:
        terms.append("Business Manager")
    if "client protection" in joined:
        terms.append("Client Protection")
    if re.search(r"\bpoint\b", joined):
        terms.append("point")
    return terms


def _append_rewrite_context(line: str, contexts: list[str]) -> str:
    low = line.lower()
    for ctx in contexts:
        if ctx.lower() not in low:
            line = f"{line} {ctx}"
            low = line.lower()
    return line


def _postprocess_rewrite_query(out: str, user_msg: str) -> str:
    cleaned: list[str] = []
    for line in _REWRITE_SPLIT_RE.split(out):
        q = _clean_rewrite_line(line)
        if q:
            cleaned.append(q)
    if not cleaned:
        return ""

    contexts = _rewrite_context_terms(cleaned)
    has_non_context = any(not _CONTEXT_ONLY_RE.fullmatch(q) for q in cleaned)
    out_lines = []
    for q in cleaned:
        if has_non_context and _CONTEXT_ONLY_RE.fullmatch(q):
            continue
        q = _append_rewrite_context(q, contexts)
        out_lines.append(q)
    return "\n".join(dict.fromkeys(out_lines))

async def _rewrite_search_query(messages: list, user_msg: str) -> str:
    """Rewrite conversational or follow-up queries into standalone keyword-rich search queries via the
    cheap LLM. Returns user_msg unchanged on bad output or failure."""
    # Keep last 2 exchanges max — just enough to resolve "1" back to an option.
    # Full 10-turn history floods the window with product lists and the rewrite
    # LLM loses track of the user's intent.
    history = messages[-(_FOLLOWUP_HISTORY_TURNS + 1):-1]
    if len(history) > 4:
        history = history[-4:]
    lines = []
    for m in reversed(history):
        role = "User" if isinstance(m, HumanMessage) else "Ava"
        content = m.content if isinstance(m.content, str) else str(m.content)
        lines.append(f"{role}: {content[:400]}")
    
    history_text = "\nConversation (newest first):\n" + "\n".join(lines) + "\n" if lines else ""
    prompt = (
        f"{REWRITE_PROMPT}\n{history_text}\nUser message: {user_msg}\n\nSearch queries:"
    )
    try:
        from app.llm.client import get_preprocessor_llm
        resp = await asyncio.wait_for(
            get_preprocessor_llm().ainvoke([HumanMessage(content=prompt)]),
            timeout=_REWRITE_TIMEOUT_S,
        )
        out = (resp.content if isinstance(resp.content, str) else str(resp.content)).strip()
        # Guard: empty or rambling output (a leaked CoT / refusal) → raw msg.
        if out and len(out) <= 300:
            return _postprocess_rewrite_query(out, user_msg) or user_msg
    except Exception as exc:
        logger.debug(f"Query rewrite skipped (degrade to raw): {exc}")
    return _postprocess_rewrite_query(user_msg, user_msg) or user_msg


def _strip_md_headings_for_context(text: str) -> str:
    """Strip ATX markdown headings (#, ##, ###) from chunk text.

    Reason: chunks come from Markdown KB documents, so they contain "# Title"
    lines. If the LLM echoes a chunk verbatim, the frontend renders those
    headings as <h1>/<h2>, producing fonts 2-4x normal body. Stripping the
    leading "#" makes the text plain — even on echo, the UI stays sane.
    Bold/italic/lists are preserved (only headings are visually catastrophic).
    """
    return _MD_HEADING_RE.sub("", text)


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
    cleaned = _LEAK_BLOCK_RE.sub("", text)
    cleaned = _LEAK_OPEN_TAG_RE.sub("", cleaned)
    cleaned = _INLINE_CITE_RE.sub("", cleaned)
    cleaned = _META_CONTEXT_LINE_RE.sub("", cleaned)
    # Layer 4: strip prompt-directive echoes (untagged prompt content the
    # LLM recites when it has no good answer — e.g. "Default: SHORT — 2-4
    # sentences..." from the <length> block, leaked without its wrapper).
    cleaned = _DIRECTIVE_LINE_RE.sub("", cleaned)
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
        return "Maaf, ada kendala merangkum jawaban. Coba tanya ulang ya."
    return cleaned


class StreamLeakGuard:
    """Stream-time leak detector. Buffers the first ~80 chars of a generated
    reply so raw-context leaks (chunk citations like "[1] Course: X (ID:N)" or
    bare <retrieved_context> tags) never reach the user mid-stream. The end-of-
    stream `_sanitize_answer` already protects cache/history/eval — this class
    is the user-facing complement so the leak isn't visible in real time.

    Modes:
      - preamble: buffer until PREAMBLE_LIMIT chars accumulated, then decide.
      - passthrough: preamble was clean — yield raw tokens straight through.
      - buffered: leak detected — keep accumulating, sanitize at flush().
    """

    PREAMBLE_LIMIT = 80
    _LEAK_PATTERNS = (
        _LEAK_CITATION_HEAD_RE,
        _LEAK_OPEN_TAG_RE,
        _INLINE_CITE_RE,
        _DIRECTIVE_LINE_RE,
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._mode = "preamble"

    def feed(self, token: str) -> str:
        """Push a streamed token. Returns the safe text to emit (may be "")."""
        if self._mode == "passthrough":
            return token
        self._buffer += token
        if self._mode == "buffered":
            return ""
        if len(self._buffer) < self.PREAMBLE_LIMIT:
            return ""
        if any(p.search(self._buffer) for p in self._LEAK_PATTERNS):
            self._mode = "buffered"
            logger.warning(
                "StreamLeakGuard: leak signature detected in preamble — "
                "switching to buffered/sanitize mode"
            )
            return ""
        out = self._buffer
        self._buffer = ""
        self._mode = "passthrough"
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


# Strong refs for fire-and-forget background metric tasks scheduled from sync
# routers (otherwise the event loop may GC the task mid-flight).
_BACKGROUND_TASKS: set[asyncio.Task] = set()


async def _incr_sparse_only_passthrough_metric() -> None:
    """Fire-and-forget counter for KNOWLEDGE queries that clear the relevance
    gate on the SPARSE (BM25 lexical) signal ALONE (H9).

    Dense cosine was below its floor but a raw term match rescued the query.
    Bucketed by UTC date (7-day retention); mirrors _incr_parse_failure_metric.
    This is the early-warning signal for the colloquial-ID false-pass risk:
    terse / slangy Indonesian queries that dense embeddings rank off-topic but
    happen to share a BM25 term with the KB. When `rag:metrics:gate_sparse_only_pass`
    climbs, pull those queries and grow the eval sample to confirm the OR-gate
    isn't waving through hallucination-prone misses. Never raises.
    """
    try:
        from datetime import datetime, timezone

        from app.database.redis_client import get_redis_client

        redis = get_redis_client()
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        key = f"rag:metrics:gate_sparse_only_pass:{day}"
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 7 * 24 * 3600)
        await pipe.execute()
    except Exception:
        pass


def _emit_sparse_only_passthrough() -> None:
    """Schedule the sparse-only-pass counter from a sync router without blocking.

    `_route_after_rag` is a sync LangGraph router, so we can't await. Schedule
    the Redis incr on the running loop and hold a strong ref so it isn't GC'd
    mid-flight. If no loop is running (a unit test calling the router directly),
    swallow the RuntimeError — the f-string log line above is the fallback
    signal, and a metrics write must never break routing.
    """
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(_incr_sparse_only_passthrough_metric())
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)
    except RuntimeError:
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
        if not cached or not prompt:
            rm = getattr(response, "response_metadata", None) or {}
            tu = rm.get("token_usage") or {}
            prompt = prompt or (tu.get("prompt_tokens", 0) or 0)
            ptd = tu.get("prompt_tokens_details") or {}
            cached = cached or (ptd.get("cached_tokens", 0) or 0)
        rm = getattr(response, "response_metadata", None) or {}
        completion = int((um or {}).get("output_tokens", 0) or 0)
        if not completion:
            tu = rm.get("token_usage") or {}
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
        if turn_id:
            try:
                await _persist_or_cache_metrics(
                    turn_id=turn_id,
                    prompt=int(prompt),
                    cached=int(cached),
                    completion=completion,
                    provider=provider,
                    duration_s=duration_s,
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


async def _pre_processor(state: RAGState, config: RunnableConfig):
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

    # ── Semantic gate (Tier-0) — catch novel chit-chat the regex missed ─────
    # Runs ONLY when the regex returned None (i.e. not already a chit-chat).
    # When enabled (settings.intent_semantic_gate_enabled, default ON after
    # 2026-06-17 calibration), the gate catches novel greeting/ambiguous
    # phrasings the regex hasn't seen — and routes them to the no-retrieval
    # bucket instead of KNOWLEDGE. Saves a Qdrant round-trip + a full
    # CONVERSATIONAL_PROMPT pass on every catch.
    #
    # Cost: ~one fresh embed (cached on repeat) on the long-tail of queries
    # the regex already filters out. The hot path (regex hits) is unaffected.
    #
    # The full GateScore (best/second cosine + margin) is attached to
    # state["gate_score"] on EVERY outcome — HIT or MISS — so chat.py
    # can persist the trace to agent_logs for the drift monitor regardless
    # of which way the decision went. The historical SKIP path (regex
    # already won, run gate solely for dashboard agreement) was REMOVED on
    # 2026-06-17: pure overhead (2× embed on every regex hit) with no
    # quality gain. The agreement dashboard now only updates from the
    # MISS path; if regex/gate disagreement drifts, sample N% of regex
    # wins through a BackgroundTasks call (TODO: re-add sampled async
    # agreement-check once Prometheus metrics land, so we can verify
    # the drift signal is meaningful before paying the embed cost).
    gate_score_out = None
    if _settings.intent_semantic_gate_enabled and rule_intent is not None:
        from app.graph.intent_classifier import GateScore

        gate_score_out = GateScore(
            decision="SKIP",
            committed=None,
            best_intent=None,
            best_cosine=0.0,
            second_intent=None,
            second_cosine=0.0,
            margin=0.0,
        )
    if _settings.intent_semantic_gate_enabled and rule_intent is None:
        try:
            from app.graph.intent_classifier import classify_semantic_with_scores
            gate_score_out = await classify_semantic_with_scores(
                user_msg_str,
                query_embedding=state.get("query_embedding"),
            )
            if gate_score_out.committed is not None:
                logger.info(
                    f"Pre-processor: semantic gate → {gate_score_out.committed} "
                    f"(regex miss, embedding gate caught it)"
                )
                # SECTION_DRILLDOWN refinement (Jun 2026): also apply to gate-derived
                # TOPIC_LIST. The gate catches novel chit-chat the regex missed —
                # but "bisnis proses ada apa aja" doesn't look like greeting/ambiguous
                # to it, so the gate routes it as TOPIC_LIST. We still want to refine
                # that to SECTION_DRILLDOWN if a specific section can be resolved.
                if gate_score_out.committed == "TOPIC_LIST" and _is_section_drilldown_shape(user_msg_str):
                    try:
                        _sm = await _load_section_map()
                    except Exception:
                        _sm = {}
                    _resolved, _respath = _resolve_drilldown_section(user_msg_str, messages, _sm)
                    if _resolved:
                        logger.info(
                            f"Pre-processor: gate TOPIC_LIST refined -> SECTION_DRILLDOWN "
                            f"(section={_resolved!r}, via {_respath!r})"
                        )
                        state["drilldown_section"] = _resolved
                        state["drilldown_resolution"] = _respath
                        return {
                            "intent": "SECTION_DRILLDOWN",
                            "rewritten_query": user_msg_str,
                            "retrieval_query": user_msg_str,
                            "intent_scores": {"needs_lookup": 0.0, "needs_reasoning": 0.0, "needs_empathy": 0.0, "needs_safety_escalation": 0.0, "learning_context": 0.0},
                            "gate_score": gate_score_out,
                            "drilldown_section": _resolved,
                            "drilldown_resolution": _respath,
                        }
                return {
                    "intent": gate_score_out.committed,
                    "rewritten_query": user_msg_str,
                    "retrieval_query": user_msg_str,
                    "intent_scores": {"needs_lookup": 0.0, "needs_reasoning": 0.0, "needs_empathy": 0.0, "needs_safety_escalation": 0.0, "learning_context": 0.0},
                    "gate_score": gate_score_out,
                }
        except Exception as exc:
            logger.warning(f"semantic gate skipped (intent classification degraded): {exc}")
            gate_score_out = None

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
                "gate_score": gate_score_out,
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
            "gate_score": gate_score_out,
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
    elif len(messages) > 1 and len(_msg_stripped) <= _REWRITE_MAX_CHARS:
        # Fallback rewrite: resolve coreference (e.g. "klo prinsipnya" → "prinsip client protection")
        # using conversation history. Only runs when chat.py didn't pre-compute.
        _msg_for_rewrite = _apply_glossary(_msg_stripped)
        rewritten = await _rewrite_search_query(messages, _msg_for_rewrite)
        if rewritten and rewritten != _msg_stripped:
            if len(rewritten) > max(len(_msg_stripped) * 4, 100):
                logger.warning(f"Query rewrite suspiciously long, using original: {rewritten[:80]!r}")
                rewritten = _msg_stripped
            else:
                logger.info(f"Query rewritten: {_msg_stripped!r} → {rewritten[:60]!r}")
            retrieval_query = rewritten

    # ── Semantic TOPIC_LIST fallback (regex missed) ─────────────────────────
    # The regex Tier-1 can't catch every typo/paraphrase of "what can I learn?"
    # ("materi yang kamu bisa pelajari", "bisa belajar apa hari ini"). These fell
    # through to KNOWLEDGE → retrieved random chunks → the model listed granular
    # sub-topics instead of the real section list. The embedding centroid
    # recognises them cleanly (>=0.70, best=TOPIC_LIST), while content questions
    # like "produk amartha apa aja" score ~0.58/best=KNOWLEDGE and correctly
    # DON'T match.
    #
    # COST GUARD (pre-gate): the semantic check needs a fresh embed (~390ms), so
    # we must NOT run it on every KNOWLEDGE turn. A topic-list question ALWAYS
    # mentions a learning/topic hint word — so we first do a cheap substring
    # gate. Pure content questions ("berapa bunga modal", "apa itu client
    # protection") have no hint word → skip the embed entirely → zero added
    # latency. Only the rare hint-bearing phrasing the regex missed pays the
    # embed (and _embed_one caches it). Length-bounded so long real questions
    # that happen to contain "materi" ("jelasin materi CP dong panjang lebar...")
    # don't trigger the embed either.
    _low_msg = user_msg_str.lower()
    _TL_HINTS = ("belajar", "pelajar", "dipelajari", "materi", "topik", "tema",
                 "konten", "course", "kursus", "pelatihan", "modul", "pembelajaran")
    _tl_pregate = len(_low_msg) <= 60 and any(h in _low_msg for h in _TL_HINTS)
    if _tl_pregate and not state.get("coaching_mode"):
        try:
            from app.graph.intent_classifier import is_topic_list_semantic
            # Fresh embed inside the check (reusing the route embedding gave
            # inconsistent borderline scores). Cached by _embed_one, and gated
            # above so it only runs on hint-bearing, regex-missed phrasings.
            # We pass the pre-computed query_embedding to reuse it and save latency.
            if await is_topic_list_semantic(user_msg_str, query_embedding=state.get("query_embedding")):
                logger.info("Pre-processor: semantic TOPIC_LIST fallback → no retrieval")
                return {
                    "intent": "TOPIC_LIST",
                    "rewritten_query": user_msg_str,
                    "retrieval_query": user_msg_str,
                    "intent_scores": {"needs_lookup": 0.0, "needs_reasoning": 0.0, "needs_empathy": 0.0, "needs_safety_escalation": 0.0, "learning_context": 0.0},
                    "gate_score": gate_score_out,
                }
        except Exception as exc:
            logger.debug(f"semantic TOPIC_LIST fallback skipped: {exc}")

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
        "gate_score": gate_score_out,
    }


async def _handle_malicious(state: RAGState, config: RunnableConfig):
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


async def _handle_low_relevance(state: RAGState, config: RunnableConfig):
    """Deterministic NOT-FOUND refusal for a KNOWLEDGE turn whose retrieval fell
    below the dense floor — NO LLM call.

    Why deterministic instead of letting generate_node's <no_context> prompt
    handle it: a weak generator (DeepSeek/Gemini Flash class) reliably IGNORES
    the "don't invent acronym expansions" rule for terms with a strong training
    prior (e.g. "apa itu BMDP" → fabricated "Buku Monitoring..."), even with the
    context correctly withheld. The floor stops the wrong chunks from entering;
    this stops the model from inventing facts when nothing valid was retrieved.
    Gentle wording so a reaction/meta turn that misroutes here ("halu banget")
    still reads as "I didn't catch that" rather than a robotic error.
    """
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=(
        "Hmm, aku belum nemu info soal itu di materi Amarthapedia yang aku punya. "
        "Coba perjelas maksudnya atau pakai kata kunci lain ya 🙏"
    ))]}


def _filter_seen_chunks(chunks: list[dict], seen_chunk_ids: set[str]) -> list[dict]:
    if not seen_chunk_ids:
        return chunks
    fresh = [
        c for c in chunks
        if not c.get("chunk_id") or str(c.get("chunk_id")) not in seen_chunk_ids
    ]
    return fresh or chunks[:1]


async def _rag_node(state: RAGState, config: RunnableConfig):
    """
    Pure retrieval node — calls hybrid_search (dense + sparse BM25 fusion)
    without any LLM call. Stores formatted context chunks into
    state['retrieved_context']. This replaces the first ReAct 'agent' call that
    previously just decided to use a tool.
    """
    from app.retrieval.hybrid_retriever import hybrid_search

    # Use the guarded retrieval query if available; it preserves critical
    # anchors that a standalone rewrite may legitimately hide from the UI.
    raw_query_to_search = (
        state.get("retrieval_query")
        or state.get("rewritten_query")
        or state["messages"][-1].content
    )
    query_to_search = raw_query_to_search if isinstance(raw_query_to_search, str) else str(raw_query_to_search)

    # H5 — reuse the embedding the chat route already computed, but ONLY if it
    # was computed from the exact text we're about to search. The route embeds
    # `resolved_query` (for cache/LTM); after rewrite/safety-anchoring the
    # retrieval query often differs, in which case reusing would search the
    # wrong vector — so we fall back to embedding inside hybrid_search.
    precomputed = state.get("query_embedding")
    precomputed_text = state.get("query_embedding_text")
    reuse_embedding = (
        precomputed
        if precomputed and precomputed_text == query_to_search
        else None
    )

    try:
        # Multi-query retrieval: when the pre-processor produced a compound
        # rewrite (e.g. "bayar mitra | skill negosiasi | fraud" — 3 sub-topics),
        # search each sub-query in parallel and merge. The old path joined them
        # into one string and searched queries[0] only, so a 3-topic question
        # retrieved chunks for ONE topic. _pre_processor passes the list via
        # state["rewritten_queries"]; retrieval_query carries queries[0].
        raw_sub_queries = state.get("rewritten_queries")
        sub_queries = (
            [q for q in raw_sub_queries if isinstance(q, str)]
            if isinstance(raw_sub_queries, list)
            else [query_to_search]
        )
        if len(sub_queries) > 1:
            from app.retrieval.hybrid_retriever import hybrid_search_multi
            # reuse the route's embedding only for the primary sub-query (it
            # was embedded from resolved_query, which == queries[0] when no
            # safety-anchor differs — other sub-queries embed fresh in-cache).
            multi_embs = [reuse_embedding] + [None] * (len(sub_queries) - 1)
            result = await hybrid_search_multi(
                sub_queries,
                top_k=_settings.final_top_k,
                final_k_multiplier=_settings.compound_final_top_k_multiplier,
                query_embeddings=multi_embs,
            )
            _mq_tag = f" (multi: {len(sub_queries)} sub-queries)"
        else:
            result = await hybrid_search(
                query=query_to_search,  # type: ignore[arg-type]  # langchain message.content is str at runtime
                top_k=_settings.final_top_k,
                query_embedding=reuse_embedding,
            )
            _mq_tag = ""
        docs = result.chunks

        chunks = []
        for d in docs:
            m = d.metadata or {}
            chunks.append({
                "chunk_id": d.chunk_id,
                "text": d.text,
                "course_id": m.get("course_id", ""),
                "course_name": m.get("course_name", d.title),
                "section_name": m.get("section_name", ""),
                "score": round(d.score, 4) if d.score is not None else 0.0,
                "hybrid_score": round(d.hybrid_score, 4) if d.hybrid_score is not None else 0.0,
                "dense_score": round(d.dense_score, 4) if d.dense_score is not None else 0.0,
                "sparse_score": round(d.sparse_score, 4) if d.sparse_score is not None else 0.0,
                "source": d.source or m.get("source", "Unknown"),
                "document_id": d.document_id or m.get("document_id", "Unknown"),
            })
        chunks = _filter_seen_chunks(chunks, set(state.get("seen_chunk_ids") or []))

        logger.info(f"RAG node retrieved {len(chunks)} chunks{_mq_tag} for query: {query_to_search[:60]}")
        # C4: surface pool-level signals (max over full fetch_k pool, pre-slice)
        # for the NOT-FOUND gate; C5: dense_retrieval_ok=False when degraded to
        # sparse-only so the gate doesn't read the missing dense signal as a miss.
        return {
            "retrieved_context": chunks,
            "pool_max_dense": result.pool_max_dense,
            "pool_max_sparse": result.pool_max_sparse,
            "dense_retrieval_ok": result.dense_available,
        }

    except Exception as e:
        logger.error(f"RAG node retrieval failed: {e}")
        # Raise instead of swallowing to allow FastAPI's error handler to return 500
        raise RuntimeError(f"Database error during context retrieval: {e}") from e


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
    """Distinct TOPIC labels from the documents table, TTL-cached (10min).

    The ground-truth list of topics Ava actually has, injected into the generate
    prompt on a TOPIC_LIST turn so "ada materi apa aja" is answered from real
    data instead of fabricated. Single-flight via asyncio.Lock so concurrent
    requests don't stampede Postgres on cache expiry.

    Topic label = the Moodle SECTION name when present, else the per-file
    `course_name` (backward-compat for docs ingested before section_name
    existed). COALESCE(NULLIF(section_name,''), course_name) means several files
    in one Moodle section (e.g. modal.md + celengan.md under "Product Amartha")
    collapse to ONE topic entry instead of spamming one row per file — the
    manual Moodle section structure becomes the topic taxonomy.
    """
    import time as _time

    now = _time.time()
    if now < _course_cache["expires_at"] and _course_cache["courses"]:
        return _course_cache["courses"]

    lock = _get_course_cache_lock()
    async with lock:
        now = _time.time()
        if now < _course_cache["expires_at"] and _course_cache["courses"]:
            return _course_cache["courses"]

        from sqlalchemy import select, distinct
        from sqlalchemy.sql import text as sql_text
        from app.database.postgres import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as session:
                # Prefer the Moodle section name; fall back to course_name when a
                # doc has no section_name (pre-section_name ingests). The outer
                # filter drops rows where BOTH are empty.
                topic_expr = (
                    "COALESCE(NULLIF(metadata->>'section_name', ''), "
                    "metadata->>'course_name')"
                )
                stmt = (
                    select(distinct(sql_text(topic_expr)).label("topic"))
                    .select_from(sql_text("documents"))
                    .where(sql_text(f"{topic_expr} IS NOT NULL"))
                    .where(sql_text(f"{topic_expr} <> ''"))
                )
                rows = (await session.execute(stmt)).all()
                courses = sorted({r.topic for r in rows if r.topic})
        except Exception as exc:
            logger.warning(f"Topic-name load failed (Postgres): {exc}")
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
    """Map each Moodle SECTION → its item (course_name) list, TTL-cached (10min).

    Ground truth for "apa aja di <section>" / "isi <section>" questions: a section
    like "Business Process" lists its files (Validasi UK, Pelayanan, ...). Pulled
    straight from Postgres so the answer is deterministic, never an LLM guess.
    Only includes docs that actually have a section_name (the per-section grouping
    only makes sense for section-tagged docs).
    """
    import time as _time

    now = _time.time()
    if now < _section_map_cache["expires_at"] and _section_map_cache["map"]:
        return _section_map_cache["map"]

    lock = _get_section_map_lock()
    async with lock:
        now = _time.time()
        if now < _section_map_cache["expires_at"] and _section_map_cache["map"]:
            return _section_map_cache["map"]

        from sqlalchemy.sql import text as sql_text
        from app.database.postgres import AsyncSessionLocal

        section_map: dict[str, list[str]] = {}
        try:
            async with AsyncSessionLocal() as session:
                stmt = sql_text(
                    "SELECT DISTINCT metadata->>'section_name' AS section, "
                    "metadata->>'course_name' AS item "
                    "FROM documents "
                    "WHERE metadata->>'section_name' IS NOT NULL "
                    "AND metadata->>'section_name' <> '' "
                    "AND metadata->>'course_name' IS NOT NULL "
                    "AND metadata->>'course_name' <> '' "
                    "ORDER BY 1, 2"
                )
                rows = (await session.execute(stmt)).all()
                for r in rows:
                    section_map.setdefault(r.section, [])
                    if r.item not in section_map[r.section]:
                        section_map[r.section].append(r.item)
        except Exception as exc:
            logger.warning(f"Section-map load failed (Postgres): {exc}")
            return {}

        _section_map_cache["map"] = section_map
        _section_map_cache["expires_at"] = now + _COURSE_CACHE_TTL_SECONDS
        return section_map


def _match_section(query: str, sections: list[str]) -> str | None:  # noqa: ARG001
    """REMOVED — replaced by UI-driven section navigation (see /chat/sections
    endpoint + the topic-list button in the chat UI). Kept as a no-op stub only
    if something still imports it; nothing in-tree does."""
    return None


async def _generate_node(state: RAGState, config: RunnableConfig):
    """Single conversational LLM call — the only answer-generating node.

    One CONVERSATIONAL_PROMPT handles everything: greetings, identity, meta-turns
    ("kok gini", "ga nyambung"), chit-chat, and grounded KB answers. Retrieved
    context is injected ONLY when it's actually relevant (the dense-floor gate
    via _route_after_rag passes); for a greeting / off-scope / no-match turn we
    inject NO context, so the model never gets irrelevant chunks forced into a
    casual reply — it just answers conversationally or says it doesn't have that
    info. Memory (STM summary, LTM profile, user prefs) is always injected when
    present. Conciseness + the detail/teach escalation live in the prompt.
    """
    chunks = state.get("retrieved_context") or []
    summary = state.get("conversation_summary") or ""
    profile = state.get("user_profile") or {}
    intent = state.get("intent") or "KNOWLEDGE"

    # Inject context ONLY when retrieval is genuinely relevant.
    # If we reached generate_node, the route already decided "generate" (or this is a
    # non-KNOWLEDGE intent that bypassed the gate). Trust the routing.
    has_kb_context = bool(chunks) and state.get("intent") in ("KNOWLEDGE", "COACHING")

    context_section = ""
    if has_kb_context:
        # Fix A — set/list enumeration ("produk apa aja", "sebutkan semua",
        # "8 prinsip"): the answer oscillated across multi-turn (model defended
        # an earlier wrong count, e.g. "dua doang"). Two mitigations:
        #  (1) surface the most enumerative chunk (the summary/overview that
        #      lists the whole set as bullets) FIRST so the complete list is
        #      salient to a weak model, and
        #  (2) a directive to re-derive the full set from context every turn,
        #      ignoring any count stated in a prior turn.
        import re as _re
        _user_q = (state.get("rewritten_query") or state.get("retrieval_query") or "").lower()
        _is_set_list = bool(_re.search(
            r"\bapa\s*aja\b|\bapa\s*saja\b|\bsebut(?:kan|in|ke)\b|\bsemua\b|"
            r"\bdaftar\b|\blist\b|\b\d+\s+(?:produk|prinsip|value|nilai|jenis|macam|tipe|fitur)\b",
            _user_q,
        ))
        _ordered = chunks
        if _is_set_list and len(chunks) > 1:
            # Bullet-rich chunk = the summary/overview that enumerates the set.
            def _bullet_count(c):
                return len(_re.findall(r"(?m)^\s*[-*]\s", c.get("text", "")))
            _ordered = sorted(chunks, key=_bullet_count, reverse=True)

        # Per-chunk char cap, ATX-heading strip (so an echoed chunk can't render
        # as an <h1>), then a token ceiling on the whole block.
        chunk_char_cap = _settings.lms_chunk_text_max_chars
        context_lines = []
        for i, c in enumerate(_ordered, 1):
            chunk_text = _normalize_dashes(_strip_md_headings_for_context(c.get("text", "")))
            if chunk_char_cap and len(chunk_text) > chunk_char_cap:
                chunk_text = chunk_text[:chunk_char_cap].rstrip() + "…"
            context_lines.append(
                f"[{i}] Course: {c.get('course_name', '?')} (ID:{c.get('course_id', '?')})\n"
                f"{chunk_text}"
            )
        context_str = truncate_to_tokens(
            "\n\n---\n\n".join(context_lines), _settings.max_context_tokens
        )
        context_section = f"\n\n<retrieved_context>\n{context_str}\n</retrieved_context>"
        # NOTE: the set/list enumeration RULE lives in <grounding> in the system
        # prompt (stable, rarely echoed). We deliberately do NOT append a
        # plain-prose directive to the context here — Flash Lite echoed it
        # verbatim to the user ("CATATAN PENTING: ..."). The chunk reordering
        # above (summary-first) is the structural nudge; the prompt carries the
        # instruction.

    # TOPIC_LIST: the user asked what materials/topics exist ("ada materi apa
    # aja"). This is a no-retrieval intent, so inject the GROUND-TRUTH course
    # list from Postgres — otherwise the model invents plausible-sounding topics
    # (the bug). The prompt is told to list ONLY these.
    topics_section = ""
    if intent == "TOPIC_LIST":
        try:
            course_names = await _load_course_names()
        except Exception:
            course_names = []
        # DATA ONLY — the "list these verbatim / don't invent" instruction lives
        # in CONVERSATIONAL_PROMPT's <grounding> block, NOT here. Appending a
        # plain-prose directive right after the data made the weak generator
        # echo it verbatim to the user (the topic-list leak). Same lesson as
        # context_section above: keep injected blocks pure data.
        if course_names:
            topics_section = (
                "\n\n<available_topics>\n"
                + "\n".join(f"- {c}" for c in course_names)
                + "\n</available_topics>"
            )
        else:
            # Empty list = Postgres load failed or no docs ingested. The non-empty
            # sentinel string still flips _is_grounded → True (deterministic temp-0
            # client) and the prompt's <grounding> rule tells the model to admit it
            # can't load the list rather than fabricate topics.
            topics_section = (
                "\n\n<available_topics>\n(could not load topic list right now)\n"
                "</available_topics>"
            )

    # Section drill-down. Two paths (Jun 2026):
    # (1) SECTION_DRILLDOWN (priority): `_pre_processor` resolved the section
    #     name from query/history and stored it in state["drilldown_section"].
    #     Inject the canonical list of items — fully dynamic, no KB dep.
    #     Handles "bisnis proses ada apa aja", "yang kedua", "topik B",
    #     "di leadership materinya apa".
    # (2) Legacy fallback: when retrieved KB chunks concentrate (>=60%) in one
    #     section, infer that section and inject its items. Used when the
    #     question is implicitly about a section but drilldown didn't fire.
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
                f"section_map has no items - falling back to legacy path"
            )
    if not section_section and has_kb_context and chunks:
        from collections import Counter as _Counter
        secs = [c.get("section_name", "").strip() for c in chunks if c.get("section_name", "").strip()]
        if secs:
            dom_sec, dom_n = _Counter(secs).most_common(1)[0]
            if dom_n >= max(2, (len(chunks) * 3 + 4) // 5):  # >=60% of chunks
                try:
                    items = (await _load_section_map()).get(dom_sec, [])
                except Exception:
                    items = []
                if len(items) > 1:  # a 1-item section has nothing to drill into
                    section_section = (
                        f"\n\n<section_materials section=\"{dom_sec}\">\n"
                        + "\n".join(f"- {it}" for it in items)
                        + "\n</section_materials>"
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
            import re
            ci = re.sub(r"<[^>]+>", "", ci)
            ci = re.sub(
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
    # Lets Ava greet by name and tailor answers to their dept/role/location.
    # Rendered only when at least one field is present (greetings from a
    # tokenless dev session carry nothing).
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
                "Sapa dengan nama depannya bila relevan dan sesuaikan jawaban "
                "dengan konteksnya:\n"
                + "\n".join(ctx_lines)
                + "\n</user_context>"
            )

    dynamic_tail = f"{user_ctx_section}{pref_section}{ltm_section}{summary_section}{topics_section}{section_section}{context_section}".strip()

    # Temperature split (no extra tokens — just which pre-built client we call):
    #   - GROUNDED turn (KB <context> present, or a TOPIC_LIST with the real
    #     course list) → temp 0.0 (get_generate_llm) so factual / enumeration
    #     answers are deterministic and consistent turn-to-turn. Fixes the
    #     "produk apa aja" giving a different list each time at temp 0.4.
    #   - CONVERSATIONAL turn (greeting, identity, vent, meta, no context) →
    #     temp 0.4 (get_chat_llm) so chit-chat stays warm and natural.
    #   - COACHING turn → temp 0.4 (get_chat_llm): the Socratic guiding question
    #     needs warmth/variation to feel like a trainer, not a form. Faithfulness
    #     is still enforced by SOCRATIC_PROMPT's grounding rules + the dense-floor
    #     gate (context injected only when relevant).
    # All clients share the same model/provider/streaming/usage flags. Coaching
    # uses a DIFFERENT cached system prefix (SOCRATIC_PROMPT) — that's a separate
    # prompt-cache entry, still cacheable, just not shared with the conv prefix.
    is_coaching = intent == "COACHING"
    _is_grounded = (has_kb_context or bool(topics_section) or bool(section_section)) and not is_coaching
    llm = get_generate_llm() if _is_grounded else get_chat_llm()
    windowed_messages = _window_generate_history(
        list(state["messages"]),
        max_fresh_turns=_settings.max_fresh_turns,
        max_ai_chars=_settings.max_history_ai_chars,
    )
    # Static system prompt kept byte-stable (dynamic per-turn context lives in a
    # separate HumanMessage) so the upstream's automatic prefix cache can hit on
    # the 2nd+ call. No explicit cache_control breakpoint: the configured
    # generators (DeepSeek native, Gemini via Vertex) BOTH cache implicitly /
    # server-side, where cache_control is a no-op — it's an Anthropic-style lever
    # (also honored by Alibaba on OpenRouter). If you ever pin the generator to a
    # provider that requires explicit breakpoints, re-add it here.
    #
    # Per-intent prompt selection (Option C — saves ~900 tok on chit-chat turns):
    #   - COACHING → SOCRATIC_PROMPT (full Socratic scaffolding)
    #   - KNOWLEDGE / TOPIC_LIST → CONVERSATIONAL_PROMPT (full grounding rules)
    #   - GREETING / AMBIGUOUS / OFF_SCOPE → CHIT_CHAT_PROMPT (minimal, no KB)
    if is_coaching:
        system_prompt_text = SOCRATIC_PROMPT
    elif intent in ("GREETING", "AMBIGUOUS", "OFF_SCOPE"):
        system_prompt_text = CHIT_CHAT_PROMPT
    else:
        # KNOWLEDGE, TOPIC_LIST
        system_prompt_text = CONVERSATIONAL_PROMPT
    system_msg = SystemMessage(content=system_prompt_text)
    msgs: list = [system_msg]
    # Dynamic per-turn context (LTM/summary/topics/sections/<context>) lives in a
    # SEPARATE HumanMessage, NOT appended to the SystemMessage. Reason: the system
    # prompt must stay byte-stable turn-to-turn so the upstream's implicit prefix
    # cache (Gemini via Vertex) can hit on call 2+. Appending dynamic_tail to
    # system_msg.content invalidated the prefix every turn → full-latency generate
    # each call. A standalone context message keeps the prefix cacheable while still
    # feeding the model the same context. (OpenRouter/Gemini treat the first
    # HumanMessage as the user turn — placing context before the real user turn is
    # fine since the final windowed_messages ends with the live HumanMessage.)
    if dynamic_tail:
        msgs.append(HumanMessage(content=dynamic_tail))
    msgs += windowed_messages

    _t0 = time.monotonic()
    try:
        response = await llm.ainvoke(msgs, config=config)
    except Exception as gen_exc:
        # Streaming retry-fallback: get_generate_llm runs streaming=True. If a
        # pinned provider (alibaba/baidu/novita) corrupts the SSE stream
        # mid-generation, the error raises inside astream_events at the chat.py
        # layer — uncatchable there. ainvoke here surfaces it as a clean
        # exception, so fall back to the non-stream client and retry once.
        # Non-stream is atomic: either it returns a full AIMessage or it raises
        # again (re-raised to the graph's error path). This keeps a provider
        # SSE flake from producing a partial/empty answer.
        logger.warning(
            f"generate_node stream ainvoke failed ({type(gen_exc).__name__}), "
            f"retrying non-stream: {gen_exc}"
        )
        _t0 = time.monotonic()
        response = await get_generate_llm_nostream().ainvoke(msgs, config=config)
    await _log_cache_usage(
        response,
        "generate",
        turn_id=state.get("turn_id") if isinstance(state, dict) else None,
        started_at=_t0,
    )

    # Defensive anti-leak: strip any <retrieved_context>/<user_history>/etc. block
    # the model may have echoed verbatim (also prevents `# Heading` chunks from
    # rendering as 4x font). Prompt-level OUTPUT_CONTRACT handles the 99% case.
    raw = response.content if hasattr(response, "content") else str(response)
    if isinstance(raw, str):
        cleaned = _sanitize_answer(raw)
        if cleaned != raw:
            logger.warning(
                "generate_node: stripped leaked instruction block from LLM output "
                f"(orig_len={len(raw)} clean_len={len(cleaned)})"
            )
            response.content = cleaned

    return {"messages": [response]}


# ─── Routing ─────────────────────────────────────────────────────────────────

def _route_by_intent(state: RAGState) -> str:
    return state.get("intent") or "KNOWLEDGE"


def _route_after_retrieval(state: RAGState) -> str:
    """After rag_node: a KNOWLEDGE turn whose retrieval fell below the dense
    floor is refused deterministically (no LLM) so the model can't invent facts
    when nothing valid was retrieved (e.g. fabricating an acronym expansion for
    an un-ingested term). COACHING is NOT refused — it keeps flowing to
    generate_node so the Socratic prompt can still open a guiding question."""
    if state.get("intent") in ("KNOWLEDGE", "COACHING") and _route_after_rag(state) == "low_relevance":
        return "low_relevance"
    return "generate_node"


def _route_after_rag(state: RAGState) -> str:
    """Decide whether to call the LLM or short-circuit when retrieval is weak.

    Applies to both KNOWLEDGE and COACHING — a Socratic guiding question must
    be grounded in the KB just like a factual answer, so COACHING gets NO
    special bypass: if retrieval is below the floor, context is withheld and the
    prompt's no-context / grounding rules take over ("Aku belum nemu ini di
    materiku") rather than inventing a question around nothing.

    The gate is a MANDATORY DENSE FLOOR (not an OR). bge-m3 dense cosine cleanly
    separates scope on this KB — off-scope tops out ~0.36, in-scope floors ~0.50
    — so dense alone is the discriminator:
      - `dense_score` — raw dense cosine [0, 1], an ABSOLUTE semantic signal.
        Calibrated: off-scope ceiling ≈ 0.36, in-scope floor ≈ 0.50; floor set
        at 0.40 (safe band [0.40, 0.45], leaning in-scope-protective).
    Sparse (raw BM25) is NOT consulted in normal operation: on a small KB it does
    NOT separate scope — off-scope function words ("gimana", "di", "siapa")
    accumulate BM25 (off-scope sparse 8.56 outscores in-scope "Poket" 6.23), so an
    OR-rescue on sparse re-admits off-scope. The fused `score`/`hybrid_score` is
    min-max normalized per-query (top hit ≈ 1.0 on every query), so it can't gate
    a global miss — the raw dense signal can. Sparse is read ONLY in the C5
    degraded window below (embedding outage), as a fail-open backstop.

    C4 — the gate reads POOL-LEVEL maxes (`pool_max_dense`/`pool_max_sparse`,
    set by rag_node over the full fetch_k pool BEFORE the top-k slice), not the
    per-chunk maxes of the returned top-k. A chunk with the highest raw dense
    cosine can rank below the final top-k by FUSED score (fusion blends in
    normalized sparse) and get sliced off — gating on the post-slice chunks
    would then read an artificially low max and emit a FALSE NOT-FOUND. Falls
    back to the per-chunk computation if pool stats are absent (defensive).

    C5 — when retrieval degraded to sparse-only (embedding outage,
    `dense_retrieval_ok` is False), the dense signal is MISSING, not low. We
    must not let an absent dense score force a NOT-FOUND; the gate runs on
    sparse alone in that window.
    """
    chunks = state.get("retrieved_context") or []
    if not chunks:
        return "low_relevance"

    # C4: prefer pool-level maxes computed over the full fetch_k pool. Fall back
    # to per-chunk maxes (legacy behaviour) only when pool stats are unavailable.
    pool_max_dense = state.get("pool_max_dense")
    pool_max_sparse = state.get("pool_max_sparse")
    if pool_max_dense is None and pool_max_sparse is None:
        dense_scores = [v for c in chunks if isinstance((v := c.get("dense_score")), (int, float))]
        sparse_scores = [v for c in chunks if isinstance((v := c.get("sparse_score")), (int, float))]
        if not dense_scores and not sparse_scores:
            return "low_relevance"
        max_dense = max(dense_scores) if dense_scores else 0.0
        max_sparse = max(sparse_scores) if sparse_scores else 0.0
    else:
        max_dense = float(pool_max_dense or 0.0)
        max_sparse = float(pool_max_sparse or 0.0)

    # C5: dense degraded to sparse-only — judge on sparse alone, don't let the
    # missing dense signal (0.0) manufacture a NOT-FOUND.
    dense_retrieval_ok = state.get("dense_retrieval_ok")
    if dense_retrieval_ok is False:
        if max_sparse >= _settings.kb_min_sparse_score:
            return "generate"
        logger.info(
            f"Sparse-only retrieval below sparse gate — skipping generate_node "
            f"(sparse={max_sparse:.4f} < {_settings.kb_min_sparse_score}, dense unavailable)"
        )
        return "low_relevance"

    # Normal operation: DENSE is the mandatory floor. Sparse is NOT consulted —
    # raw BM25 does not separate scope on a small KB (off-scope function-word
    # matches outscore real entities), so an OR-rescue on sparse re-admits
    # off-scope. The clean dense gap (off-scope ≤~0.36, in-scope ≥~0.50) is the
    # gate.
    if max_dense < _settings.kb_min_dense_score:
        # SPARSE RESCUE: If BM25 lexical match is extremely high (e.g. > 15.0),
        # it means there is strong keyword overlap even if dense embedding missed it.
        if max_sparse >= 15.0:
            logger.info(
                f"Sparse rescue (dense={max_dense:.4f} < {_settings.kb_min_dense_score}, "
                f"but sparse={max_sparse:.4f} >= 15.0) — admitting as KNOWLEDGE"
            )
            return "generate"

        # NEAR-MISS monitor: dense just below the floor. Could be a terse
        # entity/acronym wrongly rejected — instrument so ops can pull these and
        # decide whether a corroboration tier is needed. Does NOT change routing.
        if max_dense >= _settings.kb_min_dense_score - 0.05:
            logger.info(
                f"NEAR-MISS reject (dense just below floor) — "
                f"dense={max_dense:.4f} < {_settings.kb_min_dense_score}, "
                f"sparse={max_sparse:.4f}"
            )
            _emit_sparse_only_passthrough()
        logger.info(
            f"Dense below floor — NOT-FOUND "
            f"(pool dense={max_dense:.4f} < {_settings.kb_min_dense_score}, "
            f"pool sparse={max_sparse:.4f} ignored)"
        )
        return "low_relevance"
    return "generate"


# ─── Graph Assembly ───────────────────────────────────────────────────────────

def _build_agent_graph():
    """Build and compile the minimal conversational RAG StateGraph.

    Collapsed from the old 9-node / 7-intent router to 4 nodes. Routing by the
    regex Tier-1 label set in _pre_processor (no LLM):
        START → pre_processor → MALICIOUS                    → malicious      → END
                              → GREETING/AMBIGUOUS/OFF_SCOPE/TOPIC_LIST
                                                              → generate_node → END  (no retrieval)
                              → KNOWLEDGE → rag_node          → generate_node → END

    Chit-chat / no-lookup intents skip retrieval entirely and go straight to the
    conversational generate node with NO <context> — so a greeting or a vague
    "info dong" never gets an irrelevant chunk dumped on it, and the prompt asks
    a clarifying question instead of guessing. Only a real KNOWLEDGE question
    retrieves. generate_node is the single conversational LLM call; the canned
    handlers (greeting/ambiguity/off_scope/topic_list/low_relevance) are gone —
    their behavior lives in CONVERSATIONAL_PROMPT.
    """
    builder = StateGraph(RAGState)

    # Nodes
    builder.add_node("pre_processor", _pre_processor)
    builder.add_node("malicious", _handle_malicious)
    builder.add_node("rag_node", _rag_node)
    builder.add_node("low_relevance", _handle_low_relevance)
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
            # Real question → retrieve first.
            "KNOWLEDGE": "rag_node",
            # Coaching (Socratic) also retrieves first — the guiding question
            # must be grounded in the KB, so it flows through rag_node too.
            "COACHING": "rag_node",
        },
    )
    builder.add_edge("malicious", END)
    # rag_node → generate_node, UNLESS a KNOWLEDGE turn fell below the dense
    # floor — then route to the deterministic low_relevance refusal (no LLM, so
    # the model can't invent facts when nothing valid was retrieved). COACHING
    # below-floor still flows to generate_node (Socratic prompt handles it).
    builder.add_conditional_edges(
        "rag_node",
        _route_after_retrieval,
        {
            "low_relevance": "low_relevance",
            "generate_node": "generate_node",
        },
    )
    builder.add_edge("low_relevance", END)
    builder.add_edge("generate_node", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_rag_graph():
    """Return the singleton compiled RAG graph."""
    return _build_agent_graph()
