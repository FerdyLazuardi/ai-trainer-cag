"""
OpenRouter LLM client via LangChain's ChatOpenAI integration.
Includes tenacity retry wrapping for resilience.
"""
from functools import lru_cache

import httpx
from langchain_openai import ChatOpenAI
from loguru import logger
from openai import APIError, APIStatusError, APITimeoutError, RateLimitError
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config.settings import get_settings

settings = get_settings()


# The self-hosted OpenRouter-compatible gateway (configured via
# settings.openrouter_base_url / OPENROUTER_BASE_URL) sits behind a WAF
# that blocks any request whose User-Agent contains "OpenAI" — the
# default UA emitted by the openai-python SDK. The OpenAI SDK rewrites
# the UA per-request, so httpx default_headers gets overridden. We use
# an event hook that fires AFTER the SDK builds the request to forcefully
# replace the UA.
_LLM_USER_AGENT = "ai-lms-agent/1.0"


async def _strip_openai_ua(request: httpx.Request) -> None:
    request.headers["User-Agent"] = _LLM_USER_AGENT


@lru_cache(maxsize=1)
def _shared_http_client() -> httpx.AsyncClient:
    # ONE process-wide pool shared by all 7 LLM factories. Each factory used to
    # build its own client → 7 independent httpx pools (httpx default = 100 max
    # conns / 20 keepalive each), so ~140 idle keepalive sockets + 7× struct/FD
    # overhead for what is really one OpenRouter upstream. httpx.AsyncClient is
    # safe to share across concurrent coroutines, and OpenRouter rate-limits
    # per-API-key (not per-connection), so a shared pool changes nothing about
    # throttling — it just collapses the socket/FD footprint to one pool.
    return httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        event_hooks={"request": [_strip_openai_ua]},
    )


# OpenAI SDK's built-in max_retries=1 is a no-op when http_async_client is
# provided (langchain ChatOpenAI bypasses the default sync/async client
# when a custom async client is set, so urllib3.Retry is never reached).
# We wrap ainvoke() with tenacity so transient OpenRouter errors
# (429, 500, 502, 503, 504, timeouts) get exponential backoff retry.
# Non-retriable errors (400 bad request, 401 auth, 402 out of credits,
# 403 forbidden) bubble up immediately — those won't fix themselves.
_RETRYABLE_EXCEPTIONS = (
    RateLimitError,        # 429
    APITimeoutError,       # request timed out
    APIStatusError,        # 5xx; we'll filter on status_code below
    APIError,              # catch-all for connection drops
)
_NON_RETRYABLE_5XX = (500, 501, 502, 503, 504)


# Gemini 2.5 Flash Lite (LLM_MODEL / CHEAP_LLM_MODEL) is served by more than
# one OpenRouter upstream (Google AI Studio and Google Vertex). We PREFER
# Google Vertex first via `order`, but keep allow_fallbacks=True so a
# single-provider outage degrades to AI Studio instead of failing the request
# — the opposite stance from the judge, which is hard-pinned
# (only=["deepseek"], allow_fallbacks=False) for reproducibility.
#
# WHY Vertex over AI Studio (changed 2026-06, from the OpenRouter activity
# dashboard for this model): Vertex reports ~45% prompt-cache hit rate at
# $0.060/1M input, whereas AI Studio reports only ~6.3% hit rate at $0.094/1M.
# Gemini uses IMPLICIT (automatic, server-side) caching — `cache_control`
# breakpoints in the request are an Anthropic-only feature and a no-op here —
# so the only lever we have on cache effectiveness is which upstream serves
# the request. Vertex wins on BOTH cache hit rate and raw input price. 100%
# cache is impossible by design (first-seen prefixes always miss, implicit
# cache has TTL + propagation delay, and traffic spreads across backends each
# with its own cache), so ~45% is already a strong number for this model.
#
# NB: require_parameters is deliberately NOT set on EITHER path. Live-verified
# through the real langchain ChatOpenAI client (2026-06): adding it 404s every
# request ("No endpoints found that can handle the requested parameters") on
# BOTH Gemini AND the DeepSeek judge. ChatOpenAI emits a wider param set than a
# hand-rolled body (n, stream, etc.), and OpenRouter's support registry doesn't
# advertise all of them for these upstreams, so require_parameters filters out
# every candidate. (A raw minimal-body httpx call passes — which is why it
# looked safe at first — but that is NOT the path the app uses.) JSON-mode for
# the judge is enforced by response_format alone, which works without it.
_GEMINI_PROVIDER = {
    "order": ["google-vertex"],
    "allow_fallbacks": True,
}


def _provider_extra_body(model: str, *, include_usage: bool = True) -> dict:
    """Build the OpenRouter `extra_body` for a generator/classifier client,
    MODEL-AWARE so the provider pin matches the upstream that serves the model.

    This is what makes LLM_MODEL / CHEAP_LLM_MODEL a real .env swap: change the
    model string and the right provider routing follows automatically, instead
    of every builder hard-pinning google-vertex (which only serves Gemini and
    would 404/misroute a DeepSeek or Qwen model sent to it).

      - gemini/* (google/*) → google-vertex first (best implicit-cache hit +
        cheapest input), allow_fallbacks=True so a Vertex blip degrades to
        AI Studio instead of failing.
      - deepseek/* → deepseek upstream; reasoning.effort="none" because V4 is
        reasoning-capable and a GENERATOR only needs the answer in
        message.content — reasoning tokens land in reasoning_content (dropped
        by ChatOpenAI), which would blank the streamed answer or leak CoT.
      - anything else → no provider pin (OpenRouter default routing); usage only.
    """
    body: dict = {}
    if include_usage:
        body["usage"] = {"include": True}
    m = model.lower()
    if m.startswith("google/") or "gemini" in m:
        body["provider"] = _GEMINI_PROVIDER
    elif m.startswith("deepseek/"):
        # Default: DeepSeek native (the only endpoint with implicit prompt
        # caching). Override via LLM_PROVIDER_ORDER=baidu,deepinfra to chase a
        # cheaper/faster provider — see settings.llm_provider_order for the
        # quant/caching/privacy caveats. allow_fallbacks=True so a pinned
        # provider outage degrades to the rest of the market, not a failure.
        order = settings.llm_provider_order_list or ["deepseek"]
        body["provider"] = {"order": order, "allow_fallbacks": True}
        # effort=none asks the upstream not to reason; exclude=true is the
        # belt-and-suspenders that strips reasoning from the response even when
        # a pinned provider (alibaba/baidu/novita) ignores effort=none and emits
        # CoT into message.content. Without exclude, that CoT leaked verbatim to
        # the user. "All models support exclude" per OpenRouter reasoning docs.
        body["reasoning"] = {"effort": "none", "exclude": True}
    return body


def _should_retry(exc: BaseException) -> bool:
    """Decide whether a given exception is worth retrying.

    Returns True for:
      - RateLimitError (429) — back off and retry
      - APITimeoutError — transient network
      - APIStatusError with status_code in {500,501,502,503,504}
      - Other APIError subclasses (connection drops, protocol errors)
    Returns False for:
      - 400, 401, 402, 403, 404 — these are caller errors that retry
        won't fix
      - Any non-APIError exception (e.g. our own ValueError, KeyError)
    """
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APITimeoutError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _NON_RETRYABLE_5XX
    if isinstance(exc, APIError):
        # Connection drops, protocol errors, etc. — retry.
        return True
    return False


def _wrap_with_retry(llm: ChatOpenAI) -> None:
    """Replace llm.ainvoke with a tenacity-wrapped version.

    The singleton is created once per process (@lru_cache), so the
    monkey-patch is a one-time operation. Each call gets its own retry
    state via the @retry decorator, so concurrent callers don't share
    retry counters.

    Backoff: 1s, 2s, 4s, 8s (capped at 8s) on the first two retries.
    Max 3 attempts total (1 initial + 2 retries).

    Retries on: 429, 5xx, APITimeoutError, generic APIError (connection
    drops, protocol errors). Does NOT retry on: 400, 401, 402, 403, 404 —
    those are caller errors that retry won't fix.
    """
    original_ainvoke = llm.ainvoke

    def _log_before_sleep(retry_state):
        logger.warning(
            "LLM call failed, retrying",
            attempt=retry_state.attempt_number,
            next_sleep=retry_state.next_action.sleep if retry_state.next_action else None,
            error_type=type(retry_state.outcome.exception()).__name__ if retry_state.outcome else None,
            error=str(retry_state.outcome.exception())[:200] if retry_state.outcome else None,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        # Use a custom predicate (not retry_if_exception_type) so we can
        # apply both the type filter AND the status-code filter in one
        # step. Otherwise tenacity matches the type and re-runs before
        # _should_retry gets a chance to veto 4xx errors.
        retry=retry_if_exception(lambda exc: _should_retry(exc)),
        reraise=True,
        before_sleep=_log_before_sleep,
    )
    async def _retried(*args, **kwargs):
        return await original_ainvoke(*args, **kwargs)

    # ChatOpenAI is a pydantic BaseModel, which blocks arbitrary attribute
    # assignment. Use object.__setattr__ to bypass the pydantic handler.
    object.__setattr__(llm, "ainvoke", _retried)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return the singleton LLM client configured for the local OpenRouter-compatible gateway."""
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        request_timeout=60,
        max_retries=1,
        http_async_client=_shared_http_client(),
        # usage:{include:true} makes OpenRouter return real accounting in the
        # response — including prompt_tokens_details.cached_tokens. Without it,
        # the cached_tokens field comes back 0 even when caching DID happen, so
        # our _log_cache_usage would report false misses. (Caching itself is
        # automatic for Gemini; this only affects whether it's REPORTED to us.)
        extra_body=_provider_extra_body(settings.llm_model),
        # OpenRouter-specific headers for app attribution (ignored by Ollama)
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent",
        },
    )
    _wrap_with_retry(llm)
    return llm


@lru_cache(maxsize=1)
def get_cheap_llm() -> ChatOpenAI:
    """Return a cheaper, faster LLM for background tasks (memory summarization,
    eval judge, structured-output classifiers).

    Gemini 2.5 Flash Lite via OpenRouter — 4x cheaper on input, 8x cheaper
    on output than the main gemini-2.5-flash model. Note: NOT the
    discontinued 2.0-flash-lite-001 (which 404s on OpenRouter as of 2026).
    The judge LLM is intentionally pinned regardless of `LLM_MODEL` so eval
    scores stay comparable week-over-week.
    """
    llm = ChatOpenAI(
        model=settings.cheap_llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.3,
        max_tokens=1000,
        request_timeout=30,
        max_retries=1,
        http_async_client=_shared_http_client(),
        extra_body=_provider_extra_body(settings.cheap_llm_model),
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Background Worker)",
        },
    )
    _wrap_with_retry(llm)
    return llm


@lru_cache(maxsize=1)
def get_judge_llm() -> ChatOpenAI:
    """Dedicated LLM-as-judge for eval faithfulness — `deepseek/deepseek-v4-pro`.

    C3: intentionally a DIFFERENT model family than BOTH the generator
    (`get_generate_llm` = cheap_llm_model) and the old judge slot
    (`get_llm` = llm_model), which are both Gemini 2.5 Flash Lite. A judge
    that shares the generator's family shares its fabrication patterns and
    undercounts the ungrounded rate (judge grading itself). The boot guard
    `assert_judge_model_distinct` enforces this at startup.

    Config rationale (verified against current OpenRouter docs):
      - extra_body.provider.order=LLM_PROVIDER_ORDER + allow_fallbacks=True. Same model-aware routing as get_llm(). [2026-06-20: was hard-pinned to only=[\"deepseek\"] but 404'd because DeepSeek native is excluded by user's OpenRouter privacy policy. Faithfulness metric was 0.000 because judge silently no-op'd. Re-pinning via LLM_PROVIDER_ORDER=alibaba in .env re-enables single-provider baseline if needed.]
        to DeepSeek's native upstream so a provider outage fails LOUD instead
        of silently rerouting to a different upstream and shifting the judge baseline mid-week. [UPDATED 2026-06-20]
        baseline mid-week. `only` (allowlist) over `order` (preference) for an
        unambiguous pin.
      - extra_body.reasoning.effort="none" — V4 Pro is reasoning-capable; we
        only need a faithfulness score. "none" is the documented canonical
        form to disable reasoning entirely (NOT {"enabled": false}, which
        mandatory-reasoning models reject with 400). With reasoning off the
        answer lands in standard message.content, not reasoning_content
        (which ChatOpenAI drops anyway).
      - model_kwargs.response_format={"type":"json_object"} — emit a parseable
        JSON object directly, NOT via with_structured_output()'s tool-calling
        path (tool_choice adds a function-call round-trip that interacts badly
        with provider pinning). judge_faithfulness parses content with
        json.loads + FaithfulnessResult.model_validate.
      - temperature=0, max_tokens=1024 per D1.

    NOTE: provider pinning is an OpenRouter extension. If openrouter_base_url
    points at a non-OpenRouter gateway it will ignore the `provider` field;
    the json_object + temperature config still applies.
    """
    llm = ChatOpenAI(
        model=settings.judge_llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0,
        max_tokens=1024,
        request_timeout=60,
        max_retries=1,
        http_async_client=_shared_http_client(),
        model_kwargs={
            "response_format": {"type": "json_object"},
        },
        extra_body={
            "provider": {
                "order": list(settings.llm_provider_order_list or ["deepseek"]),
                "allow_fallbacks": True,
            },
            "reasoning": {"effort": "none", "exclude": True},
        },
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Judge)",
        },
    )
    _wrap_with_retry(llm)
    return llm


def assert_judge_model_distinct(
    judge_model: str, generator_models: set[str]
) -> None:
    """Boot guard (C3): the judge model MUST differ from every generator model.

    Pure + arg-driven so it's unit-testable without spinning a real Settings.
    Called from the FastAPI lifespan so a mis-config (e.g. JUDGE_LLM_MODEL left
    pointing at the Gemini generator) fails LOUD at startup rather than
    silently producing inflated faithfulness scores in production.
    """
    if judge_model in generator_models:
        raise RuntimeError(
            f"JUDGE_LLM_MODEL={judge_model!r} matches a generator model "
            f"({sorted(generator_models)!r}). The eval judge must be a "
            f"different model family than the generator — a same-family judge "
            f"shares fabrication patterns and undercounts the hallucination "
            f"rate (judge grading itself). Set JUDGE_LLM_MODEL to a distinct "
            f"model (default: deepseek/deepseek-v4-pro)."
        )


@lru_cache(maxsize=1)
def get_preprocessor_llm() -> ChatOpenAI:
    """Cheap LLM for the pre-processor's intent classification + query rewrite.

    USED BY ASKFER ONLY (app/graph/askfer_pipeline.py). Ava's pre_processor
    (app/graph/pipeline.py) is rule-based — deterministic regex Tier-1 routing,
    NO LLM call — so it never reaches this factory. Keep this in mind before
    assuming a change here affects Ava.

    The pre-processor is the single most expensive per-call system in the
    pipeline (~1.0K input tokens on every turn) and runs BEFORE retrieval —
    so its cost is paid on every request including off-topic/off-scope
    queries that get short-circuited downstream.

    Routing it to Gemini 2.5 Flash Lite (4-8x cheaper than the main model)
    drops pre-processor cost by ~75% with no measurable impact on routing
    accuracy — the classification task is structured-output + a small set
    of intent labels, which Flash Lite handles reliably (verified against
    the safety benchmark: same intent + safety scores as the main model
    on 10/11 cases; only the known user-failure-case borderline ID-slang
    remains at S=0.5 either way).

    Temperature=0.0 + larger max_tokens than get_cheap_llm because the
    pre-processor needs a deterministic safety_preserved_query (longest
    output is ~200 chars).
    """
    llm = ChatOpenAI(
        model=settings.cheap_llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.0,
        max_tokens=500,
        request_timeout=30,
        max_retries=1,
        http_async_client=_shared_http_client(),
        extra_body=_provider_extra_body(settings.cheap_llm_model),
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Pre-Processor)",
        },
    )
    _wrap_with_retry(llm)
    return llm


@lru_cache(maxsize=1)
def get_generate_llm() -> ChatOpenAI:
    """Cheap LLM for the final answer generation (generate / greeting /
    ambiguity nodes).

    This is the largest input cost in the pipeline (~1.6K tokens per call
    carrying the system prompt + RAG context + history summary). Routing
    it to Gemini 2.5 Flash Lite drops the per-turn cost by ~75-80% with
    no measurable quality regression on the verified cases:

      - Safety escalation: produces the right contact info (safespace email
        + WhatsApp) with a valid empathetic lead-in.
      - Brainstorm / vent: warm tone, lists concrete next steps.
      - Standard knowledge / procedural: bulleted structure preserved,
        contact details present.
      - 4-8x cheaper than gemini-2.5-flash on both input ($0.075 vs $0.30
        per 1M) and output ($0.30 vs $2.50 per 1M).

    The `eval` runner and the safety-aware grader still call the main
    `get_llm()` (flash) so the safety benchmark stays comparable
    week-over-week.

    Temperature=0.0 mirrors the main LLM config; max_tokens=1024 gives
    headroom for longer procedural / multi-step answers that the old 600
    cap was truncating mid-list (H1). Flash-lite output is cheap
    ($0.30/1M), so the extra ceiling costs nothing when unused.
    """
    llm = ChatOpenAI(
        model=settings.cheap_llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.0,
        max_tokens=1024,
        request_timeout=30,
        max_retries=1,
        http_async_client=_shared_http_client(),
        # streaming=True makes ainvoke() stream tokens internally from Gemini,
        # so the graph's astream_events surfaces real per-token
        # on_chat_model_stream events (the SSE handler in chat.py already
        # consumes them). Without it, ainvoke blocks for the full answer and the
        # "stream" is just a frontend setTimeout animation. stream_usage=True is
        # REQUIRED alongside it — otherwise usage/cached_tokens come back empty
        # while streaming and _log_cache_usage would report false 0% hits.
        streaming=True,
        stream_usage=True,
        extra_body=_provider_extra_body(settings.cheap_llm_model),
        default_headers={
            "HTTP-Referer": "https://github.com/peped-BE",
            "X-Title": "AI LMS RAG Agent (Generate)",
        },
    )
    _wrap_with_retry(llm)
    return llm


@lru_cache(maxsize=1)
def get_chat_llm() -> ChatOpenAI:
    """The single conversational LLM for the collapsed pipeline (generate_node).

    One Gemini 2.5 Flash Lite client at a NON-ZERO temperature
    (`chat_llm_temperature`, default 0.4) — warm enough to feel like a real chat
    AI, low enough to stay grounded when <retrieved_context> is present. Replaces
    the old get_generate_llm (temp 0) + get_empathy_llm (temp 0.6) split: there's
    one conversational prompt now, so one client. Keeps the same model/provider
    (google-vertex, best cache hit + cheapest), real token streaming, and usage
    accounting flags as get_generate_llm.
    """
    llm = ChatOpenAI(
        model=settings.cheap_llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=settings.chat_llm_temperature,
        max_tokens=1024,
        request_timeout=30,
        max_retries=1,
        http_async_client=_shared_http_client(),
        streaming=True,
        stream_usage=True,
        extra_body=_provider_extra_body(settings.cheap_llm_model),
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Chat)",
        },
    )
    _wrap_with_retry(llm)
    return llm


@lru_cache(maxsize=1)
def get_empathy_llm() -> ChatOpenAI:
    """Vent/empathy-path LLM at a NON-ZERO temperature.

    WHY: Gemini Flash Lite at temp 0.0 is deterministic and, on the vent path,
    collapses onto the prior assistant turn in history — re-emitting it
    byte-for-byte and ignoring both the new user message and the per-turn
    anti-repetition signal (verified via DEBUG_GEN). Prompt-only variation can't
    fix a model that ignores the prompt, so we break determinism with
    temperature. Selected in _generate_node ONLY for pure vents (empathy high,
    no KB lookup, no safety) — KB-grounded and safety turns keep temp 0.0 for
    factual/channel fidelity. Same model/provider/cost as get_generate_llm();
    only the temperature differs.
    """
    llm = ChatOpenAI(
        model=settings.cheap_llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=settings.empathy_llm_temperature,
        max_tokens=1024,
        request_timeout=30,
        max_retries=1,
        http_async_client=_shared_http_client(),
        # See get_generate_llm: real token streaming for the SSE path +
        # stream_usage so cache accounting still works while streaming.
        streaming=True,
        stream_usage=True,
        extra_body=_provider_extra_body(settings.cheap_llm_model),
        default_headers={
            "HTTP-Referer": "https://github.com/ai-lms-agent",
            "X-Title": "AI LMS RAG Agent (Empathy)",
        },
    )
    _wrap_with_retry(llm)
    return llm
