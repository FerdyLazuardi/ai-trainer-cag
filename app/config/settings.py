"""
Application settings using Pydantic BaseSettings.
All values are loaded from environment variables or .env file.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Application ───────────────────────────────────────────────────────
    # Required, no default. The previous 'development' default meant a prod
    # deploy that forgot to set APP_ENV would boot with dev bypass active
    # (auth.py:36 — no JWT required) and rate limiting disabled (auth.py:79).
    # Pydantic raises ValidationError at startup if this is missing.
    # No `env="APP_ENV"` — pydantic-settings 2.x reads `app_env` ↔ `APP_ENV`
    # via default case-insensitive matching, and Pydantic v2 deprecated the
    # extra `env=` kwarg with PydanticDeprecatedSince20 (removal in v3).
    app_env: Literal["development", "staging", "production"]
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ─── CORS ──────────────────────────────────────────────────────────────
    # Previously main.py used allow_origins=["*"] + allow_credentials=True,
    # which is a forbidden combination — browsers will refuse to send cookies
    # or Authorization headers, but the misconfiguration still leaks
    # preflight responses to any origin and signals 'this API trusts every
    # caller'. Default to the dev compose ports; override in prod via env.
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://localhost:8001",
        ]
    )

    # ─── PostgreSQL ─────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "lms_ai"
    postgres_user: str = "admin"
    # No default. 'admin'/'postgres' defaults that ship in tutorials let a
    # fresh deploy come up with an open database; pydantic-settings will
    # raise on a missing POSTGRES_PASSWORD env var instead.
    postgres_password: str = Field(...)
    # Pool sizing is per-PROCESS. With both the API container and the
    # worker container reading the same values, the total connection
    # ceiling is `2 * (pool_size + max_overflow)`. The default of 10 +
    # 20 = 30 per process was correct for `max_connections=100` but
    # docker-compose.yml:89 overrides Postgres to `max_connections=50`,
    # so 30 + 30 = 60 > 50 → intermittent "too many clients" / "remaining
    # connection slots are reserved" at peak.
    #
    # docker-compose.yml sets per-service overrides so:
    #   api:    pool_size=8,  max_overflow=12  → max 20 connections
    #   worker: pool_size=4,  max_overflow=6   → max 10 connections
    # Total app ceiling: 30. Postgres max_connections: 50.
    # Leaves 20 connections for admin sessions, psql, migration scripts,
    # and Postgres's own reserved slots (3 superuser reservations on
    # stock Postgres 16 by default).
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20
    # Fail fast instead of SQLAlchemy's 30s default: a request that can't get a
    # connection within this window raises immediately rather than hanging.
    postgres_pool_timeout: int = 10

    @computed_field  # type: ignore[misc]
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ─── Redis ──────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_max_connections: int = 100
    # Logical DB for the streaq task queue, kept SEPARATE from the app's data
    # DB (`redis_db`=0). Rationale (C6): the conversation HASHes carry a 24h
    # key-level TTL and are deliberately evictable under `volatile-lru`; the
    # streaq queue keys are durable (no TTL) and must never be co-mingled with
    # evictable app keys. A separate logical DB also lets ops FLUSHDB the app
    # data at cutover (e.g. clearing legacy `rag:conv:*`) without nuking
    # in-flight jobs. NOTE: maxmemory is per-INSTANCE, not per-DB — the
    # eviction *safety* comes from volatile-lru only evicting TTL-bearing keys
    # (queue keys have none); the DB split is operational isolation, not a
    # memory partition.
    redis_queue_db: int = 1

    @computed_field  # type: ignore[misc]
    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @computed_field  # type: ignore[misc]
    @property
    def streaq_redis_url(self) -> str:
        """Redis URL for the streaq worker/enqueue side — points at the
        isolated queue DB (`redis_queue_db`). Both the API (enqueue) and the
        worker (consume) import the same `worker` object, so pointing the
        Worker here moves BOTH sides onto the queue DB automatically."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_queue_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_queue_db}"

    # ─── Qdrant ─────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    # gRPC port — SEPARATE from HTTP port. The qdrant-client library's
    # AsyncQdrantClient computes its gRPC port independently of HTTP, and
    # the qdrant container exposes 6334 (gRPC) and 6333 (HTTP) by default.
    # Our docker-compose maps host 6335→container 6333 (HTTP) and host
    # 6336→container 6334 (gRPC), so when running outside docker (eval
    # scripts, ad-hoc scripts) the gRPC port must be 6336, NOT 6334
    # (which only works inside the docker network). Without this, the
    # qdrant-client's gRPC stub logs UNAVAILABLE on 127.0.0.1:6334 even
    # when `prefer_grpc=False`, because some async paths still
    # lazily initialize the gRPC channel.
    qdrant_grpc_port: int = 6336
    qdrant_collection: str = "documents"
    qdrant_kb_collection: str = "Knowledge_Base"

    # ─── Embedding ──────────────────────────────────────────────────────────
    # BAAI/bge-m3 — 568M params, native 1024-dim, MIT license, $0.01/1M tok
    # via OpenRouter. Chosen over qwen3-embedding-8b (which would be 3-5×
    # slower at p50 400-700ms) and text-embedding-3-small (2× cost) because:
    #   - MIRACL-id dense leader (56.1 nDCG@10) — the corpus is ID-heavy.
    #   - p50 latency 80-180ms vs qwen3's 400-700ms → noticeably snappier
    #     RAG (saves ~300-500ms per turn on the embedding call).
    #   - Native 1024-dim matches EMBEDDING_DIM=1024 with no MRL truncation
    #     loss (qwen3 would have to truncate 4096 → 1024).
    #   - BM25 still carries exact-match jargon (course codes, brand names)
    #     where bge-m3 dense is weaker, so hybrid design stays.
    # bge-m3's sparse/ColBERT modes are NOT exposed through OpenRouter
    # (only dense); the hybrid retriever's fastembed BM25 arm continues
    # to do the lexical work.
    embedding_model: str = "baai/bge-m3"
    embedding_dim: int = 1024

    # ─── LLM (OpenRouter) ───────────────────────────────────────────────────
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_embedding_key: str = Field(default="", alias="OPENROUTER_EMBEDDING_KEY")
    # OpenRouter is the SOLE LLM provider (no 9Router, no localhost, no Ollama).
    # Default is the real OpenRouter cloud; override in .env only when routing
    # through a different OpenRouter-compatible gateway.
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_embedding_url: str | None = None
    # Main model: Gemini 2.5 Flash Lite (Google) — 4-8x cheaper than
    # gemini-2.5-flash with comparable quality on structured-output tasks.
    # Supports OpenRouter prompt caching via cache_control content blocks.
    llm_model: str = "google/gemini-2.5-flash-lite"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    # Cheap-Lite pin for background tasks (memory summarization, eval judge,
    # pre-processor intent classification, generate node). Same as the main
    # model: Gemini 2.5 Flash Lite.
    #
    # Why not a cheaper model (Llama 3.1 8B $0.02/M, Mistral Nemo $0.02/M)?
    # Every model cheaper than Flash Lite on OpenRouter (Qwen 2.5 7B, Llama
    # 3.1 8B, Mistral Nemo, Phi-4 mini) SILENTLY IGNORES `cache_control`,
    # which would make the cached_system_message() helper in
    # app/llm/client.py:28 a no-op on the cheap slot. Pinning cheap to
    # Flash Lite preserves the prompt-cache benefit on every call (1h TTL
    # cache reads at $0.01/M vs $0.10/M full input — ~48% per-turn
    # savings, ~$24/mo at 600 DAU, scales to $500+/mo at 13k DAU).
    #
    # The 10% cheaper Llama 3.1 8B class is technically cheaper on a
    # like-for-like token basis, but intent classification runs on EVERY
    # turn (not just non-cached) and the JSON-schema adherence on the
    # structured-output slots regresses ~10% on the 7B class — not worth
    # ~$12/mo in absolute savings vs the quality risk on the highest-
    # traffic slot. If cost pressure grows, demote intent classification
    # back to a 7B class and keep Flash Lite on judge + generate.
    cheap_llm_model: str = "google/gemini-2.5-flash-lite"

    # ─── LLM-as-judge (eval faithfulness) ────────────────────────────────────
    # C3: the judge MUST be a different model family than the generator. When
    # judge == generator they share fabrication patterns and the eval
    # systematically undercounts the ungrounded/hallucination rate (the judge
    # effectively grades its own output). Both the old judge slot (`get_llm` =
    # llm_model) and the generator (`get_generate_llm` = cheap_llm_model) are
    # Gemini 2.5 Flash Lite, so the judge was Flash-Lite grading Flash-Lite.
    #
    # DeepSeek V4 Pro is pinned to DeepSeek's NATIVE OpenRouter provider with
    # allow_fallbacks=false (so a provider outage can't silently reroute and
    # shift the judge baseline mid-week), reasoning DISABLED (effort:"none" —
    # we only need a faithfulness score, not chain-of-thought, and reasoning
    # tokens would land in reasoning_content which ChatOpenAI drops), and
    # response_format=json_object (NOT tool_choice — structured-output via
    # tool calling adds a function-call round-trip the gateway may not pin
    # cleanly under provider routing). See app/llm/client.get_judge_llm.
    #
    # Fallback (manual swap via env) is Qwen 2.5 72B — also distinct from the
    # Gemini generator family — if V4 Pro correlation vs gold-grade fails the
    # 50-query calibration (D3).
    judge_llm_model: str = Field(
        default="deepseek/deepseek-v4-pro", alias="JUDGE_LLM_MODEL"
    )
    judge_llm_fallback_model: str = Field(
        default="qwen/qwen-2.5-72b-instruct", alias="JUDGE_LLM_FALLBACK_MODEL"
    )

    # ─── OpenRouter prompt caching ───────────────────────────────────────────
    # When True, system prompts are wrapped in content blocks with
    # cache_control={type: "ephemeral", ttl: "1h"} so OpenRouter can
    # serve the prefix from cache on the 2nd+ identical call. Static
    # prompts (persona, output contract, mode rules) get the cache
    # breakpoint; dynamic per-turn context (retrieved chunks, user
    # history, preferences) lives in a separate non-cached user
    # message so it doesn't invalidate the cache.
    openrouter_prompt_cache_enabled: bool = True
    openrouter_prompt_cache_ttl: str = "1h"  # "5m" or "1h" per OpenRouter spec

    # ─── Evaluation (LLM-as-judge, async via streaq) ─────────────────────────
    # Phase 1: faithfulness eval only. Sample to control cost — 100% eval
    # at 13k user scale would 2x the LLM bill for diminishing signal.
    # Always-eval gates (low dense relevance / high empathy) catch high-stakes
    # turns regardless of sample rate so the riskiest outputs are never missed.
    eval_enabled: bool = Field(default=True, alias="EVAL_ENABLED")
    eval_sample_rate: float = Field(default=0.10, alias="EVAL_SAMPLE_RATE")
    eval_always_if_dense_below: float = Field(default=0.30, alias="EVAL_ALWAYS_IF_DENSE_BELOW")
    eval_always_if_empathy_above: float = Field(default=0.90, alias="EVAL_ALWAYS_IF_EMPATHY_ABOVE")

    # D3 LOCKED faithfulness threshold. An answer with judge.score < this is
    # graded UNFAITHFUL (runner.py gates on `judge.score < min_faithfulness`).
    #
    # LOCKED at 0.75 (not the provisional 0.70) on calibration evidence from a
    # balanced, discrimination-valid set (41 items: 15 faithful / 10 partial /
    # 16 hallucinated; refusals excluded from the math, judged via the
    # provided-answer harness mode). DeepSeek V4 Pro was VALIDATED against human
    # gold: Spearman ρ=0.915 [0.82,0.98], quadratic-weighted κ=0.906, and all
    # 16/16 realistic hallucinations scored 0.00 (perfect hard-negative
    # detection) — so the judge is trustworthy and was NOT swapped to the Qwen
    # fallback.
    #
    # Why 0.75 over 0.70: moving the gate 0.70→0.75 raised best-τ Cohen's
    # κ 0.515→0.741 and precision 0.62→0.81 while recall stayed IDENTICAL at
    # 0.87 — i.e. it rejects 5 additional partial answers (each mostly grounded
    # but carrying one fabricated claim: an invented POJK number, a made-up
    # limit, a daily-visit mandate, etc.) at ZERO cost to genuinely faithful
    # answers. This matches the gate's purpose (a faithfulness gate should fail
    # answers with unsupported claims) and the gold_floor=0.70 binarization. The
    # κ-max τ (0.75) is itself a stable plateau (0.75/0.80=0.741). Re-calibrate
    # with more natural partials if the judge model or generator changes.
    # Calibration harness: app/eval/calibrate_judge.py;
    # dataset builder: data/eval/build_balanced_calibration.py.
    faithfulness_min: float = Field(default=0.75, alias="FAITHFULNESS_MIN")

    # ─── Intent Classification (Tier-0 semantic gate) ──────────────────────
    # Used by app/graph/intent_classifier.classify_semantic. Embedding-based
    # cosine similarity against pre-computed intent centroids. Replaces the
    # regex-only path for novel greeting variants (assalam, shalom, om
    # swastiastu, etc.) so the LLM pre-processor is skipped for ~50%+ of
    # greeting traffic that the regex misses. NO hardcoded language patterns.
    #
    # threshold: minimum best-centroid similarity to commit to an intent.
    #   0.55 (bge-m3 cosine stopgap). Was 0.78 for text-embedding-3-small
    #   which has a steeper cosine distribution. bge-m3 puts valid
    #   matches lower in cosine space — re-measure on 50 centroids via
    #   scripts/calibrate_intent_threshold.py and tune. Raise to 0.65+ if
    #   you see false-positive intent commits.
    # margin: minimum gap between best and second-best centroid. Stops the
    #   gate from committing on ambiguous queries that sit between two
    #   centroids (e.g. "halo info" between GREETING and AMBIGUOUS).
    intent_semantic_threshold: float = 0.55
    intent_semantic_margin: float = 0.10
    # Master switch for the Tier-1.5 semantic gate (app/graph/intent_classifier.
    # classify_semantic), wired into _pre_processor between the regex Tier-1 and
    # the LLM pre-processor. DEFAULT OFF: the threshold above (0.55) is a
    # self-described uncalibrated stopgap, and a false-positive would mis-route a
    # production turn — so the gate stays dormant until calibrated against real
    # traffic via scripts/calibrate_intent_threshold.py, then flipped on by env.
    # When enabled, the gate ONLY short-circuits the canned, score-free intents
    # (GREETING/AMBIGUOUS/OFF_SCOPE/TOPIC_LIST) — never KNOWLEDGE/BRAINSTORM
    # (they need the LLM's 4-axis scores) and never MALICIOUS (injection
    # detection stays on the deterministic regex + LLM path), so it can never
    # strip a safety/vent turn's escalation scores. Off → behaviour is
    # byte-identical to today (regex Tier-1 → LLM).
    intent_semantic_gate_enabled: bool = Field(
        default=False, alias="INTENT_SEMANTIC_GATE_ENABLED"
    )

    # ─── Context Engineering ────────────────────────────────────────────────
    # Hard ceiling on the retrieved-context block sent to the generate LLM.
    # 3000 (was 6000): production telemetry showed median context fill was
    # ~1100 tokens; 6000 was 5x headroom for ~zero gain on quality. Cuts
    # input tokens on the generate call by ~45% on full-context turns.
    max_context_tokens: int = 3000
    # Candidate pool pulled per modality (dense + sparse BM25) before fusion.
    # 20 (was 10): at ~42 KB chunks 10 already covered ~25% of the corpus, but
    # the KB is scaling toward ~300+ chunks where a narrow pool would miss
    # relevant hits before fusion. Widening the pool costs ~nothing — it does
    # NOT add embedding calls or LLM tokens (only final_top_k reaches the LLM).
    retrieval_top_k: int = 20
    # Final number of fused chunks fed to the generate LLM. 3 (was 4): the
    # safety + harassment cases need ≤ 3 chunks; the multi-hop KB hot-fix
    # showed 3 chunks gives the same answer as 4 for 85% of queries. The
    # dropped chunk is always the lowest-fused score. Saves ~225 input
    # tokens/turn (~18% on the chunk portion). If multi-hop quality
    # regresses, raise to 4.
    final_top_k: int = 3
    # Per-chunk char cap for the LMS generate path. 1600 (was 1000): 5/52
    # KB chunks exceed 1000 (largest 1513) and were silently trimmed at
    # retrieval — "Profile-Amartha.md" 8-DNA list was cut at "Planning &
    # Organi…". 1600 covers p100 of current KB. Fits 2 chunks × 1600 ≈
    # 3200 chars ≈ 800 tokens under max_context_tokens=3000 ceiling.
    # Phase 3 (switch TokenTextSplitter→MarkdownNodeParser + re-ingest)
    # is the proper fix so oversized chunks stop existing at all; this
    # is the stopgap that fixes the user-reported budaya bug now.
    lms_chunk_text_max_chars: int = 1600
    # Relative-score fusion weights: fused = vector_weight * dense_norm +
    # bm25_weight * sparse_norm. vector_weight is the `alpha` in hybrid_search.
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    # MANDATORY DENSE FLOOR for the NOT-FOUND gate (app/graph/pipeline.py
    # _route_after_rag). Below this RAW DENSE COSINE pool-max, treat retrieval as
    # a miss and skip generate_node (saves ~2700 input tokens + 1 LLM call).
    # Absolute [0, 1] cosine. bge-m3 cleanly separates scope on this KB:
    # off-scope tops out ≈ 0.36 (weather/recipe/trivia), in-scope floors ≈ 0.50.
    # Set to 0.40 — safe band [0.40, 0.45], chosen toward the low end to protect
    # legit in-scope queries (production not-found turns can land ≈ 0.45). Dense
    # is now the SOLE normal-operation discriminator; sparse is NOT OR'd in (see
    # kb_min_sparse_score). Re-derive after major KB growth or any embedding-model
    # change via: python -m app.eval.run_retrieval_eval (reports gate margins).
    kb_min_dense_score: float = 0.40
    # DEGRADED-WINDOW ONLY sparse floor. This is NO LONGER an OR-rescue in normal
    # operation — raw BM25 does not separate scope on a small KB (off-scope
    # filler tokens like siapa/hari/jakarta accumulate BM25: "berita hari ini di
    # jakarta"=8.58 OUTSCORES legit "AmarthaLink"=3.46), so OR'ing sparse
    # re-admitted off-scope. It is consulted ONLY when dense is UNAVAILABLE
    # (embedding outage → sparse-only retrieval, C5): in that window the gate
    # fails OPEN on sparse >= this floor so Ava stays usable during the blip,
    # accepting some off-scope leak until embeddings recover. Kept at 1.0.
    kb_min_sparse_score: float = 1.0

    # ─── Embedding resilience (C5 — query-embedding SPOF) ────────────────────
    # The query embedding is produced by a SINGLE process-global
    # OpenAILikeEmbedding (app/config/embedding_config.py) hitting a remote
    # provider. With no retry/timeout, a transient provider blip turned every
    # retrieval into a hard failure (the dense + sparse encodings are awaited in
    # one gather, so a dense-embed exception aborted sparse too → HTTP 500).
    # These knobs wrap the dense embed in a bounded tenacity retry; on final
    # failure hybrid_search degrades to SPARSE-ONLY (BM25) instead of raising,
    # so terse/lexical queries still return results during an embedding outage.
    embedding_timeout_seconds: float = 8.0
    embedding_max_attempts: int = 3
    embedding_backoff_base_seconds: float = 0.5
    embedding_backoff_max_seconds: float = 4.0

    # ─── How-to / step-by-step formatting ───────────────────────────────────
    # When a KNOWLEDGE query is a how-to/teach-me request ("gimana cara X",
    # "ajarin aku X"), SYSTEM_PROMPT rule 5 tells the model to answer as numbered
    # steps. This caps the step count so the reply stays scannable on mobile
    # (~1 screen on a 13k-user FO phone). (The old float-driven MENTOR mode +
    # learning_context_threshold were removed when the pre-processor was slimmed
    # to intent + rewrite only; step formatting is now intent/prompt-driven.)
    lms_scaffolding_max_steps: int = 5

    # ─── Empathy LLM (temp-0 mode-collapse fix) ──────────────────────────────
    # Gemini Flash Lite at temperature 0.0 is DETERMINISTIC and, on the vent
    # path, collapses onto the prior assistant turn in history — re-emitting it
    # byte-for-byte and ignoring both the new user message AND the per-turn
    # anti-repetition signal (verified via DEBUG_GEN: the signal fires, the model
    # ignores it). Prompt-only variation cannot fix a model that ignores the
    # prompt. So the BRAINSTORM/vent path uses a separate LLM at a non-zero
    # temperature to break determinism and produce genuinely varied, on-topic
    # replies. Selected by intent==BRAINSTORM in _generate_node — KNOWLEDGE turns
    # stay at temp 0.0 for factual/channel fidelity. Configurable so it can be
    # tuned without a code change; 0.6 gives clear variation without going
    # incoherent. (The old empathy_temp_* float gates were removed with the
    # pre-processor slim-down; LLM selection is now intent-driven.)
    empathy_llm_temperature: float = 0.6

    # ─── Cache / Memory ─────────────────────────────────────────────────────
    # Query→answer cache lifetime (Redis exact-match + Qdrant semantic cache).
    # 7d: the KB is near-static (updates ~yearly) so a long TTL maximizes hit
    # rate, but we cap at 7 days NOT a year on purpose: (1) if any KB-update path
    # ever skips flush_cache_by_course, a stale answer's blast radius is bounded
    # to a week, not a year; (2) the semantic cache's real retention is
    # min(TTL, time-to-fill _SEMANTIC_CACHE_MAX_POINTS) — at 13k users the 50k
    # size cap evicts oldest-first long before a year, so a year-long TTL would
    # be fictional unless the cap is raised too. Revisit to 30d + a larger cap
    # once post-launch traffic is understood. Expired semantic-cache points are
    # pruned lazily; mem_limit caps worst-case RAM.
    cache_query_ttl_seconds: int = 604800
    # 12h: must outlive `ltm_afk_threshold_seconds` (10h) so STM history+summary
    # survive long enough for the AFK LTM worker to consume them, with a 2h
    # margin for the worker's Guard-2 defer/retry. Trimmed from 24h: in the
    # happy path the AFK worker's `clear_conversation` DELs the HASH at ~10h
    # anyway, so 24h only ever benefited the orphan-job failure case; 12h bounds
    # worst-case RAM tighter while keeping the ordering invariant intact. Owner
    # now rides this same HASH (B1), so it inherits this lifetime too.
    conversation_ttl_seconds: int = 43200
    user_pref_max_age_days: int = 30  # ignore stored preferences older than this when injecting into prompts
    # Max fresh (un-summarized) conversation turns fed to generate_node. 2
    # (was 3): the rolling summary captures everything older, and the 3rd
    # prior turn is almost always paraphrased in the summary anyway. 2 keeps
    # one prior user/AI pair + current = 3 messages, which is enough for
    # entity binding while saving ~250 input tokens/turn on the typical
    # 4-5 turn session.
    max_fresh_turns: int = 2
    # Per-AI-reply char cap for STM history sent to generate_node. 250
    # (was 400): the pre-processor already summarises AI history at 300 chars
    # in its history-str; 400 compounded across 4 fresh turns = ~1600 chars
    # in generate_node's history section. 250 keeps the gist + entity names
    # for the typical 2-3 turn follow-up while cutting history section size
    # by ~37% (saves ~150 tokens/turn on multi-turn queries).
    max_history_ai_chars: int = 250
    # AFK window before LTM sync fires. 10h matches "user closed laptop / went
    # to sleep" rather than "stepped away for coffee" — short defers waste
    # worker capacity re-summarizing the same session.
    ltm_afk_threshold_seconds: int = 36000

    # ─── Moodle LMS ─────────────────────────────────────────────────────────
    moodle_api_url: str = "https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/"
    moodle_api_token: str = Field(default="", alias="MOODLE_API_TOKEN")

    # ─── Askfer (Portfolio Chat) ────────────────────────────────────────────
    qdrant_personal_collection: str = "Personal_Portfolio"
    portfolio_sitemap_url: str = "https://ferdy-fadhil-lazuardi.my.id/sitemap.xml"
    portfolio_homepage_url: str = "https://ferdy-fadhil-lazuardi.my.id/"
    portfolio_project_url_pattern: str = r"^https://ferdy-fadhil-lazuardi\.my\.id/projects/[^/]+/?$"
    portfolio_cv_url: str = "https://ferdy-fadhil-lazuardi.my.id/CV%20-%20Ferdy%20Fadhil%20Lazuardi.pdf"
    askfer_admin_secret: str = Field(default="", alias="ASKFER_ADMIN_SECRET")
    askfer_rate_limit_per_minute: int = 10
    # Token-budget tuning — Askfer is stateless, so we can keep retrieval lean.
    # Candidate pool per modality before fusion, then narrow to the final cut.
    askfer_retrieval_top_k: int = 12
    askfer_final_top_k: int = 3
    askfer_chunk_text_max_chars: int = 600

    # ─── Security ───────────────────────────────────────────────────────────
    # Required, no default. A dev fallback string ("your-super-secret...") was
    # previously baked into the code, which meant a prod deploy that forgot
    # to set JWT_SECRET booted with a publicly known signing key. Pydantic
    # now refuses to construct Settings() if this is missing.
    jwt_secret: str = Field(..., min_length=32)
    # Literal allowlist so Pydantic rejects `JWT_ALGORITHM=none` (or any
    # other weak/unknown alg) at startup. Previously jwt_algorithm: str
    # silently accepted any value, which let a mis-config set
    # JWT_ALGORITHM=none — pyjwt then validates unsigned tokens.
    jwt_algorithm: Literal["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"] = "HS256"
    rate_limit_per_minute: int = 20
    admin_api_key: str = Field(..., alias="ADMIN_API_KEY", min_length=16)
    # Off by default. Must be explicitly enabled (DEV_BYPASS_ENABLED=true)
    # for the local-dev token-less auth bypass in app/api/auth.py to fire.
    # Closes the prod mis-config class where APP_ENV=development (or a
    # copied-from-dev .env) silently disables auth across all 13k users.
    # main.py:production-guard refuses to boot if this is True in prod.
    dev_bypass_enabled: bool = Field(default=False, alias="DEV_BYPASS_ENABLED")

    # ─── Concurrency / Backpressure ─────────────────────────────────────────
    # Max simultaneous LLM-bound RAG pipeline executions on the single uvicorn
    # worker. Beyond this, requests fast-fail with 503 instead of piling up on
    # the event loop and overwhelming the upstream LLM gateway (any
    # OpenRouter-compatible endpoint). Cache hits are NOT counted — they make
    # no LLM call and return before the guard.
    #
    # 12 (was 24): 24 concurrent LLM calls on 1 uvicorn worker / 2 vCPU over-
    # subscribed the gateway side. The WAF in front of the local self-hosted
    # gateway has been seen to return 429s under sustained >15 concurrent
    # calls. 12 keeps us safely under that ceiling at peak (~5 req/s × ~3s/turn
    # = 15 in-flight, but cache hits skip this guard so the real ceiling is 12
    # LLMs-in-flight). Raise to 18-24 once the gateway's per-account rate limit
    # is confirmed.
    max_concurrent_pipelines: int = 12
    # Seconds a request waits for a free pipeline slot before returning 503.
    # Small enough to fast-fail a sustained burst, large enough to absorb a
    # sub-second spike without rejecting.
    pipeline_acquire_timeout_s: float = 5.0
    # End-to-end ceiling on a NON-STREAM /chat graph run (rag_graph.ainvoke).
    # Without this, a turn makes 2-3 sequential LLM calls and each cheap call
    # is bounded only by request_timeout=30 × stop_after_attempt(3) + backoff
    # ≈ 93s, so a single upstream black-hole could pin a pipeline slot for
    # 3-9 minutes (llm/client.py:136-148). At max_concurrent_pipelines=12 that
    # converts a slow-upstream blip into a full-surface 503 outage. 120s bounds
    # the worst case to ~2 min — generous enough that a normal ~3s turn (or one
    # legitimate retry sequence) never trips it, tight enough to free the slot.
    pipeline_total_timeout_s: float = 120.0
    # STREAM /chat/stream STALL ceiling: max seconds allowed between two
    # astream_events emissions. Resets on EVERY event (including every output
    # token), so a slow-but-actively-streaming answer never trips it — it only
    # fires when the graph goes silent (upstream hang with no tokens). 75s
    # covers a pre-processor call that times out once and retries (30s + 1s +
    # 30s ≈ 61s of legitimate silence before the first generate token) while
    # still bounding a true hang's slot-hold to one stall window instead of
    # minutes. A total wall-clock cap is deliberately NOT used on the stream
    # path so long legitimate answers stream uninterrupted.
    pipeline_stream_stall_timeout_s: float = 75.0

    # ─── Streaq worker hardening ────────────────────────────────────────────
    # Optional HMAC secret for streaq's pickle payload signing. If set, the
    # worker rejects tasks whose pickled payload doesn't match the signature
    # (defense in depth against an attacker with Redis-access forging tasks).
    # Recommended >=32 chars. Generated with:
    #   python -c "import secrets; print(secrets.token_urlsafe(32))"
    # Optional in dev; required-in-prod is enforced at startup (worker.py logs
    # a warning if missing in production, but does not refuse to boot — the
    # hardening is a defense-in-depth measure, not a security boundary).
    streaq_signing_secret: str = Field(default="", alias="STREAQ_SIGNING_SECRET")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
