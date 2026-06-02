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
    app_env: Literal["development", "staging", "production"] = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ─── PostgreSQL ─────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "lms_ai"
    postgres_user: str = "admin"
    postgres_password: str = "admin"
    # Pool sizing: base 10 + overflow 20 = up to 30 concurrent connections.
    # Sized so a burst of in-flight requests doesn't exhaust the pool now that
    # streaming endpoints no longer hold a connection for the SSE lifetime.
    # Postgres 16 default max_connections=100 leaves ample headroom.
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

    @computed_field  # type: ignore[misc]
    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ─── Qdrant ─────────────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"
    qdrant_kb_collection: str = "Knowledge_Base"

    # ─── Embedding ──────────────────────────────────────────────────────────
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536

    # ─── LLM (OpenRouter) ───────────────────────────────────────────────────
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_embedding_key: str = Field(default="", alias="OPENROUTER_EMBEDDING_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_embedding_url: str | None = None
    ollama_base_url: str = "http://172.16.10.2:11434/v1"
    llm_model: str = "google/gemini-2.5-flash"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048

    # ─── Phoenix (Observability — self-hosted) ──────────────────────────────
    phoenix_endpoint: str = Field(default="http://phoenix:6006", alias="PHOENIX_ENDPOINT")
    phoenix_otlp_endpoint: str = Field(default="http://phoenix:4317", alias="PHOENIX_OTLP_ENDPOINT")
    phoenix_project_name: str = Field(default="ai-lms-agent", alias="PHOENIX_PROJECT_NAME")

    # ─── Evaluation (LLM-as-judge, async via arq) ───────────────────────────
    # Phase 1: faithfulness eval only. Sample to control cost — 100% eval
    # at 13k user scale would 2x the LLM bill for diminishing signal.
    # Always-eval gates (low dense relevance / high empathy) catch high-stakes
    # turns regardless of sample rate so the riskiest outputs are never missed.
    eval_enabled: bool = Field(default=True, alias="EVAL_ENABLED")
    eval_sample_rate: float = Field(default=0.10, alias="EVAL_SAMPLE_RATE")
    eval_always_if_dense_below: float = Field(default=0.30, alias="EVAL_ALWAYS_IF_DENSE_BELOW")
    eval_always_if_empathy_above: float = Field(default=0.90, alias="EVAL_ALWAYS_IF_EMPATHY_ABOVE")

    # ─── Context Engineering ────────────────────────────────────────────────
    max_context_tokens: int = 6000
    # Candidate pool pulled per modality (dense + sparse BM25) before fusion.
    # 20 (was 10): at ~42 KB chunks 10 already covered ~25% of the corpus, but
    # the KB is scaling toward ~300+ chunks where a narrow pool would miss
    # relevant hits before fusion. Widening the pool costs ~nothing — it does
    # NOT add embedding calls or LLM tokens (only final_top_k reaches the LLM).
    retrieval_top_k: int = 20
    # Final number of fused chunks fed to the generate LLM. 4 (was 5/6):
    # the safety + harassment cases need ≤ 3 chunks; a hot-fix on the
    # multi-hop KB queries showed 4 chunks gives the same answer as 6 for
    # 90% of queries. Dropping from 5 to 4 saves ~275 input tokens/turn
    # (≈18% on the chunk portion). The 1 chunk that gets dropped is always
    # the lowest-fused score — i.e. the least relevant. If multi-hop quality
    # regresses, raise to 5. This is the only knob that adds LLM input
    # tokens — keep it ≤ 8.
    final_top_k: int = 4
    # Per-chunk char cap for the LMS generate path. 900 (was 1100): measured
    # chunk sizes after Q4-2025 ingestion average ~850 chars (median 720),
    # so 900 only trims the long tail (top decile). 1100 was wasting ~150
    # tokens/turn on the few outliers that were at the cap. Budget: 4 ×
    # ~225 tok ≈ 900 tok, well under max_context_tokens.
    lms_chunk_text_max_chars: int = 900
    # Relative-score fusion weights: fused = vector_weight * dense_norm +
    # bm25_weight * sparse_norm. vector_weight is the `alpha` in hybrid_search.
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    # Below this RAW DENSE COSINE top-1 score, treat retrieval as a miss and
    # skip generate_node entirely (saves ~2700 input tokens + 1 LLM call per
    # off-topic query). Absolute [0, 1] cosine — calibrated from production:
    # answered turns median ≈ 0.68 vs not-found ≈ 0.45; 0.30 blocks only the
    # weakest misses while sparing virtually all valid answers.
    kb_min_dense_score: float = 0.30
    # Lexical-match rescue for the NOT-FOUND gate. Terse entity queries
    # ("Modal", "CP", "AmarthaLink") score LOW on dense cosine but have a strong
    # exact BM25 match (raw score ≫ 0), so the gate ALSO passes when raw BM25
    # top-score clears this floor — rescuing real KB entities dense alone would
    # wrongly reject.
    #
    # IMPORTANT — this is a LEAKY backstop, not the primary off-scope gate.
    # Calibration (scripts/calibrate_sparse_gate.py, measured against the live
    # KB) shows BM25 does NOT cleanly separate scope:
    #   - legit entities: sparse 3.46 .. 13.12
    #   - off-scope:      sparse 0.00 ..  8.58  (e.g. "berita hari ini di
    #     jakarta"=8.58, "harga bitcoin sekarang"=7.08) — common Indonesian
    #     filler tokens (siapa/kapan/hari/jakarta/harga) have KB IDF overlap, so
    #     "off-scope = 0.0" only holds for ZERO-overlap phrases ("resep nasi
    #     goreng"=0.0). Dense overlaps too ("berita..."=0.30 ≥ "Modal"=0.24).
    # There is NO single sparse floor that admits all entities while blocking
    # all off-scope — the OFF_SCOPE intent classifier in _pre_processor is the
    # real scope gate; this floor only catches misclassified-as-KNOWLEDGE turns,
    # and imperfectly. Kept at 1.0 (admits entities, blocks only zero-overlap
    # junk) on purpose: raising it would kill low-scoring entities like
    # AmarthaLink (3.46) without blocking the 6-8 scoring off-scope phrases.
    kb_min_sparse_score: float = 1.0

    # ─── Cache / Memory ─────────────────────────────────────────────────────
    # Query→answer cache lifetime (Redis exact-match + Qdrant semantic cache).
    # 24h: KB is stable and Moodle sync auto-invalidates affected entries, so a
    # long TTL maximizes hit rate (fewer LLM calls). Expired semantic-cache
    # points are pruned lazily; mem_limit caps worst-case RAM.
    cache_query_ttl_seconds: int = 86400
    # 24h: must outlive `ltm_afk_threshold_seconds` so STM history+summary
    # survive long enough for the AFK LTM worker to consume them.
    conversation_ttl_seconds: int = 86400
    user_pref_max_age_days: int = 30  # ignore stored preferences older than this when injecting into prompts
    # Max fresh (un-summarized) conversation turns fed to generate_node. 2
    # (was 3): the rolling summary captures everything older, and the 3rd
    # prior turn is almost always paraphrased in the summary anyway. 2 keeps
    # one prior user/AI pair + current = 3 messages, which is enough for
    # entity binding while saving ~250 input tokens/turn on the typical
    # 4-5 turn session.
    max_fresh_turns: int = 2
    # Per-AI-reply char cap for STM history sent to generate_node. The
    # pre-processor already caps AI history at 400 chars; generate_node fed
    # full replies, which compounds across turns. 400 keeps entity names +
    # the gist while bounding the worst case (was 500, trimmed 20%).
    max_history_ai_chars: int = 400
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
    jwt_algorithm: str = "HS256"
    rate_limit_per_minute: int = 20

    # ─── Concurrency / Backpressure ─────────────────────────────────────────
    # Max simultaneous LLM-bound RAG pipeline executions on the single uvicorn
    # worker. Beyond this, requests fast-fail with 503 instead of piling up on
    # the event loop and overwhelming the upstream LLM gateway (OpenRouter).
    # Cache hits are NOT counted — they make no LLM call and return before the
    # guard. Sized for ~2 LLM calls/turn: at peak ~5 req/s × ~3s/turn ≈ 15 in
    # flight, so 24 leaves headroom. Tune once OpenRouter's real per-account
    # rate limits are known.
    max_concurrent_pipelines: int = 24
    # Seconds a request waits for a free pipeline slot before returning 503.
    # Small enough to fast-fail a sustained burst, large enough to absorb a
    # sub-second spike without rejecting.
    pipeline_acquire_timeout_s: float = 5.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
