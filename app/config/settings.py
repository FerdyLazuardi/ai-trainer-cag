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
    postgres_pool_size: int = 5
    postgres_max_overflow: int = 10

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
    embedding_model: str = "text-embedding-3-small"
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

    # ─── Reranker (local cross-encoder) ─────────────────────────────────────
    reranker_enabled: bool = Field(default=True, alias="RERANKER_ENABLED")
    reranker_model_name: str = Field(
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        alias="RERANKER_MODEL_NAME",
    )
    reranker_device: str = Field(default="cpu", alias="RERANKER_DEVICE")
    reranker_max_length: int = Field(default=512, alias="RERANKER_MAX_LENGTH")

    # ─── Phoenix (Observability — self-hosted) ──────────────────────────────
    phoenix_endpoint: str = Field(default="http://phoenix:6006", alias="PHOENIX_ENDPOINT")
    phoenix_otlp_endpoint: str = Field(default="http://phoenix:4317", alias="PHOENIX_OTLP_ENDPOINT")
    phoenix_project_name: str = Field(default="ai-lms-agent", alias="PHOENIX_PROJECT_NAME")

    # ─── Evaluation (LLM-as-judge, async via arq) ───────────────────────────
    # Phase 1: faithfulness eval only. Sample to control cost — 100% eval
    # at 13k user scale would 2x the LLM bill for diminishing signal.
    # Always-eval gates (low rerank / high empathy) catch high-stakes turns
    # regardless of sample rate so the riskiest outputs are never missed.
    eval_enabled: bool = Field(default=True, alias="EVAL_ENABLED")
    eval_sample_rate: float = Field(default=0.10, alias="EVAL_SAMPLE_RATE")
    eval_always_if_rerank_below: float = Field(default=0.30, alias="EVAL_ALWAYS_IF_RERANK_BELOW")
    eval_always_if_empathy_above: float = Field(default=0.90, alias="EVAL_ALWAYS_IF_EMPATHY_ABOVE")

    # ─── Context Engineering ────────────────────────────────────────────────
    max_context_tokens: int = 6000
    retrieval_top_k: int = 10
    reranked_top_k: int = 7
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    # Below this rerank score, treat retrieval as a miss and skip generate_node
    # entirely (saves ~2700 input tokens + 1 LLM call per off-topic query).
    # Score is the post-sigmoid value in [0, 1] from the cross-encoder.
    rerank_min_score: float = 0.30

    # ─── Cache / Memory ─────────────────────────────────────────────────────
    cache_query_ttl_seconds: int = 14400   # 4 hours — KB content is stable, longer TTL = more cache hits
    # 24h: must outlive `ltm_afk_threshold_seconds` so STM history+summary
    # survive long enough for the AFK LTM worker to consume them.
    conversation_ttl_seconds: int = 86400
    user_pref_max_age_days: int = 30  # ignore stored preferences older than this when injecting into prompts
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
    askfer_retrieval_top_k: int = 12
    askfer_reranked_top_k: int = 3
    askfer_chunk_text_max_chars: int = 600

    # ─── Security ───────────────────────────────────────────────────────────
    jwt_secret: str = "your-super-secret-jwt-key-for-local-dev"
    jwt_algorithm: str = "HS256"
    rate_limit_per_minute: int = 20


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
