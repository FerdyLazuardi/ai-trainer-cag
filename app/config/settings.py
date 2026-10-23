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
    app_env: Literal["development", "staging", "production"]
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ─── CORS ──────────────────────────────────────────────────────────────
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: [
            "https://ai-trainer.lifeatamartha.com",
            "https://lifeatamartha.com",
            "https://academy.amartha.com",
            "http://localhost:3000",
            "http://localhost:4321",
            "http://localhost:8000",
            "http://localhost:8001",
        ]
    )

    # ─── PostgreSQL ─────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5455
    postgres_db: str = "lms_ai"
    postgres_user: str = "admin"
    postgres_password: str = Field(...)
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20
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
    redis_port: int = 6391
    redis_db: int = 0
    redis_password: str = ""
    redis_max_connections: int = 100
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
        isolated queue DB (`redis_queue_db`)."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_queue_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_queue_db}"

    # ─── LLM (OpenRouter) ───────────────────────────────────────────────────
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1/"
    llm_model: str = Field(default="deepseek/deepseek-v4-flash", alias="LLM_MODEL")
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    cheap_llm_model: str = Field(
        default="inclusionai/ling-2.6-flash", alias="CHEAP_LLM_MODEL"
    )
    llm_provider_order: str = Field(default="", alias="LLM_PROVIDER_ORDER")

    # ─── CAG mode ───────────────────────────────────────────────────────────
    retrieval_mode: Literal["cag"] = Field(default="cag", alias="RETRIEVAL_MODE")
    cag_kb_source: str = Field(default="moodle", alias="CAG_KB_SOURCE")
    cag_moodle_course_id: int = Field(default=3, alias="CAG_MOODLE_COURSE_ID")
    spreadsheet_sync_url: str = Field(default="", alias="SPREADSHEET_SYNC_URL")
    spreadsheet_sync_token: str = Field(default="amartha_secret_kpi_token_2026", alias="SPREADSHEET_SYNC_TOKEN")
    course_spreadsheet_url: str = Field(default="", alias="COURSE_SPREADSHEET_URL")
    course_spreadsheet_token: str = Field(default="", alias="COURSE_SPREADSHEET_TOKEN")

    @property
    def llm_provider_order_list(self) -> list[str]:
        """Parse LLM_PROVIDER_ORDER into a clean slug list (order preserved)."""
        return [s.strip() for s in self.llm_provider_order.split(",") if s.strip()]

    # ─── LLM-as-judge (eval faithfulness — disabled by default) ──────────────
    judge_llm_model: str = Field(
        default="deepseek/deepseek-v4-pro", alias="JUDGE_LLM_MODEL"
    )
    judge_llm_fallback_model: str = Field(
        default="qwen/qwen-2.5-72b-instruct", alias="JUDGE_LLM_FALLBACK_MODEL"
    )
    eval_enabled: bool = Field(default=False, alias="EVAL_ENABLED")
    eval_sample_rate: float = Field(default=0.10, alias="EVAL_SAMPLE_RATE")
    eval_always_if_dense_below: float = Field(default=0.30, alias="EVAL_ALWAYS_IF_DENSE_BELOW")
    eval_always_if_empathy_above: float = Field(default=0.90, alias="EVAL_ALWAYS_IF_EMPATHY_ABOVE")
    faithfulness_min: float = Field(default=0.75, alias="FAITHFULNESS_MIN")

    # Topic-streak auto-hook: offer coaching once the user has asked about the
    # SAME dominant topic this many turns in a row.
    coaching_streak_threshold: int = 3

    # ─── Conversational LLM Temperatures ─────────────────────────────────────
    chat_llm_temperature: float = 0.2
    empathy_llm_temperature: float = 0.3
    cheap_llm_temperature: float = 0.0
    judge_llm_temperature: float = 0.0
    preprocessor_llm_temperature: float = 0.0
    generate_llm_temperature: float = 0.0

    # ─── Cache / Memory ─────────────────────────────────────────────────────
    cache_query_ttl_seconds: int = 604800
    conversation_ttl_seconds: int = 43200
    user_pref_max_age_days: int = 30
    agent_log_retention_days: int = Field(
        default=90, ge=7, le=365, alias="AGENT_LOG_RETENTION_DAYS"
    )
    max_fresh_turns: int = 8
    max_history_ai_chars: int = 600
    # Long-Term Memory (LTM) toggle — off by default until target LTM model is finalized.
    ltm_enabled: bool = Field(default=False, alias="LTM_ENABLED")
    ltm_afk_threshold_seconds: int = 36000

    # ─── Moodle LMS ─────────────────────────────────────────────────────────
    moodle_api_url: str = "https://amarthapedia-staging.lifeatamartha.com/"
    moodle_api_token: str = Field(default="", alias="MOODLE_API_TOKEN")

    # ─── Security ───────────────────────────────────────────────────────────
    jwt_secret: str = Field(..., min_length=32)
    jwt_algorithm: Literal["HS256", "HS384", "HS512", "RS256", "RS384", "RS512"] = "HS256"
    rate_limit_per_minute: int = 10
    admin_api_key: str = Field(..., alias="ADMIN_API_KEY", min_length=16)
    dev_bypass_enabled: bool = Field(default=False, alias="DEV_BYPASS_ENABLED")

    # ─── Concurrency / Backpressure ─────────────────────────────────────────
    max_concurrent_pipelines: int = 8
    pipeline_acquire_timeout_s: float = 5.0
    pipeline_total_timeout_s: float = 120.0
    pipeline_stream_stall_timeout_s: float = 75.0

    # ─── Streaq worker hardening ────────────────────────────────────────────
    streaq_signing_secret: str = Field(default="", alias="STREAQ_SIGNING_SECRET")

    # ─── Off-Scope Violation & Banning ───────────────────────────────────────
    max_off_scope_violations: int = 4
    warning_off_scope_violations: int = 3
    off_scope_violation_window_seconds: int = 86400
    off_scope_ban_duration_seconds: int = 7200


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
