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

    # ─── Cohere (Reranking) ─────────────────────────────────────────────────
    cohere_api_key: str = Field(default="", alias="COHERE_API_KEY")

    # ─── Langfuse (Observability) ───────────────────────────────────────────
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = "https://cloud.langfuse.com"

    # ─── Context Engineering ────────────────────────────────────────────────
    max_context_tokens: int = 6000
    retrieval_top_k: int = 15
    reranked_top_k: int = 5
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    # ─── Follow-up validation ───────────────────────────────────────────────
    followup_validation_enabled: bool = True
    followup_validation_threshold: float = 0.5

    # ─── Cache / Memory ─────────────────────────────────────────────────────
    cache_query_ttl_seconds: int = 1800
    conversation_ttl_seconds: int = 3600

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
    askfer_retrieval_top_k: int = 15
    askfer_reranked_top_k: int = 4
    askfer_chunk_text_max_chars: int = 600

    # ─── Security ───────────────────────────────────────────────────────────
    jwt_secret: str = "your-super-secret-jwt-key-for-local-dev"
    jwt_algorithm: str = "HS256"
    rate_limit_per_minute: int = 20


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
