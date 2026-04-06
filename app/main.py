"""
FastAPI application factory with lifespan context manager.
Initializes all DB connections on startup and tears them down on shutdown.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.api.routes import chat, ingest
from app.api.auth import get_current_user
from app.observability import set_langfuse_client, get_langfuse_client

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: boot → yield → shutdown."""
    setup_logging(debug=settings.app_debug)

    from app.database.postgres import init_db
    from app.database.redis_client import get_redis_client
    from app.database.qdrant_client import get_qdrant_client
    from app.utils.logger_batch import batch_logger

    logger.info("Starting application", env=settings.app_env)

    # Start BatchLogger background task
    await batch_logger.start()

    # ── Langfuse v4 Observability (MUST init first, before any LLM calls) ──
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        try:
            import os
            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
            os.environ["LANGFUSE_HOST"] = settings.langfuse_host

            from langfuse import Langfuse
            lf = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
                debug=settings.app_debug,
            )
            set_langfuse_client(lf)
            logger.info("Langfuse v4 initialized (OTEL auto-instrumentation active)")
        except Exception as e:
            logger.warning(f"Langfuse init failed (tracing disabled): {e}")
    else:
        logger.warning("Langfuse keys not set — tracing disabled")

    # Initialize PostgreSQL tables
    await init_db()
    logger.info("PostgreSQL initialized")

    # Initialize Redis connection
    redis = get_redis_client()
    await redis.ping()
    logger.info("Redis connection established")

    import asyncio
    # Wait for Qdrant to be ready (docker race condition fix)
    qdrant = get_qdrant_client()
    qdrant_ready = False
    for _ in range(15):
        try:
            await qdrant.ensure_collection()
            qdrant_ready = True
            break
        except Exception as e:
            logger.warning(f"Waiting for Qdrant... ({e})")
            await asyncio.sleep(2)
            
    if not qdrant_ready:
        logger.error("Failed to connect to Qdrant after 30 seconds.")
        raise RuntimeError("Qdrant connection failed.")
        
    logger.info("Qdrant collection ready", collection=settings.qdrant_collection)

    # Initialize Knowledge_Base collection for Moodle ingestion
    await qdrant.ensure_kb_collection()
    logger.info("Qdrant Knowledge_Base collection ready", collection=settings.qdrant_kb_collection)

    # Initialize Long-Term Memory (LTM) collection in Qdrant
    await qdrant.ensure_ltm_collection()
    logger.info("Qdrant LTM collection ready (user_ltm_memories)")

    yield

    # Shutdown
    logger.info("Shutting down application")
    from app.utils.logger_batch import batch_logger
    await batch_logger.stop()

    if get_langfuse_client():
        get_langfuse_client().flush()
    await redis.aclose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI LMS RAG Agent",
        description="Production-grade Hybrid RAG system powered by LangGraph",
        version="1.0.0",
        docs_url="/docs" if settings.app_debug else None,
        redoc_url="/redoc" if settings.app_debug else None,
        lifespan=lifespan,
    )

    # ─── CORS ───────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Routes ─────────────────────────────────────────────────────────────
    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
    app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])

    # Mount static files correctly
    import os
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", summary="API Status")
    async def root():
        """Backend root response."""
        return {
            "name": "AI LMS RAG Agent",
            "version": "1.0.0",
            "status": "online",
            "docs": "/docs" if settings.app_debug else "private",
            "test_ui": "/test-ui" if settings.app_debug else "hidden"
        }

    @app.get("/test-ui", summary="Chat UI (Development)")
    async def chat_ui():
        """Serve the simple chat interface for testing."""
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return {"message": "UI files not found."}
        
    @app.get("/health", summary="Health Check")
    async def health_check() -> dict:
        return {"status": "ok", "env": settings.app_env}

    return app


app = create_app()