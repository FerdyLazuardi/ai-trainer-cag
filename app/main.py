"""
FastAPI application factory with lifespan context manager.
Initializes all DB connections on startup and tears them down on shutdown.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config.logging import setup_logging
from app.config.settings import get_settings
from app.api.routes import chat, ingest

settings = get_settings()


class _BodyTooLarge(Exception):
    """Internal signal: streamed request body exceeded the cap mid-read."""


class MaxBodySizeMiddleware:
    """Reject oversized request bodies — including chunked requests that OMIT
    Content-Length.
    """

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers") or []:
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await self._send_413(send)
                        return
                except ValueError:
                    pass
                break

        received = 0
        response_started = False

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await self._send_413(send)

    async def _send_413(self, send) -> None:
        body = b'{"detail":"Request body too large."}'
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        })
        await send({"type": "http.response.body", "body": body})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: boot → yield → shutdown."""
    setup_logging(debug=settings.app_debug)
    from app.worker import worker

    async with worker:
        from app.database.postgres import init_db
        from app.database.redis_client import get_redis_client
        from app.utils.logger_batch import batch_logger

        logger.info("Starting application", env=settings.app_env)

        if settings.app_env == "production" and "dev" in settings.jwt_secret.lower():
            raise RuntimeError(
                "JWT_SECRET contains 'dev' — refusing to start. Set a real secret in production."
            )
        if settings.app_env == "production" and settings.dev_bypass_enabled:
            raise RuntimeError(
                "DEV_BYPASS_ENABLED=true under APP_ENV=production — refusing to start. "
                "This would silently disable JWT auth across the entire app. "
                "Unset DEV_BYPASS_ENABLED (or set APP_ENV to a non-production value) and restart."
            )

        if settings.conversation_ttl_seconds <= settings.ltm_afk_threshold_seconds:
            raise RuntimeError(
                f"conversation_ttl_seconds ({settings.conversation_ttl_seconds}s) must be "
                f"> ltm_afk_threshold_seconds ({settings.ltm_afk_threshold_seconds}s) — "
                f"otherwise the AFK LTM worker reads expired STM history. "
                f"Raise STM TTL or lower AFK threshold in settings."
            )

        await batch_logger.start()

        await init_db()
        logger.info("PostgreSQL initialized")

        redis = get_redis_client()
        await redis.ping()

        _info = await redis.info("server")
        _ver = str(_info.get("redis_version", "0"))
        try:
            _major = int(_ver.split(".", 1)[0])
        except (ValueError, IndexError):
            _major = 0
        if _major < 8:
            raise RuntimeError(
                f"Redis {_ver} detected — this app requires Redis >= 8.0 for "
                f"HSETEX/HEXPIRE field-level TTL used by the STM/LTM memory "
                f"layer. On older servers those writes fail silently and "
                f"memory never persists. Upgrade Redis or fix REDIS_HOST."
            )
        logger.info("Redis connection established", redis_version=_ver)

        if settings.eval_enabled:
            from app.llm.client import assert_judge_model_distinct
            assert_judge_model_distinct(
                settings.judge_llm_model,
                {settings.cheap_llm_model, settings.llm_model},
            )
            logger.info(
                "Eval judge model distinct from generator",
                judge=settings.judge_llm_model,
            )

        # Pre-warm active CAG KB pack into memory so Turn 1 does not incur cold DB query
        try:
            from app.graph.pipeline import _filter_kb_by_role, _load_active_cag_kb_text
            kb_text = await _load_active_cag_kb_text()
            if kb_text:
                for r in ("ALL", "BP", "BM", "FO", "HO"):
                    _filter_kb_by_role(kb_text, r)
            logger.info("CAG startup: Pre-warmed active KB pack and role caches", kb_length=len(kb_text) if kb_text else 0)
        except Exception as e:
            logger.warning(f"Failed to pre-warm CAG KB pack: {e}")

        # Pre-warm OpenRouter HTTPS socket keep-alive to eliminate cold TLS handshake on Turn 1
        try:
            from app.llm.client import _shared_http_client
            client = _shared_http_client()
            url = settings.openrouter_base_url.rstrip("/") + "/models"
            headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"} if settings.openrouter_api_key else {}
            resp = await client.get(url, headers=headers, timeout=5.0)
            logger.info("CAG startup: Pre-warmed OpenRouter HTTP connection", status_code=resp.status_code)
        except Exception as e:
            logger.warning(f"OpenRouter pre-warm connection skipped: {e}")

        yield

        logger.info("Shutting down application")
        from app.utils.logger_batch import batch_logger
        await batch_logger.stop()

        await redis.aclose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="CAG LMS Agent",
        description="Cache-augmented LMS assistant powered by LangGraph and OpenRouter prompt caching",
        version="1.0.0",
        docs_url="/docs" if settings.app_debug else None,
        redoc_url="/redoc" if settings.app_debug else None,
        lifespan=lifespan,
    )

    MAX_REQUEST_BYTES = 256 * 1024  # 256 KB
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=MAX_REQUEST_BYTES)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.opt(exception=exc).error(
            "Unhandled exception",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else None,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    from app.api.routes import admin
    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
    app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])

    import os
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", summary="API Status")
    async def root():
        """Backend root response."""
        return {
            "name": "CAG LMS Agent",
            "version": "1.0.0",
            "status": "online",
            "docs": "/docs" if settings.app_debug else "private",
            "test_ui": "/test-ui" if settings.app_debug else "hidden"
        }

    @app.get("/test-ui", summary="Chat UI (Development)", include_in_schema=settings.app_debug)
    async def chat_ui():
        """Serve the simple chat interface for testing (gated on APP_DEBUG)."""
        if not settings.app_debug:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not Found")
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return {"message": "UI files not found."}

    @app.get("/health", summary="Liveness Check (always 200 if process is up)")
    async def health_check() -> dict:
        """Liveness probe — does NOT check downstream services."""
        return {"status": "ok", "env": settings.app_env}

    @app.get("/readyz", summary="Readiness Check (verifies Postgres + Redis)")
    async def readyz() -> JSONResponse:
        """Readiness probe for CAG dependencies."""
        import asyncio as _asyncio
        from fastapi.responses import JSONResponse

        from app.database import postgres as _pg
        from app.database.redis_client import get_redis_client

        async def _check_postgres() -> str:
            try:
                async with _pg.engine.connect() as conn:
                    await _asyncio.wait_for(
                        conn.execute(__import__("sqlalchemy").text("SELECT 1")),
                        timeout=2.0,
                    )
                return "ok"
            except Exception:
                return "down"

        async def _check_redis() -> str:
            try:
                redis = get_redis_client()
                await _asyncio.wait_for(redis.ping(), timeout=2.0)
                return "ok"
            except Exception:
                return "down"

        pg, rd = await _asyncio.gather(
            _check_postgres(), _check_redis(), return_exceptions=True
        )

        def _norm(v) -> str:
            if isinstance(v, Exception):
                logger.warning("readyz component error: {}", v)
                return f"error: {type(v).__name__}"
            return v

        pg, rd = _norm(pg), _norm(rd)
        all_ok = all(s == "ok" for s in (pg, rd))
        return JSONResponse(
            status_code=200 if all_ok else 503,
            content={"status": "ok" if all_ok else "degraded", "postgres": pg, "redis": rd},
        )
    return app


app = create_app()
