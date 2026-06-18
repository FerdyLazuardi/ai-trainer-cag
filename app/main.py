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
from app.api.routes import chat, ingest, askfer
from app.worker import worker  # shared streaq Worker instance for status_by_id from API

settings = get_settings()


class _BodyTooLarge(Exception):
    """Internal signal: streamed request body exceeded the cap mid-read."""


class MaxBodySizeMiddleware:
    """Reject oversized request bodies — including chunked requests that OMIT
    Content-Length, which the previous header-only check let through.

    The Content-Length header is a fast pre-filter (normal clients send it).
    For a streaming/chunked body with no Content-Length, we count bytes as they
    arrive and abort with 413 the instant the cap is crossed — so an attacker
    can't OOM the single-worker API by trickling a multi-GB body past a missing
    header. Pure ASGI (not BaseHTTPMiddleware) so the byte-counting wrapper sees
    every `http.request` chunk regardless of how the body is framed.
    """

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Fast path: a present, oversized Content-Length is rejected before we
        # read a single body byte.
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
            # Only safe to emit a response if the app hasn't already started one.
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
    """Application lifespan: boot → yield → shutdown.

    Opens the streaq Worker async-context for the entire app lifetime so that
    API handlers can call `worker.status_by_id()` / `worker.result_by_id()`
    against the shared Redis pool without raising `StreaqError: Worker not
    initialized` on every request. The custom `_worker_lifespan` in
    `app/worker.py` only does `setup_logging` + `ensure_llamaindex_configured`
    on startup, both safe to run in the API process.
    """
    setup_logging(debug=settings.app_debug)

    async with worker:
        from app.database.postgres import init_db
        from app.database.redis_client import get_redis_client
        from app.database.qdrant_client import get_qdrant_client
        from app.utils.logger_batch import batch_logger

        logger.info("Starting application", env=settings.app_env)

        # Production-only guard: refuse to boot if the JWT secret looks like a
        # dev placeholder. Catches the "forgot to set env in prod" failure mode
        # where the entire /api/v1/chat* surface becomes forgeable.
        if settings.app_env == "production" and "dev" in settings.jwt_secret.lower():
            raise RuntimeError(
                "JWT_SECRET contains 'dev' — refusing to start. Set a real secret in production."
            )
        # Production-only guard: refuse to boot if the dev-bypass flag is
        # enabled in prod. DEV_BYPASS_ENABLED off-by-default in settings
        # closes the "APP_ENV=development in a prod .env" mis-config class
        # where the auth dep at app/api/auth.py:40 returns a synthetic
        # user with no token. Setting the flag in a prod .env would be
        # an explicit (loud, fatal) operator action — this check makes
        # that loud and fatal at boot, not silent at request time.
        if settings.app_env == "production" and settings.dev_bypass_enabled:
            raise RuntimeError(
                "DEV_BYPASS_ENABLED=true under APP_ENV=production — refusing to start. "
                "This would silently disable JWT auth across the entire app. "
                "Unset DEV_BYPASS_ENABLED (or set APP_ENV to a non-production value) and restart."
            )

        # Audit 3.5: STM TTL must outlive LTM AFK window, or the AFK-LTM worker
        # tries to read history that has already expired — silent memory loss.
        # The 2h margin (43200s STM vs 36000s AFK) is what lets the AFK worker
        # tolerate a retry/defer cycle. Bumping STM down without bumping AFK
        # is the realistic mistake (someone tuning TTL down for RAM); this
        # guard catches it at boot instead of in production drift.
        if settings.conversation_ttl_seconds <= settings.ltm_afk_threshold_seconds:
            raise RuntimeError(
                f"conversation_ttl_seconds ({settings.conversation_ttl_seconds}s) must be "
                f"> ltm_afk_threshold_seconds ({settings.ltm_afk_threshold_seconds}s) — "
                f"otherwise the AFK LTM worker reads expired STM history. "
                f"Raise STM TTL or lower AFK threshold in settings."
            )

        # Start BatchLogger background task
        await batch_logger.start()

        # Initialize PostgreSQL tables
        await init_db()
        logger.info("PostgreSQL initialized")

        # Initialize Redis connection
        redis = get_redis_client()
        await redis.ping()

        # Fail-fast: the STM/LTM layer relies on HSETEX/HEXPIRE (field-level
        # TTL on conversation + memory HASHes), which only exist in Redis
        # >= 8.0. On an older server those write paths fail SILENTLY — the
        # command errors are swallowed and memory simply never persists,
        # so a mis-pointed REDIS_HOST or a half-finished migration would
        # degrade memory with no loud signal. Assert the version at boot
        # so that class of failure is caught here, not in production drift.
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

        # Fail-fast (C3): the eval judge MUST be a different model family than
        # the generator. When judge == generator they share fabrication
        # patterns and the faithfulness eval undercounts the hallucination
        # rate (judge grading itself). The generator slots are
        # cheap_llm_model (get_generate_llm) and llm_model (the old judge
        # slot, get_llm) — both Gemini Flash Lite. JUDGE_LLM_MODEL must not
        # collide with either, or eval scores silently inflate in production.
        from app.llm.client import assert_judge_model_distinct
        assert_judge_model_distinct(
            settings.judge_llm_model,
            {settings.cheap_llm_model, settings.llm_model},
        )
        logger.info(
            "Eval judge model distinct from generator",
            judge=settings.judge_llm_model,
        )

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

        # Initialize Personal_Portfolio collection (Askfer)
        await qdrant.ensure_personal_collection()
        logger.info("Qdrant Personal_Portfolio collection ready", collection=settings.qdrant_personal_collection)

        # Pre-warm the embedding model config in the API process too.
        # NOTE: entering `async with worker:` above runs Worker.__aenter__,
        # which opens the coredis connection but does NOT run the worker's
        # `_worker_lifespan` — that only fires in streaq's run_async() consume
        # path (streaq/worker.py:597 enters `self.lifespan` separately). So the
        # API process never pre-warmed embeddings; the first chat request paid
        # the one-time LlamaIndex Settings.embed_model init inside
        # _prepare_rag_context. Do it here at boot. Idempotent (module-level
        # _initialized flag), so the lazy call at chat.py becomes a no-op.
        from app.config.embedding_config import ensure_llamaindex_configured
        ensure_llamaindex_configured()
        logger.info("Embedding model config pre-warmed (API process)")

        yield

        # Shutdown
        logger.info("Shutting down application")
        from app.utils.logger_batch import batch_logger
        await batch_logger.stop()

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

    # ─── Request body size cap ─────────────────────────────────────────────
    # Reject oversized bodies before they reach the JSON parser. A 50 MB POST
    # to /ingest would otherwise OOM the single-worker API during the body read,
    # long before Pydantic's max_length on IngestRequest.text (200_000 chars)
    # could fire. 256 KB is a small overhead above the 200 KB text cap to cover
    # JSON envelope fields. MaxBodySizeMiddleware also counts streamed bytes, so
    # a chunked request with no Content-Length can't slip past the header check.
    MAX_REQUEST_BYTES = 256 * 1024  # 256 KB
    app.add_middleware(MaxBodySizeMiddleware, max_bytes=MAX_REQUEST_BYTES)

    # ─── CORS ───────────────────────────────────────────────────────────────
    # ngrok-free-domain invariants — DO NOT add any of the following without
    # re-checking the Moodle plugin tunnel flow (settings.moodle_api_url):
    #   1. Host-header validation: ngrok rotates subdomains, so any allowlist
    #      tied to a specific Host header breaks the tunnel the moment the
    #      free-tier subdomain changes.
    #   2. IP allowlist: ngrok's exit IPs rotate within a wide NAT range;
    #      allowlists block the tunnel intermittently.
    #   3. Per-IP rate limit: same — shared ngrok NAT IPs would hammer the
    #      limit and lock out legitimate Moodle traffic. Global `rate_limit`
    #      in settings.py is fine; per-IP is not.
    #   4. CSP / X-Frame-Options / Referrer-Policy: ngrok's first-visit
    #      interstitial is a same-origin page; aggressive CSP would block
    #      the click-through. Leave defaults.
    #   5. Origin reflection (allow_origin_regex=".*"): CSRF exfil vector.
    #      The explicit allowlist below is the correct posture — Moodle POSTs
    #      are server-to-server (no Origin header) so they never trip CORS
    #      anyway; only the dashboard frontend hits CORS preflight.
    #
    # The default allowlist below is localhost-only (dev). Production must
    # set ALLOWED_ORIGINS env to the real frontend origins (dashboard,
    # Askfer preview). Tunnel traffic (ngrok → api) is NOT affected by CORS
    # because the Moodle plugin uses a server-side HTTP client, not a
    # browser — CORS is a browser-only concern.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ─── Global exception handler ──────────────────────────────────────────
    # Without this, an unhandled exception anywhere in a route surfaces as
    # Starlette's bare 500 with no structured-log hook, so the failure never
    # lands in logs/app.json and ops can't see it. This catch-all logs the
    # exception with request context (loguru captures the traceback) and
    # returns an opaque 500 — it deliberately does NOT leak the exception
    # string to the client. HTTPException is handled by FastAPI's own handler
    # and never reaches here, so intended 4xx/403/413 responses are unaffected.
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

    # ─── Routes ─────────────────────────────────────────────────────────────
    from app.api.routes import admin
    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
    app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
    app.include_router(askfer.router, prefix="/api/v1", tags=["Askfer"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])

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

    @app.get("/test-ui", summary="Chat UI (Development)", include_in_schema=settings.app_debug)
    async def chat_ui():
        """Serve the simple chat interface for testing.

        Gated on APP_DEBUG. In production this route returns 404 even
        though it's registered — the audit flagged it as a dev
        playground that any internet visitor can hit to fire
        arbitrary queries at the LLM/Qdrant (cost amplification +
        prompt-cache-miss provocation).
        """
        if not settings.app_debug:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not Found")
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return {"message": "UI files not found."}

    @app.get("/askfer-ui", summary="Askfer Portfolio Chat UI (Development)", include_in_schema=settings.app_debug)
    async def askfer_ui():
        """Serve the Askfer dev preview chat interface. Same gating as /test-ui."""
        if not settings.app_debug:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not Found")
        askfer_path = os.path.join(static_dir, "askfer.html")
        if os.path.exists(askfer_path):
            with open(askfer_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return {"message": "Askfer UI files not found."}
        
    @app.get("/health", summary="Liveness Check (always 200 if process is up)")
    async def health_check() -> dict:
        """Liveness probe — does NOT check downstream services.

        Use /readyz for readiness. K8s/Proxmox should hit /health as the
        liveness probe (so the container is not killed for transient
        downstream hiccups) and /readyz for routing (so traffic is shed
        when Postgres/Qdrant/Redis are unreachable).
        """
        return {"status": "ok", "env": settings.app_env}

    @app.get("/readyz", summary="Readiness Check (verifies Postgres + Qdrant + Redis)")
    async def readyz() -> JSONResponse:
        """Readiness probe — pings every downstream the API depends on.

        Returns 200 with a per-component status dict when all three
        backends are reachable within a 2s timeout each; returns 503
        (with the same dict, one of whose values is "down") if any one
        is unreachable. Load balancers should pull this instance out
        of rotation when 503, without killing the process (that's
        /health's job).
        """
        import asyncio as _asyncio
        from fastapi.responses import JSONResponse

        from app.database import postgres as _pg
        from app.database.qdrant_client import get_qdrant_client
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

        async def _check_qdrant() -> str:
            try:
                client = get_qdrant_client()
                # QdrantManager wraps AsyncQdrantClient; .client exposes the
                # raw async client with a .get_collections() coroutine.
                await _asyncio.wait_for(client.client.get_collections(), timeout=2.0)
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

        pg, qd, rd = await _asyncio.gather(
            _check_postgres(), _check_qdrant(), _check_redis(),
            return_exceptions=True,
        )
        # If a check raised instead of returning "ok"/"down", surface the
        # error in the response so operators can see WHY it's down.
        def _norm(v) -> str:
            if isinstance(v, Exception):
                logger.warning("readyz component error: {}", v)
                return f"error: {type(v).__name__}"
            return v
        pg, qd, rd = _norm(pg), _norm(qd), _norm(rd)
        all_ok = all(s == "ok" for s in (pg, qd, rd))
        return JSONResponse(
            status_code=200 if all_ok else 503,
            content={"status": "ok" if all_ok else "degraded", "postgres": pg, "qdrant": qd, "redis": rd},
        )

    return app


app = create_app()