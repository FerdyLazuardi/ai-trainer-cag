"""
Public-facing dependencies for Askfer endpoints.

`rate_limit_by_ip` is intentionally separate from `app.api.auth.get_current_user`
so the Askfer route is fully decoupled from JWT auth flow used by A-Pedi.
"""
from fastapi import HTTPException, Request, status
from loguru import logger

from app.config.settings import get_settings
from app.database.redis_client import get_redis_client

settings = get_settings()


def _client_ip(request: Request) -> str:
    """Resolve the real client IP, honoring X-Forwarded-For first hop."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit_by_ip(request: Request) -> str:
    """Public endpoint guard: rate-limit by client IP. No auth.

    Fail-closed in production, fail-open in development (same policy
    as `app.api.auth.get_current_user`):

      - Production: a Redis outage means the rate limiter is
        untrustworthy. Returning 503 lets the LB shed the instance
        via /readyz, and an attacker who can DoS Redis can't bypass
        the rate limit.
      - Development: a 30s Redis blip would otherwise brick the
        local iteration loop. The dev env is single-instance, so a
        hard 503 would be the worst of both worlds (no healthy
        peers but the user is still logged out).
    """
    ip = _client_ip(request)
    redis = get_redis_client()
    key = f"askfer:rate:{ip}"
    limit = settings.askfer_rate_limit_per_minute

    try:
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 60)
        results = await pipe.execute()
        count = results[0]
        if count > limit:
            logger.warning(f"Askfer rate limit exceeded for {ip}: {count}/{limit}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Limit is {limit}/min.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Askfer rate limit Redis error: {exc}")
        if settings.app_env == "production":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiter unavailable. Please retry shortly.",
                headers={"Retry-After": "5"},
            ) from exc
        else:
            logger.warning("Askfer rate limit Redis error (dev: allowing request)")
    return ip


async def rate_limit_sync_by_ip(request: Request) -> str:
    """Tighter rate limit for /askfer/sync (admin re-sync trigger).

    Gated on `X-Admin-Secret`, not JWT — so a compromised secret
    bypasses user-level limits. Each call enqueues a full
    re-scrape (homepage + projects + CV) which can OOM Qdrant
    if fired in a loop. Cap at 5 calls per 60s per IP. The admin
    workflow needs ~1 call per refresh, so 5/min is plenty of
    headroom.

    Same fail-closed-in-prod / fail-open-in-dev policy as
    `rate_limit_by_ip`.
    """
    ip = _client_ip(request)
    redis = get_redis_client()
    key = f"askfer:sync:rate:{ip}"
    limit = 5
    window = 60

    try:
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        results = await pipe.execute()
        count = results[0]
        if count > limit:
            logger.warning(f"Askfer sync rate limit exceeded for {ip}: {count}/{limit}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Askfer sync rate limit exceeded. Limit is {limit}/{window}s.",
                headers={"Retry-After": str(window)},
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Askfer sync rate limit Redis error: {exc}")
        if settings.app_env == "production":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiter unavailable. Please retry shortly.",
                headers={"Retry-After": "5"},
            ) from exc
    return ip
