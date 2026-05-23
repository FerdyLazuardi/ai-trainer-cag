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

    Fail-open if Redis errors (mirrors `auth.py` pattern) — abuse would still
    be blocked at infra layer, and we'd rather serve real visitors than 500.
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
        logger.warning(f"Askfer rate limit Redis error (allowing request): {exc}")
    return ip
