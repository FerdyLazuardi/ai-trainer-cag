"""
JWT Authentication logic for validating tokens issued by Moodle LMS.
"""
import jwt
import time
import uuid
from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from loguru import logger

from app.config.settings import get_settings
from app.database.redis_client import get_redis_client

settings = get_settings()
# auto_error=False is crucial to prevent automatic 403/401 from FastAPI
security = HTTPBearer(auto_error=False)

class User(BaseModel):
    user_id: str
    role: str
    username: str

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """
    Dependency to validate JWT token and return the current user.
    """
    user: Optional[User] = None
    
    # ── 1. Attempt JWT Authentication ─────────────────────────────────────
    if not credentials or not credentials.credentials:
        # Development Bypass
        if settings.app_env == "development":
            logger.info("Development bypass active: Authenticating as Dev User")
            user = User(user_id="dev_user_123", role="moodle_user", username="Dev User")
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    if not user:
        token = credentials.credentials

        # Brute-force throttle: count failed JWT decodes per client IP
        # in 60s buckets. Above the threshold, refuse before we even
        # try to decode — saves CPU + closes the "spray random tokens
        # at the prod host" amplification path. Key is keyed on IP
        # (not user_id) because the attacker is guessing against
        # unknown users. The same Redis client the rate limiter uses,
        # so a Redis outage still fails closed (H-6).
        await _check_brute_force_throttle(request)

        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm]
            )
            raw_user_id = payload.get("user_id")
            user_id: str = str(raw_user_id) if raw_user_id is not None else ""

            role: str = payload.get("role", "moodle_user")
            username: str = payload.get("username", "Moodle User")

            if not user_id or not user_id.strip():
                raise ValueError("Invalid user_id in token payload")

            user = User(user_id=user_id, role=role, username=username)
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, ValueError, Exception) as e:
            # Single uniform 401 response regardless of why the token
            # was rejected. The previous code returned "Token
            # expired" / "Invalid token" / "Authentication failed"
            # as distinct messages, which let an attacker enumerate
            # which users have an active session. The new
            # behavior is the OWASP-recommended opaque 401.
            logger.warning(f"Auth rejected: {type(e).__name__}: {e}")
            await _record_auth_failure(request)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # ── 2. Role-Based Rate Limiting ───────────────────────────────────────
    # Skip rate limiting for the dev bypass user — when APP_ENV=development we
    # want unfettered ability to load-test, replay golden eval queries, and
    # iterate on prompts without hitting 429s. Real Moodle users still rate-
    # limited normally.
    if user.user_id == "dev_user_123" or settings.app_env == "development":
        return user

    redis = get_redis_client()
    rate_limit_key = f"rate_limit:{user.user_id}"
    limit = settings.rate_limit_per_minute

    try:
        # Atomic pipeline: incr + expire in a single round-trip to avoid TOCTOU
        pipe = redis.pipeline()
        pipe.incr(rate_limit_key)
        pipe.expire(rate_limit_key, 60)
        results = await pipe.execute()
        request_count = results[0]

        if request_count > limit:
            logger.warning(f"Rate limit exceeded for {user.user_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Limit is {limit}/min.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rate limiting error: {e}")
        # Fail-closed in production: a Redis outage means the rate
        # limiter is untrustworthy, so the safest default is to refuse
        # the request and let /readyz shed the instance from the load
        # balancer. Operators get paged via the 503, the LB pulls us
        # out, and users on the still-healthy instance (if any)
        # continue normally.
        #
        # Fail-open in development: 30s Redis blips would otherwise
        # brick the local iteration loop. The dev environment is
        # single-instance, so a hard 503 would be the worst of both
        # worlds (no healthy peers, but logged out anyway).
        if settings.app_env == "production":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiter unavailable. Please retry shortly.",
                headers={"Retry-After": "5"},
            ) from e

    return user


# ── Brute-force throttle (H-7) ───────────────────────────────────────────────
# Sliding-window counter in Redis, keyed on client IP, 60s buckets.
# Above 30 failed decodes per minute from the same IP, refuse with
# 429 before we even try to decode. Threshold is generous (real users
# don't see 30+ failed logins) but the kind of ceiling a token-sprayer
# will hit in a few seconds.
_BRUTE_FORCE_WINDOW_SECONDS = 60
_BRUTE_FORCE_MAX_FAILURES = 30


def _client_ip(request: Request) -> str:
    """Resolve the real client IP, honoring X-Forwarded-For first hop.

    Same logic as `app.api.askfer_deps._client_ip`. Inlined here to
    avoid a circular import.
    """
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _check_brute_force_throttle(request: Request) -> None:
    """If this IP has too many recent auth failures, 429 the request.

    Reads (does NOT increment) the failure counter. The counter is
    incremented by `_record_auth_failure()` from the except block
    when jwt.decode raises. A single INCR per failed request keeps
    the threshold semantics clean: 30 failed requests → 31st gets
    429. The key rotates every 60s (via the suffix), so the counter
    self-expires. We don't EXPIRE explicitly — a stale bucket just
    sits in Redis until idle eviction.
    """
    ip = _client_ip(request)
    bucket = int(time.time()) // _BRUTE_FORCE_WINDOW_SECONDS
    key = f"bf:auth:{ip}:{bucket}"
    try:
        redis = get_redis_client()
        n_raw = await redis.get(key)
        n = int(n_raw) if n_raw is not None else 0
        if n > _BRUTE_FORCE_MAX_FAILURES:
            logger.warning(f"Brute-force throttle engaged for IP {ip} (n={n})")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed authentication attempts. Please retry shortly.",
                headers={"Retry-After": "30"},
            )
    except HTTPException:
        raise
    except Exception as e:
        # Redis outage: fail open here (the rate limiter in the
        # caller is the durable fail-closed gate; this is a
        # best-effort short-circuit for known-suspicious IPs).
        logger.warning(f"Brute-force throttle check failed (allowing): {e}")


async def _record_auth_failure(request: Request) -> None:
    """Increment the same counter _check_brute_force_throttle reads from.

    Called from the except block when jwt.decode raises. Best-effort —
    if Redis is down the throttle check itself will fail open.
    """
    ip = _client_ip(request)
    bucket = int(time.time()) // _BRUTE_FORCE_WINDOW_SECONDS
    key = f"bf:auth:{ip}:{bucket}"
    try:
        redis = get_redis_client()
        await redis.incr(key)
    except Exception as e:
        logger.warning(f"Failed to record auth failure (non-fatal): {e}")
