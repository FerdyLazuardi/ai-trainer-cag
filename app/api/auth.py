"""
JWT Authentication logic for validating tokens issued by Moodle LMS.
"""
import jwt
import time
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
    fullname: str = ""
    dept: str = ""
    location: str = ""
    position: str = ""
    grade: str = ""
    point: str = ""
    gender: str = ""
    area: str = ""
    regional: str = ""


def _extract_fullname_from_session_id(session_id: str, user_id: str) -> str:
    """Fallback helper to extract user full name from session_id if fullname is absent from JWT payload."""
    if not session_id or not user_id:
        return ""
    prefix = f"{user_id}_"
    if not session_id.startswith(prefix):
        return ""
    rem = session_id[len(prefix):]
    parts = rem.split("_")
    if not parts or parts[0] == "user":
        return ""

    name_tokens = []
    for p in parts:
        if p.lower() in ("ho", "fo", "na") or p.isdigit():
            break
        name_tokens.append(p.capitalize())
    return " ".join(name_tokens)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    """
    Dependency to validate JWT token and return the current user.
    """
    user: Optional[User] = None

    # ── 1. Attempt JWT Authentication ─────────────────────────────────────
    if not credentials or not credentials.credentials:
        if settings.app_env == "development" and settings.dev_bypass_enabled:
            loc = request.headers.get("x-mock-location", "FO")
            pos = request.headers.get("x-mock-position", "Business Manager")
            gender = request.headers.get("x-mock-gender", "Male")
            grade = request.headers.get("x-mock-grade", "BM - Junior")
            dept = request.headers.get("x-mock-dept", "Micro Business - Lending")
            point = request.headers.get("x-mock-point", "Tutuyan")
            area = request.headers.get("x-mock-area", "Minahasa")
            reg = request.headers.get("x-mock-regional", "Sulawesi Utara")

            mock_user = request.headers.get("x-mock-username") or request.query_params.get("mock_username") or "mock_user"
            mock_fullname = request.headers.get("x-mock-fullname") or mock_user
            logger.info(
                f"Development bypass active: username={mock_user}, loc={loc}, pos={pos}, gender={gender}, "
                f"grade={grade}, dept={dept}, point={point}, area={area}, reg={reg}"
            )
            user = User(
                user_id="dev_user_123",
                role="moodle_user",
                username=mock_user,
                fullname=mock_fullname,
                dept=dept,
                location=loc,
                position=pos,
                grade=grade,
                point=point,
                gender=gender,
                area=area,
                regional=reg,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    if not user:
        assert credentials is not None
        token = credentials.credentials

        # Verify HMAC-SHA256 JWT locally first (<0.05ms CPU) so valid requests
        # skip the Redis brute-force ZSET round-trip completely.
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm],
                options={"require": ["exp", "iat", "user_id"]},
            )
            raw_user_id = payload.get("user_id")
            user_id: str = str(raw_user_id) if raw_user_id is not None else ""

            role: str = payload.get("role", "moodle_user")
            username: str = payload.get("username", "Moodle User")

            raw_fullname = payload.get("fullname") or payload.get("full_name") or payload.get("name")
            fullname: str = str(raw_fullname).strip() if raw_fullname else _extract_fullname_from_session_id(
                str(payload.get("session_id") or ""), user_id
            )

            if not user_id or not user_id.strip():
                raise ValueError("Invalid user_id in token payload")

            user = User(
                user_id=user_id,
                role=role,
                username=username,
                fullname=fullname,
                dept=payload.get("dept", "") or "",
                location=payload.get("location", "") or "",
                position=payload.get("position", "") or "",
                grade=payload.get("grade", "") or "",
                point=payload.get("point", "") or "",
                gender=payload.get("gender", "") or "",
                area=payload.get("area", "") or "",
                regional=payload.get("regional", "") or "",
            )
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, ValueError, Exception) as e:
            logger.warning(f"Auth rejected: {type(e).__name__}: {e}")
            await _check_brute_force_throttle(request)
            await _record_auth_failure(request)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # ── 2. Role-Based Rate Limiting ───────────────────────────────────────
    if settings.app_env == "development" and settings.dev_bypass_enabled:
        return user

    redis = get_redis_client()
    rate_limit_key = f"rate_limit:{user.user_id}"
    limit = settings.rate_limit_per_minute

    try:
        pipe = redis.pipeline()
        pipe.incr(rate_limit_key)
        pipe.expire(rate_limit_key, 60, nx=True)
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
        if settings.app_env == "production":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiter unavailable. Please retry shortly.",
                headers={"Retry-After": "5"},
            ) from e

    return user


async def get_admin_or_jwt_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    """
    Allow authentication via X-API-Key (admin portal / worker / scripts) OR
    Moodle JWT Bearer token.
    """
    import secrets
    api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if api_key and settings.admin_api_key and secrets.compare_digest(api_key, settings.admin_api_key):
        return User(
            user_id="admin",
            role="admin",
            username="admin",
            fullname="System Administrator",
        )
    return await get_current_user(request, credentials)


# ── Brute-force throttle (H-7) ───────────────────────────────────────────────
_BRUTE_FORCE_WINDOW_SECONDS = 60
_BRUTE_FORCE_MAX_FAILURES = 30


def _client_ip(request: Request) -> str:
    """Resolve the real client IP, honoring X-Forwarded-For first hop."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def _check_brute_force_throttle(request: Request) -> None:
    """If this IP has too many recent auth failures, 429 the request."""
    ip = _client_ip(request)
    key = f"bf:auth:sw:{ip}"
    now = time.time()
    window_start = now - _BRUTE_FORCE_WINDOW_SECONDS
    try:
        redis = get_redis_client()
        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, "-inf", window_start)
        pipe.zcard(key)
        results = await pipe.execute()
        n = results[1]
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
        logger.warning(f"Brute-force throttle check failed (allowing): {e}")


async def _record_auth_failure(request: Request) -> None:
    """Add a timestamped entry to the sliding-window ZSET for this IP."""
    ip = _client_ip(request)
    key = f"bf:auth:sw:{ip}"
    now = time.time()
    try:
        redis = get_redis_client()
        pipe = redis.pipeline()
        member = f"{now}:{id(request)}"
        pipe.zadd(key, {member: now})
        pipe.expire(key, _BRUTE_FORCE_WINDOW_SECONDS * 2)
        await pipe.execute()
    except Exception as e:
        logger.warning(f"Failed to record auth failure (non-fatal): {e}")
