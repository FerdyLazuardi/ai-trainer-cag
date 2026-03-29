"""
JWT Authentication logic for validating tokens issued by Moodle LMS.
Supports Moodle Users, Guest Mode, and Development Bypass.
"""
import jwt
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
    Refactored to ALWAYS fallback to Guest instead of raising 401,
    preventing CORS blocks on cross-origin requests.
    """
    user: Optional[User] = None
    
    # ── 1. Attempt JWT Authentication ─────────────────────────────────────
    if credentials and credentials.credentials:
        token = credentials.credentials
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm]
            )
            user_id: str = str(payload.get("user_id"))
            role: str = payload.get("role", "moodle_user")
            username: str = payload.get("username", "Moodle User")
            
            if user_id:
                user = User(user_id=user_id, role=role, username=username)
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired, falling back to guest")
        except jwt.InvalidTokenError:
            logger.warning("Invalid token, falling back to guest")
        except Exception as e:
            logger.error(f"Unexpected auth error: {e}")

    # ── 2. Handle Guest / Fallback Mode ───────────────────────────────────
    if not user:
        # Generate a semi-stable guest ID for the session if possible
        # For now using a random ID, but identified as Guest.
        guest_id = str(uuid.uuid4())[:8]
        user = User(
            user_id=f"guest_{guest_id}",
            role="guest",
            username="Guest User"
        )
        
        if settings.app_env == "development":
            logger.info(f"Accessing as Guest User: {user.user_id}")

    # ── 3. Role-Based Rate Limiting ───────────────────────────────────────
    # We still keep rate limiting but avoid 401s for CORS stability.
    redis = get_redis_client()
    rate_limit_key = f"rate_limit:{user.user_id}"
    limit = settings.rate_limit_per_minute if user.role != "guest" else settings.rate_limit_guest_per_minute
    
    try:
        request_count = await redis.incr(rate_limit_key)
        if request_count == 1:
            await redis.expire(rate_limit_key, 60)
        
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
        
    return user
