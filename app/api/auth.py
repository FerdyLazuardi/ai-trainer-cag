"""
JWT Authentication logic for validating tokens issued by Moodle LMS.
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
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        except Exception as e:
            logger.error(f"Unexpected auth error: {e}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed")

    # ── 2. Role-Based Rate Limiting ───────────────────────────────────────
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
        
    return user
