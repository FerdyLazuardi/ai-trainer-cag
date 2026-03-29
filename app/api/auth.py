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
    If no token is provided, returns a Guest User identity.
    Includes Redis-backed role-based rate limiting.
    """
    user: Optional[User] = None
    
    # ── 1. Attempt JWT Authentication ─────────────────────────────────────
    if credentials:
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
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception:
            # Fallback to guest if token parsing fails unexpectedly
            pass

    # ── 2. Handle Guest / Dev Bypass Mode ─────────────────────────────────
    if not user:
        # For Guests, try to derive a stable user_id from conversation_id if available
        # This helps in maintaining separate rate limits and history per guest session.
        guest_id = str(uuid.uuid4())
        
        # In a real request, we might try to peek at the body for conversation_id
        # But for this simple implementation, we'll use a placeholder or session-based ID.
        user = User(
            user_id=f"guest_{guest_id[:8]}",
            role="guest",
            username="Guest User"
        )
        
        if settings.app_env == "development":
            logger.info("Dev Bypass: Accessing as Guest User")

    # ── 3. Role-Based Rate Limiting ───────────────────────────────────────
    redis = get_redis_client()
    rate_limit_key = f"rate_limit:{user.user_id}"
    
    # Set limit based on role
    limit = settings.rate_limit_per_minute # Default for Moodle users (10)
    if user.role == "guest":
        limit = settings.rate_limit_guest_per_minute # Default for guests (3)
    
    try:
        # Increment request count
        request_count = await redis.incr(rate_limit_key)
        
        # If this is the first request in the window, set expiration (60s)
        if request_count == 1:
            await redis.expire(rate_limit_key, 60)
        
        if request_count > limit:
            logger.warning(f"Rate limit exceeded for {user.user_id} ({user.role})")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Guests are limited to {limit} requests/min.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rate limiting error: {e}")
        # Fail open on rate limiting errors to ensure availability
        pass
        
    return user
