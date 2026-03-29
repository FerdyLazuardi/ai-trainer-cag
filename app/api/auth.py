"""
JWT Authentication logic for validating tokens issued by Moodle LMS.
"""
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from app.config.settings import get_settings
from app.database.redis_client import get_redis_client

settings = get_settings()
security = HTTPBearer()

class User(BaseModel):
    user_id: str
    role: str
    username: str

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    Dependency to validate JWT token and return the current user.
    Includes Redis-backed rate limiting per user.
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm]
        )
        user_id: str = str(payload.get("user_id"))
        role: str = payload.get("role")
        username: str = payload.get("username")
        
        if user_id is None or role is None or username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: missing user info",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # ─── Rate Limiting ──────────────────────────────────────────────────
        redis = get_redis_client()
        rate_limit_key = f"rate_limit:{user_id}"
        
        # Increment request count
        request_count = await redis.incr(rate_limit_key)
        
        # If this is the first request in the window, set expiration
        if request_count == 1:
            await redis.expire(rate_limit_key, 60)
        
        if request_count > settings.rate_limit_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
            )
        
        return User(user_id=user_id, role=role, username=username)
        
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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
