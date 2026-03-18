"""
Tenacity retry presets for external API calls and database operations.
"""
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


def retry_on_api_error(func=None, *, max_attempts: int = 3):
    """
    Retry decorator for external API calls (LLM, OpenRouter, etc.).
    Exponential backoff: 2s, 4s, 8s.
    """
    decorator = retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    if func is not None:
        return decorator(func)
    return decorator


def retry_on_db_error(func=None, *, max_attempts: int = 3):
    """
    Retry decorator for database operations.
    Short exponential backoff: 0.5s, 1s, 2s.
    """
    decorator = retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        reraise=True,
    )
    if func is not None:
        return decorator(func)
    return decorator
