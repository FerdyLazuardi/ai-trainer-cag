"""
Utility functions for user identity validation.
Single source of truth for long-term memory gating.
"""

from app.config.settings import get_settings


def is_real_user(user_id: str, role: str) -> bool:
    """Return True if user is eligible for long-term memory.

    Allows dev_user_123 when dev_bypass_enabled is True in local dev settings.
    """
    if not user_id or not user_id.strip():
        return False
    if user_id.lower() in ("none", "null", "undefined"):
        return False
    settings = get_settings()
    if settings.dev_bypass_enabled:
        return True
    if role != "moodle_user":
        return False
    return True



