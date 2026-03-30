"""
Utility functions for user identity validation.
Single source of truth for long-term memory gating.
"""

def is_real_user(user_id: str, role: str) -> bool:
    """
    Return True only if this user is an authenticated Moodle user
    eligible for long-term memory storage.

    Blocked cases:
    - Empty or whitespace-only user_id
    - Literal "None" / "null" / "undefined" (from str() conversion bug)
    - user_id starting with "guest_" (fallback guest users)
    - Role is NOT "moodle_user"
    """
    if not user_id or not user_id.strip():
        return False
    if role != "moodle_user":
        return False
    if user_id.startswith("guest_"):
        return False
    if user_id.lower() in ("none", "null", "undefined"):
        return False
    return True

def ltm_load_redis_key(user_id: str) -> str:
    """Redis key to mark that LTM was already loaded for this user in this session."""
    return f"rag:ltm:loaded:{user_id}"
