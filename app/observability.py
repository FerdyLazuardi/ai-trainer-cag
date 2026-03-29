"""
Langfuse v4 client singleton.

Initialize once at app startup. All LangChain/LangGraph calls
are traced via OpenTelemetry auto-instrumentation.

Import this module from anywhere to get the shared client.
"""

_langfuse_client = None


def set_langfuse_client(client) -> None:
    global _langfuse_client
    _langfuse_client = client


def get_langfuse_client():
    """Return the initialized Langfuse singleton, or None if not configured."""
    return _langfuse_client
