"""
Observability no-op shim.

Phoenix / OpenInference were removed (Jun 2026). Monitoring now reads
from Postgres (agent_logs) via Streamlit. This module is preserved as
a thin no-op so the 20+ existing call sites in graph/, api/, and
pipeline code keep importing the same names and get back deterministic
no-ops. No openinference / opentelemetry / phoenix libraries are
imported at module load — the only external symbol exposed is
`get_tracer`, which returns the OpenTelemetry SDK's no-op tracer
(`opentelemetry.trace.get_tracer` always returns a working tracer
even when no provider is set, so callers that invoke
`tracer.start_as_current_span` on it stay safe).
"""
from contextlib import contextmanager
from typing import Any, Iterator, Optional


def set_tracer_provider(provider: Any) -> None:
    """No-op. Kept so legacy call sites that import this still work."""


def get_tracer_provider() -> Any:
    """Always returns the OpenTelemetry proxy tracer — never a real provider."""
    from opentelemetry import trace as _otel_trace
    return _otel_trace.get_tracer_provider()


def get_tracer(name: str = "app"):
    """Return a tracer. With no provider set, opentelemetry's SDK gives
    back its built-in NoOp tracer, so any `start_as_current_span` call
    against the returned object is safe and cheap."""
    from opentelemetry import trace as _otel_trace
    return _otel_trace.get_tracer(name)


def set_phoenix_client(client: Any) -> None:
    """No-op. Phoenix is gone; we keep the name so old imports succeed."""


def get_phoenix_client() -> Optional[Any]:
    """Always None. Phoenix is gone — callers that depended on submitting
    span annotations should be rewritten to persist the score to
    agent_logs directly (see app/eval/tasks.py for the pattern)."""
    return None


def is_observability_enabled() -> bool:
    """Always False. Tracing is off; rely on agent_logs (Postgres) for
    observability."""
    return False


def flush() -> None:
    """No-op. Kept so legacy shutdown hooks can call this safely."""


@contextmanager
def trace_attributes(**_kwargs: Any) -> Iterator[None]:
    """No-op context manager. Yields immediately.

    Previously stacked OpenInference `using_session` / `using_user` /
    `using_metadata` / `using_tags` context managers. All are no-ops
    now (the underlying provider is never set).
    """
    yield


def set_current_span_io(*, input: Any = None, output: Any = None) -> None:
    """No-op. The INPUT_VALUE / OUTPUT_VALUE span attributes that
    OpenInference used to consume are not used anywhere — for durable
    persistence, callers should write to agent_logs via batch_logger."""


def set_current_span_attributes(attrs: dict[str, Any]) -> None:
    """No-op. Same rationale as set_current_span_io — durability lives
    in agent_logs, not in OTel span attributes."""


def get_current_span_id() -> Optional[str]:
    """Always None. No active span can exist (no provider)."""
    return None


def update_current_span_name(name: str) -> None:
    """No-op. Phoenix span names are not consumed anywhere."""


def annotate_span(span_id: str, name: str, score: float) -> None:
    """No-op. Phoenix span annotations (the LLM-judge / quality scores)
    have been replaced by direct UPDATEs to agent_logs.faithfulness_score
    (see app/eval/tasks.py)."""


@contextmanager
def root_span(name: str, **_kwargs: Any) -> Iterator[Optional[str]]:
    """No-op root span context manager. Yields None.

    Previously opened a parent span with OpenInference span_kind=AGENT
    so Phoenix UI grouped these correctly. With Phoenix gone, callers
    can drop `with root_span(...)` blocks entirely in a follow-up
    cleanup; for now this keeps the 20+ call sites working.
    """
    yield None
