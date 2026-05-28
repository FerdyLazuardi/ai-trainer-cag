"""
Phoenix (Arize) self-hosted observability singleton.

Initialize once at app startup via `setup_phoenix()`. All LangChain/LangGraph
calls are traced through OpenInference + OpenTelemetry auto-instrumentation.

Usage:
    from app.observability import get_tracer, get_phoenix_client
    tracer = get_tracer()
    client = get_phoenix_client()  # for span annotations / custom scores
"""
from __future__ import annotations

from typing import Any, Optional

_tracer_provider: Any | None = None
_phoenix_client: Any | None = None


def set_tracer_provider(provider: Any) -> None:
    global _tracer_provider
    _tracer_provider = provider


def get_tracer_provider() -> Any | None:
    return _tracer_provider


def get_tracer(name: str = "app"):
    """Return an OTel tracer bound to the app's provider, or a no-op fallback."""
    from opentelemetry import trace
    if _tracer_provider is not None:
        return _tracer_provider.get_tracer(name)
    return trace.get_tracer(name)


def set_phoenix_client(client: Any) -> None:
    global _phoenix_client
    _phoenix_client = client


def get_phoenix_client() -> Any | None:
    """Phoenix REST client used for span annotations (custom scores).
    Returns None if Phoenix wasn't initialized — callers must guard.
    """
    return _phoenix_client


def is_observability_enabled() -> bool:
    return _tracer_provider is not None


def flush() -> None:
    """Force-flush pending spans on shutdown. Safe to call when disabled."""
    if _tracer_provider is None:
        return
    try:
        _tracer_provider.force_flush()
    except Exception:
        pass


def setup_phoenix(
    *,
    project_name: str,
    otlp_endpoint: str,
    phoenix_endpoint: Optional[str] = None,
) -> bool:
    """Initialize Phoenix tracer + LangChain auto-instrumentation.

    Returns True on success, False on failure (caller logs).
    Idempotent — second call is a no-op.
    """
    global _tracer_provider, _phoenix_client
    if _tracer_provider is not None:
        return True

    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    provider = register(
        project_name=project_name,
        endpoint=otlp_endpoint,
        auto_instrument=False,
        set_global_tracer_provider=True,
    )
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _tracer_provider = provider

    if phoenix_endpoint:
        try:
            from phoenix.client import Client
            _phoenix_client = Client(base_url=phoenix_endpoint)
        except Exception:
            _phoenix_client = None

    return True


from contextlib import contextmanager


@contextmanager
def trace_attributes(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict | None = None,
    tags: list[str] | None = None,
):
    """Stack OpenInference context managers for session/user/metadata/tags.

    Replaces Langfuse's `propagate_attributes(...)`. Safe no-op when
    observability is disabled.
    """
    if _tracer_provider is None:
        yield
        return

    from contextlib import ExitStack
    from openinference.instrumentation import (
        using_session,
        using_user,
        using_metadata,
        using_tags,
    )

    with ExitStack() as stack:
        if session_id:
            stack.enter_context(using_session(session_id))
        if user_id:
            stack.enter_context(using_user(user_id))
        if metadata:
            stack.enter_context(using_metadata(metadata))
        if tags:
            stack.enter_context(using_tags(list(tags)))
        yield


def set_current_span_io(*, input: Any | None = None, output: Any | None = None) -> None:
    """Set INPUT_VALUE / OUTPUT_VALUE on the active span (OpenInference convention)."""
    if _tracer_provider is None:
        return
    try:
        import json as _json
        from opentelemetry import trace as _trace
        from openinference.semconv.trace import SpanAttributes

        span = _trace.get_current_span()
        if span is None or not span.is_recording():
            return
        if input is not None:
            span.set_attribute(
                SpanAttributes.INPUT_VALUE,
                input if isinstance(input, str) else _json.dumps(input, default=str),
            )
        if output is not None:
            span.set_attribute(
                SpanAttributes.OUTPUT_VALUE,
                output if isinstance(output, str) else _json.dumps(output, default=str),
            )
    except Exception:
        pass


def set_current_span_attributes(attrs: dict[str, Any]) -> None:
    """Set arbitrary attributes on the active span.

    Use this for values you want visible directly in Phoenix's main span view
    (no click needed). Span annotations are good for evaluator scores (judge
    faithfulness, retrieval quality), but UI surface attributes belong on the
    span itself — intent classification, scoring axes, latency breakdowns.

    Numbers and strings pass through; nested dicts/lists are JSON-encoded.
    """
    if _tracer_provider is None or not attrs:
        return
    try:
        import json as _json
        from opentelemetry import trace as _trace

        span = _trace.get_current_span()
        if span is None or not span.is_recording():
            return
        for key, value in attrs.items():
            if value is None:
                continue
            if isinstance(value, (str, bool, int, float)):
                span.set_attribute(key, value)
            else:
                span.set_attribute(key, _json.dumps(value, default=str))
    except Exception:
        pass


def get_current_span_id() -> str | None:
    """Return the active span's hex ID, or None when no recording span exists."""
    if _tracer_provider is None:
        return None
    try:
        from opentelemetry import trace as _trace
        span = _trace.get_current_span()
        ctx = span.get_span_context() if span is not None else None
        if ctx is None or not ctx.is_valid:
            return None
        return format(ctx.span_id, "016x")
    except Exception:
        return None


def annotate_span(span_id: str, name: str, score: float) -> None:
    """Submit a numeric score to Phoenix as a span annotation. No-op on failure."""
    if _phoenix_client is None or not span_id:
        return
    try:
        _phoenix_client.spans.add_span_annotation(
            span_id=span_id,
            annotation_name=name,
            annotator_kind="CODE",
            score=float(score),
        )
    except Exception:
        pass


@contextmanager
def root_span(
    name: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict | None = None,
    tags: list[str] | None = None,
):
    """Open a parent span with trace attributes attached. Yields the hex span_id
    (or None when observability is disabled). Auto-instrumented spans created
    inside the block become children of this span. Tags the span with
    OpenInference span_kind=AGENT so Phoenix UI groups these as agents instead
    of "Unknown".
    """
    if _tracer_provider is None:
        yield None
        return
    from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
    from opentelemetry.trace import Status, StatusCode

    tracer = get_tracer(name)
    with trace_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
    ):
        with tracer.start_as_current_span(name) as span:
            try:
                span.set_attribute(
                    SpanAttributes.OPENINFERENCE_SPAN_KIND,
                    OpenInferenceSpanKindValues.AGENT.value,
                )
            except Exception:
                pass
            ctx = span.get_span_context() if span is not None else None
            span_id = format(ctx.span_id, "016x") if ctx and ctx.is_valid else None
            try:
                yield span_id
                # No exception bubbled out → mark span OK so Phoenix UI shows
                # a green checkmark instead of an "unset" strip. Errors raised
                # inside the with-block bypass this and OTel records the
                # exception status automatically.
                try:
                    span.set_status(Status(StatusCode.OK))
                except Exception:
                    pass
            except BaseException:
                raise
