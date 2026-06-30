import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.asyncio
async def test_preprocessor_regex_hit_skips_semantic_gate(monkeypatch):
    from app.graph import intent_classifier, pipeline

    called = False

    async def fake_semantic_gate(*args, **kwargs):
        nonlocal called
        called = True
        return intent_classifier.GateScore(
            decision="SKIP",
            committed=None,
            best_intent=None,
            best_cosine=0.0,
            second_intent=None,
            second_cosine=0.0,
            margin=0.0,
        )

    monkeypatch.setattr(pipeline._settings, "intent_semantic_gate_enabled", True)
    monkeypatch.setattr(intent_classifier, "classify_semantic_with_scores", fake_semantic_gate)

    result = await pipeline._pre_processor(
        {"messages": [HumanMessage(content="halo")]},
        {},
    )

    assert result["intent"] == "GREETING"
    assert result["gate_score"].decision == "SKIP"
    assert called is False


@pytest.mark.asyncio
async def test_flush_cache_by_course_deletes_global_keys(monkeypatch):
    from app.utils import cache

    class FakeRedis:
        def __init__(self):
            self.matches = []

        async def scan(self, cursor, match, count):
            self.matches.append(match)
            return 0, []

        async def unlink(self, *keys):
            raise AssertionError("unlink should not run when scan returns no keys")

    fake = FakeRedis()
    monkeypatch.setattr(cache, "get_redis_client", lambda: fake)

    await cache.flush_cache_by_course(42)

    assert "rag:cache:42:*" in fake.matches
    assert "rag_user_*:cache:42:*" in fake.matches
    assert "rag:cache:global:*" in fake.matches
    assert "rag_user_*:cache:global:*" in fake.matches
    assert "rag:cache:None:*" not in fake.matches
    assert "rag_user_*:cache:None:*" not in fake.matches


class _DummyRequest:
    async def is_disconnected(self):
        return False


async def _collect_sse(response):
    chunks = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        chunks.append(chunk)
    return "".join(chunks)


def _sse_events(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        if not block:
            continue
        name = "message"
        payload = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                payload = json.loads(line.removeprefix("data: "))
        events.append((name, payload))
    return events


def test_sanitize_answer_strips_directive_only_leak():
    from app.graph.pipeline import _sanitize_answer

    assert _sanitize_answer(
        "Irrelevant with the user question: None of the items cover Syariah link."
    ) == "Maaf, ada kendala merangkum jawaban. Coba tanya ulang ya."


def test_sanitize_answer_strips_course_context_dump():
    from app.graph.pipeline import _sanitize_answer

    raw = (
        "of [] nd\n\n"
        "1. Tokoh dan Capaian Utama\n\n"
        "> **[Meta-Context]** This is module metadata.\n\n"
        "2. Course: Welcome to Amartha (ID:) Profile, Vision-Mission\n"
        "Chunk text that should not reach the user.\n\n"
        "**Visi Amartha:** Kemakmuran Bersama."
    )

    cleaned = _sanitize_answer(raw)

    assert "Meta-Context" not in cleaned
    assert "Course:" not in cleaned
    assert "Chunk text" not in cleaned
    assert cleaned == "**Visi Amartha:** Kemakmuran Bersama."


def _patch_stream_basics(monkeypatch, graph):
    from app.api.routes import chat

    async def fake_acquire():
        return lambda: None

    async def noop_async(*args, **kwargs):
        return None

    async def fake_prepare(request, current_user, conversation_id, resolved_query):
        return {
            "cached": None,
            "query_embedding": None,
            "was_personalized": False,
            "skip_cache": False,
            "initial_state": {"messages": [HumanMessage(content=resolved_query)]},
        }

    async def fake_resolve_numeric_query(query, conversation_id):
        return query

    monkeypatch.setattr(chat, "acquire_pipeline_slot_or_503", fake_acquire)
    monkeypatch.setattr(chat, "_verify_conversation_ownership", noop_async)
    monkeypatch.setattr(chat, "resolve_numeric_query", fake_resolve_numeric_query)
    monkeypatch.setattr(chat, "_prepare_rag_context", fake_prepare)
    monkeypatch.setattr(chat, "get_rag_graph", lambda: graph)
    monkeypatch.setattr(chat, "_schedule_afk_ltm_sync", noop_async)
    monkeypatch.setattr(chat, "_track_session_courses", noop_async)
    monkeypatch.setattr(chat, "set_cached_response", noop_async)
    return chat


@pytest.mark.asyncio
async def test_stream_empty_answer_fallback_persists_final_history(monkeypatch):
    class EmptyThenFallbackGraph:
        def astream_events(self, *args, **kwargs):
            async def gen():
                if False:
                    yield {}

            return gen()

        async def ainvoke(self, *args, **kwargs):
            return {
                "messages": [AIMessage(content="fallback answer")],
                "retrieved_context": [],
            }

    chat = _patch_stream_basics(monkeypatch, EmptyThenFallbackGraph())
    history = []

    async def fake_append(conversation_id, user_message, assistant_message, max_turns=10):
        history.append((conversation_id, user_message, assistant_message))
        return len(history)

    async def fake_log(row):
        return None

    monkeypatch.setattr(chat, "append_to_history", fake_append)
    monkeypatch.setattr(chat.batch_logger, "add_log", fake_log)

    response = await chat.chat_stream(
        chat.ChatRequest(query="apa itu modal", conversation_id="conv-1"),
        _DummyRequest(),
        chat.User(user_id="u-1", role="moodle_user", username="User"),
    )

    body = await _collect_sse(response)
    events = _sse_events(body)

    assert ("message", {"token": "fallback answer"}) in events
    assert any(name == "done" for name, _payload in events)
    assert history == [("conv-1", "apa itu modal", "fallback answer")]


@pytest.mark.asyncio
async def test_stream_error_path_logs_and_does_not_write_cache(monkeypatch):
    class ErrorGraph:
        def astream_events(self, *args, **kwargs):
            async def gen():
                raise RuntimeError("provider down")
                if False:
                    yield {}

            return gen()

    chat = _patch_stream_basics(monkeypatch, ErrorGraph())
    logs = []
    cache_writes = []

    async def fake_log(row):
        logs.append(row)

    async def fake_cache_write(*args, **kwargs):
        cache_writes.append((args, kwargs))

    async def fake_append(*args, **kwargs):
        return 1

    monkeypatch.setattr(chat.batch_logger, "add_log", fake_log)
    monkeypatch.setattr(chat, "set_cached_response", fake_cache_write)
    monkeypatch.setattr(chat, "append_to_history", fake_append)

    response = await chat.chat_stream(
        chat.ChatRequest(query="jelaskan modal", conversation_id="conv-2"),
        _DummyRequest(),
        chat.User(user_id="u-1", role="moodle_user", username="User"),
    )

    body = await _collect_sse(response)

    assert "RAG pipeline failed" in body
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "chat-stream"
    assert logs[0]["answer"] == ""
    assert cache_writes == []
