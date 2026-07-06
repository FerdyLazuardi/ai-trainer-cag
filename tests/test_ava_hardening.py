import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [
    "Gimana caranya menangani mitra yang telat bayar cicilan?",
    "gimana caranya aku melindungi data mitra ya",
])
async def test_meta_convo_regex_does_not_swallow_how_to_knowledge(monkeypatch, query):
    from app.graph import pipeline

    result = await pipeline._pre_processor(
        {"messages": [HumanMessage(content=query)]},
        {},
    )

    assert result["intent"] == "KNOWLEDGE"


@pytest.mark.asyncio
async def test_meta_convo_regex_keeps_bare_how_to_ambiguous(monkeypatch):
    from app.graph import pipeline

    result = await pipeline._pre_processor(
        {"messages": [HumanMessage(content="gimana caranya?")]},
        {},
    )

    assert result["intent"] == "AMBIGUOUS"


@pytest.mark.asyncio
async def test_example_followup_can_use_knowledge_rewrite_path(monkeypatch):
    from app.graph import pipeline

    result = await pipeline._pre_processor(
        {"messages": [HumanMessage(content="bisa kasih contoh ga")]},
        {},
    )

    assert result["intent"] == "KNOWLEDGE"


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





@pytest.mark.asyncio
async def test_resolve_numeric_query_uses_latest_option_up_to_five():
    from app.agents import conversation_state

    history = [
        {"role": "assistant", "content": "Pilihan lama:\n1. Lama A\n2. Lama B\n3. Lama C\n4. Lama D"},
        {"role": "user", "content": "bahas yang lain"},
        {"role": "assistant", "content": "Pilihan baru:\n1. Baru A\n2. Baru B\n3. Baru C\n4. Baru D"},
    ]

    class FakeRedis:
        async def hget(self, *_args):
            return json.dumps(history)

    resolved = await conversation_state.resolve_numeric_query(FakeRedis(), "4", "conv-1")

    assert resolved == "Baru D"


@pytest.mark.asyncio
async def test_seen_chunk_ids_merge_dedupe_and_cap(monkeypatch):
    from app.agents import conversation_state

    class FakeRedis:
        def __init__(self):
            self.data = {}

        async def hget(self, key, field):
            return self.data.get(key, {}).get(field)

        def pipeline(self, transaction=True):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def hsetex(self, key, mapping, ex):
            self.data.setdefault(key, {}).update(mapping)

        def expire(self, *_args):
            return None

        async def execute(self):
            return None

    fake = FakeRedis()

    await conversation_state.add_seen_chunk_ids(fake, "conv-1", ["a", "b", "a"], max_ids=3)
    await conversation_state.add_seen_chunk_ids(fake, "conv-1", ["c", "d"], max_ids=3)

    assert await conversation_state.get_seen_chunk_ids(fake, "conv-1") == {"b", "c", "d"}


@pytest.mark.asyncio
async def test_prepare_cag_context_skips_rewrite_and_embedding(monkeypatch):
    from app.api.routes import chat
    from app.graph import pipeline

    async def fake_history(*args, **kwargs):
        return "", []

    async def fake_seen(*args, **kwargs):
        return set()

    async def fake_cached(*args, **kwargs):
        return None

    async def fake_refresh(*args, **kwargs):
        return None

    async def fake_rewrite(*args, **kwargs):
        return "rewritten retrieval query"

    monkeypatch.setattr(chat, "get_or_summarize_history", fake_history)
    monkeypatch.setattr(chat, "get_seen_chunk_ids", fake_seen)
    monkeypatch.setattr(chat, "get_cached_response", fake_cached)
    monkeypatch.setattr(chat, "_schedule_summary_refresh", fake_refresh)
    monkeypatch.setattr(chat, "get_cheap_llm", lambda: object())
    monkeypatch.setattr(chat, "is_real_user", lambda **kwargs: False)
    monkeypatch.setattr(pipeline, "_apply_glossary", lambda query: query)

    context = await chat._prepare_cag_context(
        chat.ChatRequest(query="Apa itu Client Protection?"),
        chat.User(user_id="user-a", role="moodle_user", username="User A"),
        "conv-a",
        "Apa itu Client Protection?",
    )

    assert context["query_embedding"] is None
    assert context["initial_state"]["retrieval_query"] == "Apa itu Client Protection?"
    assert context["initial_state"]["rewritten_queries"] is None


def _patch_stream_basics(monkeypatch, graph):
    from app.api.routes import chat
    from app.graph import pipeline

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

    graph_events = None

    async def fake_graph_events():
        nonlocal graph_events
        if graph_events is None:
            graph_events = []
            async for event in graph.astream_events({}, config={}, version="v2"):
                graph_events.append(event)
        return graph_events

    async def fake_pre_processor(state, config):
        for event in await fake_graph_events():
            if event.get("event") == "on_chain_end" and event.get("name") == "pre_processor":
                return event.get("data", {}).get("output", {})
        return {
            "intent": "KNOWLEDGE",
            "rewritten_query": state["messages"][-1].content,
            "intent_scores": {},
            "gate_score": None,
        }

    async def fake_stream_openrouter_generate(state, config=None):
        for event in await fake_graph_events():
            if event.get("event") == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and getattr(chunk, "content", None):
                    yield {"type": "token", "text": chunk.content}
            if event.get("event") == "on_chain_end" and event.get("name") == "generate_node":
                output = event.get("data", {}).get("output", {})
                msgs = output.get("messages") if isinstance(output, dict) else None
                if msgs:
                    msg = msgs[-1]
                    content = getattr(msg, "content", "") or ""
                    if content:
                        yield {"type": "token", "text": content}
                    yield {"type": "usage", "usage": chat.extract_openrouter_usage(msg)}

    monkeypatch.setattr(chat, "acquire_pipeline_slot_or_503", fake_acquire)
    monkeypatch.setattr(chat, "_verify_conversation_ownership", noop_async)
    monkeypatch.setattr(chat, "resolve_numeric_query", fake_resolve_numeric_query)
    monkeypatch.setattr(chat, "_prepare_cag_context", fake_prepare)
    monkeypatch.setattr(chat, "get_cag_graph", lambda: graph)
    monkeypatch.setattr(pipeline, "_pre_processor", fake_pre_processor)
    monkeypatch.setattr(pipeline, "stream_openrouter_generate", fake_stream_openrouter_generate)
    monkeypatch.setattr(chat, "_schedule_afk_ltm_sync", noop_async)
    monkeypatch.setattr(chat, "_track_session_courses", noop_async)
    monkeypatch.setattr(chat, "add_seen_chunk_ids", noop_async)
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
async def test_stream_dedupes_provider_restart_tokens(monkeypatch):
    class Chunk:
        def __init__(self, content):
            self.content = content

    class RestartingStreamGraph:
        def astream_events(self, *args, **kwargs):
            async def gen():
                for token in ("Oke ", "ini ", "Oke ", "ini ", "jawaban final"):
                    yield {
                        "event": "on_chat_model_stream",
                        "metadata": {"langgraph_node": "generate_node"},
                        "data": {"chunk": Chunk(token)},
                    }

            return gen()

    chat = _patch_stream_basics(monkeypatch, RestartingStreamGraph())
    history = []

    async def fake_append(conversation_id, user_message, assistant_message, max_turns=10):
        history.append((conversation_id, user_message, assistant_message))
        return len(history)

    async def fake_log(row):
        return None

    monkeypatch.setattr(chat, "append_to_history", fake_append)
    monkeypatch.setattr(chat.batch_logger, "add_log", fake_log)

    response = await chat.chat_stream(
        chat.ChatRequest(query="jelaskan modal", conversation_id="conv-restart"),
        _DummyRequest(),
        chat.User(user_id="u-1", role="moodle_user", username="User"),
    )

    events = _sse_events(await _collect_sse(response))
    answer = "".join(
        payload["token"]
        for name, payload in events
        if name == "message" and payload and "token" in payload
    )

    assert answer == "Oke ini jawaban final"
    assert history == [("conv-restart", "jelaskan modal", "Oke ini jawaban final")]


@pytest.mark.asyncio
async def test_stream_log_fetches_generation_cost_when_usage_cost_missing(monkeypatch):
    class EndOnlyGraph:
        def astream_events(self, *args, **kwargs):
            async def gen():
                yield {
                    "event": "on_chain_end",
                    "name": "generate_node",
                    "data": {
                        "output": {
                            "messages": [AIMessage(
                                content="jawaban",
                                response_metadata={
                                    "id": "gen-cost",
                                    "model_name": "model-x",
                                    "token_usage": {
                                        "prompt_tokens": 10,
                                        "completion_tokens": 2,
                                        "total_tokens": 12,
                                    },
                                },
                            )],
                        },
                    },
                }

            return gen()

    chat = _patch_stream_basics(monkeypatch, EndOnlyGraph())
    logs = []

    async def fake_log(row):
        logs.append(row)

    async def fake_cost(generation_id):
        assert generation_id == "gen-cost"
        return chat.OpenRouterUsage(
            prompt_tokens=10,
            cached_tokens=3,
            completion_tokens=2,
            provider="provider-x",
            cost=0.00042,
            generation_id=generation_id,
        )

    monkeypatch.setattr(chat.batch_logger, "add_log", fake_log)
    monkeypatch.setattr(chat, "fetch_openrouter_generation_usage", fake_cost)

    response = await chat.chat_stream(
        chat.ChatRequest(query="jelaskan modal", conversation_id="conv-cost"),
        _DummyRequest(),
        chat.User(user_id="u-1", role="moodle_user", username="User"),
    )

    await _collect_sse(response)

    assert logs[0]["or_generation_id"] == "gen-cost"
    assert logs[0]["or_cost"] == 0.00042
    assert logs[0]["or_provider"] == "provider-x"


@pytest.mark.asyncio
async def test_stream_history_keeps_original_user_question_when_query_is_rewritten(monkeypatch):
    class Chunk:
        def __init__(self, content):
            self.content = content

    class RewritingStreamGraph:
        def astream_events(self, *args, **kwargs):
            async def gen():
                yield {
                    "event": "on_chain_end",
                    "name": "pre_processor",
                    "data": {
                        "output": {
                            "intent": "KNOWLEDGE",
                            "rewritten_query": "melindungi data mitra",
                            "intent_scores": {},
                            "gate_score": None,
                        }
                    },
                }
                yield {
                    "event": "on_chat_model_stream",
                    "metadata": {"langgraph_node": "generate_node"},
                    "data": {"chunk": Chunk("AI answer")},
                }

            return gen()

    chat = _patch_stream_basics(monkeypatch, RewritingStreamGraph())
    history = []

    async def fake_append(conversation_id, user_message, assistant_message, max_turns=10):
        history.append((conversation_id, user_message, assistant_message))
        return len(history)

    async def fake_log(row):
        return None

    monkeypatch.setattr(chat, "append_to_history", fake_append)
    monkeypatch.setattr(chat.batch_logger, "add_log", fake_log)

    response = await chat.chat_stream(
        chat.ChatRequest(
            query="gimana caranya aku melindungi data mitra ya",
            conversation_id="conv-rewrite",
        ),
        _DummyRequest(),
        chat.User(user_id="u-1", role="moodle_user", username="User"),
    )

    await _collect_sse(response)

    assert history == [(
        "conv-rewrite",
        "gimana caranya aku melindungi data mitra ya",
        "AI answer",
    )]


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
    history_writes = []

    async def fake_log(row):
        logs.append(row)

    async def fake_cache_write(*args, **kwargs):
        cache_writes.append((args, kwargs))

    async def fake_append(*args, **kwargs):
        history_writes.append((args, kwargs))
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

    assert "CAG pipeline failed" in body
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "chat-stream"
    assert logs[0]["answer"] == ""
    assert cache_writes == []
    assert history_writes == []


def test_sanitize_answer_strips_offscope():
    from app.graph.pipeline import _sanitize_answer
    text = "Maaf, itu di luar kapasitas saya. [OFFSCOPE]"
    cleaned = _sanitize_answer(text)
    assert "[OFFSCOPE]" not in cleaned
    assert "di luar kapasitas saya" in cleaned


@pytest.mark.asyncio
async def test_user_violation_and_ban_logic(monkeypatch):
    import app.api.routes.chat as chat

    # Mock Redis client
    mock_redis_data = {}
    class DummyRedis:
        async def get(self, key):
            return mock_redis_data.get(key)
        async def incr(self, key):
            val = mock_redis_data.get(key, 0) + 1
            mock_redis_data[key] = val
            return val
        async def expire(self, key, ttl, **kwargs):
            pass
        async def setex(self, key, ttl, val):
            mock_redis_data[key] = val
        async def ttl(self, key):
            # Return positive seconds if key exists (simulating a banned user), else -2
            return 7200 if key in mock_redis_data else -2

    def mock_get_redis_client():
        return DummyRedis()

    monkeypatch.setattr(chat, "get_redis_client", mock_get_redis_client)

    # Test _get_ban_ttl (not banned yet)
    assert await chat._get_ban_ttl("user-1") == 0

    # Test _handle_off_scope_violation increments
    new_count, just_banned = await chat._handle_off_scope_violation("user-1")
    assert new_count == 1
    assert not just_banned
    assert await chat._get_ban_ttl("user-1") == 0

    # 2nd violation
    new_count, just_banned = await chat._handle_off_scope_violation("user-1")
    assert new_count == 2
    assert not just_banned
    assert await chat._get_ban_ttl("user-1") == 0

    # 3rd violation (Warning threshold)
    new_count, just_banned = await chat._handle_off_scope_violation("user-1")
    assert new_count == 3
    assert not just_banned
    assert await chat._get_ban_ttl("user-1") == 0

    # 4th violation (Ban threshold) — now user IS banned
    new_count, just_banned = await chat._handle_off_scope_violation("user-1")
    assert new_count == 4
    assert just_banned
    assert await chat._get_ban_ttl("user-1") > 0


@pytest.mark.asyncio
async def test_chat_short_circuits_if_banned(monkeypatch):
    import app.api.routes.chat as chat

    # Mock Redis so user is banned (ttl returns 7200 seconds)
    class DummyRedisBanned:
        async def ttl(self, key):
            if "banned" in key:
                return 7200
            return -2
        async def get(self, key):
            return None

    monkeypatch.setattr(chat, "get_redis_client", lambda: DummyRedisBanned())

    # Call _run_chat
    req = chat.ChatRequest(query="resep rendang", conversation_id="conv-1")
    bg_tasks = chat.BackgroundTasks()
    user = chat.User(user_id="user-banned", role="moodle_user", username="Test")

    # We patch ownership verification so it passes
    async def fake_verify(*args):
        pass
    monkeypatch.setattr(chat, "_verify_conversation_ownership", fake_verify)

    res = await chat._run_chat(req, bg_tasks, user)
    assert "Ai Trainer dinonaktifkan sementara" in res.answer
    assert res.ban_remaining_seconds == 7200
    assert res.cached is False
    assert res.latency_ms >= 0


@pytest.mark.asyncio
async def test_chat_ban_status_exposes_current_ttl(monkeypatch):
    import app.api.routes.chat as chat

    class DummyRedisBanned:
        async def ttl(self, key):
            assert key == "ava:user:user-banned:banned"
            return 321

    monkeypatch.setattr(chat, "get_redis_client", lambda: DummyRedisBanned())

    user = chat.User(user_id="user-banned", role="moodle_user", username="Test")
    assert await chat.chat_ban_status(user) == {"ban_remaining_seconds": 321}


def test_sanitize_answer_strips_course_num():
    from app.graph.pipeline import _sanitize_answer
    text = "Bisa cek materi di Course 3: Tentang Amartha ya atau di Course 1 Tentang Credit Risk."
    cleaned = _sanitize_answer(text)
    assert "Course 3" not in cleaned
    assert "Course 1" not in cleaned
    assert "Tentang Amartha" in cleaned
    assert "Tentang Credit Risk" in cleaned
