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


def test_filter_seen_chunks_drops_repeats_but_keeps_fresh_context():
    from app.graph.pipeline import _filter_seen_chunks

    chunks = [
        {"chunk_id": "old-1", "text": "old"},
        {"chunk_id": "new-1", "text": "new"},
        {"text": "no id"},
    ]

    assert _filter_seen_chunks(chunks, {"old-1"}) == [
        {"chunk_id": "new-1", "text": "new"},
        {"text": "no id"},
    ]


def test_filter_seen_chunks_keeps_top_chunk_when_everything_repeated():
    from app.graph.pipeline import _filter_seen_chunks

    chunks = [
        {"chunk_id": "old-1", "text": "best repeat"},
        {"chunk_id": "old-2", "text": "second repeat"},
    ]

    assert _filter_seen_chunks(chunks, {"old-1", "old-2"}) == [
        {"chunk_id": "old-1", "text": "best repeat"},
    ]


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
async def test_prepare_rag_context_cag_skips_rewrite_and_embedding(monkeypatch):
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
    monkeypatch.setattr(pipeline, "_rewrite_search_query", fake_rewrite)
    monkeypatch.setattr(pipeline, "_apply_glossary", lambda query: query)

    context = await chat._prepare_rag_context(
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
