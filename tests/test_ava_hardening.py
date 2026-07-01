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
@pytest.mark.parametrize("query", [
    "Gimana caranya menangani mitra yang telat bayar cicilan?",
    "gimana caranya aku melindungi data mitra ya",
])
async def test_meta_convo_regex_does_not_swallow_how_to_knowledge(monkeypatch, query):
    from app.graph import pipeline

    monkeypatch.setattr(pipeline._settings, "intent_semantic_gate_enabled", False)

    result = await pipeline._pre_processor(
        {"messages": [HumanMessage(content=query)]},
        {},
    )

    assert result["intent"] == "KNOWLEDGE"


@pytest.mark.asyncio
async def test_meta_convo_regex_keeps_bare_how_to_ambiguous(monkeypatch):
    from app.graph import pipeline

    monkeypatch.setattr(pipeline._settings, "intent_semantic_gate_enabled", False)

    result = await pipeline._pre_processor(
        {"messages": [HumanMessage(content="gimana caranya?")]},
        {},
    )

    assert result["intent"] == "AMBIGUOUS"


@pytest.mark.asyncio
async def test_example_followup_can_use_knowledge_rewrite_path(monkeypatch):
    from app.graph import pipeline

    monkeypatch.setattr(pipeline._settings, "intent_semantic_gate_enabled", False)

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
async def test_rewrite_prompt_puts_newest_history_first(monkeypatch):
    from app.graph import pipeline
    import app.llm.client as llm_client

    captured = {}

    class FakeLLM:
        async def ainvoke(self, messages):
            captured["prompt"] = messages[0].content
            return AIMessage(content="resolved query")

    monkeypatch.setattr(llm_client, "get_preprocessor_llm", lambda: FakeLLM())

    messages = [
        HumanMessage(content="ancient user"),
        AIMessage(content="ancient assistant"),
        HumanMessage(content="oldest user"),
        AIMessage(content="oldest assistant"),
        HumanMessage(content="middle user"),
        AIMessage(content="middle assistant"),
        HumanMessage(content="newest user"),
        AIMessage(content="newest options\n1. Latest A\n2. Latest B"),
        HumanMessage(content="1"),
    ]

    await pipeline._rewrite_search_query(messages, "1")
    prompt = captured["prompt"]

    assert prompt.index("Ava: newest options") < prompt.index("User: newest user")
    assert prompt.index("User: newest user") < prompt.index("Ava: middle assistant")
    assert "ancient user" not in prompt


@pytest.mark.asyncio
async def test_rewrite_prompt_forbids_dropping_subquestions_and_inventing_terms(monkeypatch):
    from app.graph import pipeline
    import app.llm.client as llm_client

    captured = {}

    class FakeLLM:
        async def ainvoke(self, messages):
            captured["prompt"] = messages[0].content
            return AIMessage(content="baju hari kamis\nmelaporkan fraud")

    monkeypatch.setattr(llm_client, "get_preprocessor_llm", lambda: FakeLLM())

    await pipeline._rewrite_search_query(
        [HumanMessage(content="sebelumnya"), HumanMessage(content="besok pake baju apa, cara lapor fraud gimana")],
        "besok pake baju apa, cara lapor fraud gimana",
    )

    prompt = captured["prompt"]

    assert "NEVER DROP" in prompt
    assert "retrieval/gating decides relevance later" in prompt
    assert "Do not invent new domain terms" in prompt


@pytest.mark.asyncio
async def test_rewrite_query_cleans_verbatim_noise_and_splits_fraud_facets(monkeypatch):
    from app.graph import pipeline
    import app.llm.client as llm_client

    noisy = (
        "aku kan Business Manager point lenteng, point aku ini lagi fraud parah, "
        "aku harusn gapian biar angka fraud juga menurun"
    )

    class FakeLLM:
        async def ainvoke(self, _messages):
            return AIMessage(content=noisy)

    monkeypatch.setattr(llm_client, "get_preprocessor_llm", lambda: FakeLLM())

    rewritten = await pipeline._rewrite_search_query(
        [HumanMessage(content="sebelumnya"), HumanMessage(content=noisy)],
        noisy,
    )

    assert rewritten.splitlines() == [
        "point fraud Business Manager",
        "fraud menurun Business Manager point",
    ]
    assert "lenteng" not in rewritten.lower()
    assert "aku" not in rewritten.lower()
    assert "surprise visit" not in rewritten.lower()


@pytest.mark.asyncio
async def test_rewrite_query_cleans_generic_question_filler(monkeypatch):
    from app.graph import pipeline
    import app.llm.client as llm_client

    class FakeLLM:
        async def ainvoke(self, _messages):
            return AIMessage(content="client protection itu apaan ya")

    monkeypatch.setattr(llm_client, "get_preprocessor_llm", lambda: FakeLLM())

    rewritten = await pipeline._rewrite_search_query(
        [HumanMessage(content="sebelumnya"), HumanMessage(content="client protection itu apaan ya")],
        "client protection itu apaan ya",
    )

    assert rewritten == "client protection"


@pytest.mark.asyncio
async def test_rewrite_query_splits_mixed_questions_without_compressing(monkeypatch):
    from app.graph import pipeline
    import app.llm.client as llm_client

    mixed = "besok pake baju apa, cara lapor fraud gmna?"

    class FakeLLM:
        async def ainvoke(self, _messages):
            return AIMessage(content="baju hari kamis\nmelaporkan fraud")

    monkeypatch.setattr(llm_client, "get_preprocessor_llm", lambda: FakeLLM())

    rewritten = await pipeline._rewrite_search_query(
        [HumanMessage(content="sebelumnya"), HumanMessage(content=mixed)],
        mixed,
    )

    assert rewritten.splitlines() == [
        "baju hari kamis",
        "melaporkan fraud",
    ]


@pytest.mark.asyncio
async def test_rewrite_query_raw_fallback_splits_without_semantic_hardcode(monkeypatch):
    from app.graph import pipeline
    import app.llm.client as llm_client

    mixed = "besok pake baju apa, cara lapor fraud gmna?"

    class FakeLLM:
        async def ainvoke(self, _messages):
            return AIMessage(content=mixed)

    monkeypatch.setattr(llm_client, "get_preprocessor_llm", lambda: FakeLLM())

    rewritten = await pipeline._rewrite_search_query(
        [HumanMessage(content="sebelumnya"), HumanMessage(content=mixed)],
        mixed,
    )

    assert rewritten.splitlines() == [
        "besok pake baju",
        "cara lapor fraud",
    ]


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

    assert "RAG pipeline failed" in body
    assert len(logs) == 1
    assert logs[0]["endpoint"] == "chat-stream"
    assert logs[0]["answer"] == ""
    assert cache_writes == []
    assert history_writes == []
