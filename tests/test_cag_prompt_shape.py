from langchain_core.messages import HumanMessage, SystemMessage
import pytest


def test_build_cag_messages_keeps_stable_prefix_before_dynamic_context():
    from app.llm.cag_client import build_cag_messages

    kb_text = '<knowledge_base version="sha256:abc">KB</knowledge_base>'
    messages = build_cag_messages(
        kb_pack_text=kb_text,
        user_query="Apa itu Client Protection?",
        user_context={"name": "Ferdy", "role": "BM"},
        conversation_summary="User sebelumnya bahas fraud.",
        user_preferences={"preferred_tone": "singkat"},
    )

    assert [type(m) for m in messages] == [SystemMessage, SystemMessage, HumanMessage, HumanMessage]
    assert messages[1].content == kb_text
    assert "Ferdy" not in messages[0].content
    assert "Ferdy" not in messages[1].content
    assert "User sebelumnya bahas fraud." in messages[2].content
    assert messages[3].content == "Apa itu Client Protection?"


def test_build_cag_messages_omits_empty_dynamic_context():
    from app.llm.cag_client import build_cag_messages

    messages = build_cag_messages(kb_pack_text="KB", user_query="Halo")

    assert [type(m) for m in messages] == [SystemMessage, SystemMessage, HumanMessage]
    assert messages[-1].content == "Halo"


def test_extract_openrouter_usage_reads_cache_details():
    from app.llm.cag_client import extract_openrouter_usage

    class Message:
        response_metadata = {
            "model_name": "deepseek/deepseek-v4-flash",
            "provider_name": "alibaba",
            "token_usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {
                    "cached_tokens": 80,
                    "cache_write_tokens": 20,
                },
                "completion_tokens": 12,
            },
        }
        usage_metadata = {}

    usage = extract_openrouter_usage(Message())

    assert usage.prompt_tokens == 100
    assert usage.cached_tokens == 80
    assert usage.cache_write_tokens == 20
    assert usage.completion_tokens == 12
    assert usage.provider == "alibaba"


def test_cag_graph_routes_knowledge_directly_to_generate_node():
    from app.graph.pipeline import _route_by_intent

    assert _route_by_intent({"intent": "KNOWLEDGE"}) == "KNOWLEDGE"


@pytest.mark.asyncio
async def test_cag_graph_does_not_call_retrieval_for_knowledge(monkeypatch):
    from langchain_core.messages import AIMessageChunk

    from app.graph import pipeline

    async def fake_load_kb():
        return "<knowledge_base>KB</knowledge_base>"

    async def fake_log_cache_usage(*args, **kwargs):
        return None

    class FakeStreamingLLM:
        async def astream(self, messages, config=None):
            yield AIMessageChunk(content="OK")

    pipeline.get_cag_graph.cache_clear()
    monkeypatch.setattr(pipeline, "_load_active_cag_kb_text", fake_load_kb)
    monkeypatch.setattr(pipeline, "get_generate_llm", lambda: FakeStreamingLLM())
    monkeypatch.setattr(pipeline, "_log_cache_usage", fake_log_cache_usage)

    try:
        result = await pipeline.get_cag_graph().ainvoke(
            {"messages": [HumanMessage(content="Apa itu Client Protection?")]}
        )
    finally:
        pipeline.get_cag_graph.cache_clear()

    assert result["messages"][-1].content == "OK"


def test_cag_model_knob_removed_from_settings():
    from app.config.settings import Settings

    assert "cag_model" not in Settings.model_fields


@pytest.mark.asyncio
async def test_generate_node_binds_stable_prefix_openrouter_session_id(monkeypatch):
    import hashlib

    from langchain_core.messages import AIMessageChunk

    from app.graph import pipeline

    class FakeStreamingLLM:
        extra_body = {"usage": {"include": True}, "provider": {"order": ["xiaomi"]}}

        def __init__(self):
            self.bound_extra_bodies = []

        def bind(self, **kwargs):
            self.bound_extra_bodies.append(kwargs["extra_body"])
            return self

        async def astream(self, messages, config=None):
            yield AIMessageChunk(content="Halo")

    fake = FakeStreamingLLM()

    async def fake_log_cache_usage(*args, **kwargs):
        return None

    monkeypatch.setattr(pipeline, "get_chat_llm", lambda: fake)
    monkeypatch.setattr(pipeline, "_log_cache_usage", fake_log_cache_usage)

    for conversation_id in ("conv-a", "conv-b"):
        await pipeline._generate_node(
            {
                "messages": [HumanMessage(content="Halo")],
                "intent": "GREETING",
                "conversation_id": conversation_id,
            },
            {},
        )

    expected_session_id = "ava-prefix-" + hashlib.sha256(
        pipeline.CHIT_CHAT_PROMPT.encode()
    ).hexdigest()[:32]
    assert fake.bound_extra_bodies == [{
        "usage": {"include": True},
        "provider": {"order": ["xiaomi"]},
        "session_id": expected_session_id,
    }, {
        "usage": {"include": True},
        "provider": {"order": ["xiaomi"]},
        "session_id": expected_session_id,
    }]


@pytest.mark.asyncio
async def test_generate_node_accumulates_stream_chunks(monkeypatch):
    from langchain_core.messages import AIMessageChunk

    from app.graph import pipeline

    class FakeStreamingLLM:
        async def astream(self, messages, config=None):
            yield AIMessageChunk(content="Ha")
            yield AIMessageChunk(content="lo")

    async def fake_load_kb():
        return "<knowledge_base>KB</knowledge_base>"

    async def fake_log_cache_usage(*args, **kwargs):
        return None

    monkeypatch.setattr(pipeline, "_load_active_cag_kb_text", fake_load_kb)
    monkeypatch.setattr(pipeline, "get_generate_llm", lambda: FakeStreamingLLM())
    monkeypatch.setattr(pipeline, "_log_cache_usage", fake_log_cache_usage)

    result = await pipeline._generate_node(
        {"messages": [HumanMessage(content="Apa itu modal?")], "intent": "KNOWLEDGE"},
        {},
    )

    assert result["messages"][-1].content == "Halo"


@pytest.mark.asyncio
async def test_generate_node_falls_back_when_stream_returns_no_chunks(monkeypatch):
    from langchain_core.messages import AIMessage

    from app.graph import pipeline

    class EmptyStreamingLLM:
        async def astream(self, messages, config=None):
            if False:
                yield None

    class FallbackLLM:
        async def ainvoke(self, messages, config=None):
            return AIMessage(content="fallback answer")

    async def fake_load_kb():
        return "<knowledge_base>KB</knowledge_base>"

    async def fake_log_cache_usage(*args, **kwargs):
        return None

    monkeypatch.setattr(pipeline, "_load_active_cag_kb_text", fake_load_kb)
    monkeypatch.setattr(pipeline, "get_generate_llm", lambda: EmptyStreamingLLM())
    monkeypatch.setattr(pipeline, "get_generate_llm_nostream", lambda: FallbackLLM())
    monkeypatch.setattr(pipeline, "_log_cache_usage", fake_log_cache_usage)

    result = await pipeline._generate_node(
        {"messages": [HumanMessage(content="Apa itu modal?")], "intent": "KNOWLEDGE"},
        {},
    )

    assert result["messages"][-1].content == "fallback answer"


def test_chat_llm_is_streaming():
    from app.llm.client import get_chat_llm

    get_chat_llm.cache_clear()
    try:
        assert get_chat_llm().streaming is True
    finally:
        get_chat_llm.cache_clear()


def test_conversational_prompt_allows_offtopic_examples_for_in_scope_concepts():
    from app.llm.prompts import CONVERSATIONAL_PROMPT

    assert "off-topic example" in CONVERSATIONAL_PROMPT
    assert "actual requested subject is off-topic" in CONVERSATIONAL_PROMPT


def test_stream_leak_guard_streams_clean_text_immediately():
    from app.graph.pipeline import StreamLeakGuard

    guard = StreamLeakGuard()

    assert guard.feed("Halo") == "Halo"
    assert guard.feed(", siap") == ", siap"
    assert guard.flush() == ""
