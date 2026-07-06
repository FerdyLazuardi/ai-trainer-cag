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
            "id": "gen-abc",
            "model_name": "deepseek/deepseek-v4-flash",
            "provider_name": "alibaba",
            "token_usage": {
                "prompt_tokens": 100,
                "prompt_tokens_details": {
                    "cached_tokens": 80,
                    "cache_write_tokens": 20,
                },
                "completion_tokens": 12,
                "cost": 0.00042,
            },
        }
        usage_metadata = {}

    usage = extract_openrouter_usage(Message())

    assert usage.prompt_tokens == 100
    assert usage.cached_tokens == 80
    assert usage.cache_write_tokens == 20
    assert usage.completion_tokens == 12
    assert usage.provider == "deepseek/deepseek-v4-flash"
    assert usage.cost == 0.00042
    assert usage.generation_id == "gen-abc"


@pytest.mark.asyncio
async def test_fetch_openrouter_generation_usage_reads_total_cost():
    from app.llm.cag_client import fetch_openrouter_generation_usage

    class Response:
        status_code = 200

        def json(self):
            return {
                "data": {
                    "id": "gen-abc",
                    "tokens_prompt": 100,
                    "native_tokens_cached": 80,
                    "tokens_completion": 12,
                    "provider_name": "alibaba",
                    "total_cost": 0.00042,
                }
            }

    class Client:
        async def get(self, url, **kwargs):
            assert url.endswith("/generation")
            assert kwargs["params"] == {"id": "gen-abc"}
            return Response()

    usage = await fetch_openrouter_generation_usage(
        "gen-abc",
        client=Client(),
        sleep=lambda _delay: None,
    )

    assert usage.prompt_tokens == 100
    assert usage.cached_tokens == 80
    assert usage.completion_tokens == 12
    assert usage.provider == "alibaba"
    assert usage.cost == 0.00042


def test_cag_graph_routes_knowledge_directly_to_generate_node():
    from app.graph.pipeline import _route_by_intent

    assert _route_by_intent({"intent": "KNOWLEDGE"}) == "KNOWLEDGE"


@pytest.mark.asyncio
async def test_cag_graph_does_not_call_retrieval_for_knowledge(monkeypatch):
    from langchain_core.messages import AIMessage

    from app.graph import pipeline

    async def fake_load_kb():
        return "<knowledge_base>KB</knowledge_base>"

    async def fake_log_cache_usage(*args, **kwargs):
        return None

    class FakeLLM:
        async def ainvoke(self, messages, config=None):
            return AIMessage(content="OK")

    pipeline.get_cag_graph.cache_clear()
    monkeypatch.setattr(pipeline, "_load_active_cag_kb_text", fake_load_kb)
    monkeypatch.setattr(pipeline, "get_generate_llm_nostream", lambda: FakeLLM())
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

    from langchain_core.messages import AIMessage

    from app.graph import pipeline

    class FakeLLM:
        extra_body = {"usage": {"include": True}, "provider": {"order": ["xiaomi"]}}

        def __init__(self):
            self.bound_extra_bodies = []

        def bind(self, **kwargs):
            self.bound_extra_bodies.append(kwargs["extra_body"])
            return self

        async def ainvoke(self, messages, config=None):
            return AIMessage(content="Halo")

    fake = FakeLLM()

    async def fake_log_cache_usage(*args, **kwargs):
        return None

    monkeypatch.setattr(pipeline, "get_generate_llm_nostream", lambda: fake)
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
async def test_generate_node_uses_nonstream_response_metadata(monkeypatch):
    from langchain_core.messages import AIMessage

    from app.graph import pipeline

    class FakeLLM:
        async def ainvoke(self, messages, config=None):
            return AIMessage(
                content="Halo",
                response_metadata={
                    "id": "gen-test",
                    "model_name": "inclusionai/ling-2.6-flash-20260421",
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "cost": 0.00042,
                    },
                },
            )

    async def fake_load_kb():
        return "<knowledge_base>KB</knowledge_base>"

    async def fake_log_cache_usage(*args, **kwargs):
        return None

    monkeypatch.setattr(pipeline, "_load_active_cag_kb_text", fake_load_kb)
    monkeypatch.setattr(pipeline, "get_generate_llm_nostream", lambda: FakeLLM())
    monkeypatch.setattr(pipeline, "_log_cache_usage", fake_log_cache_usage)

    result = await pipeline._generate_node(
        {"messages": [HumanMessage(content="Apa itu modal?")], "intent": "KNOWLEDGE"},
        {},
    )

    assert result["messages"][-1].content == "Halo"
    assert result["messages"][-1].response_metadata["id"] == "gen-test"


@pytest.mark.asyncio
async def test_generate_node_uses_nonstream_llm(monkeypatch):
    from langchain_core.messages import AIMessage

    from app.graph import pipeline

    class FakeLLM:
        async def ainvoke(self, messages, config=None):
            return AIMessage(content="fallback answer")

    async def fake_load_kb():
        return "<knowledge_base>KB</knowledge_base>"

    async def fake_log_cache_usage(*args, **kwargs):
        return None

    monkeypatch.setattr(pipeline, "_load_active_cag_kb_text", fake_load_kb)
    monkeypatch.setattr(pipeline, "get_generate_llm_nostream", lambda: FakeLLM())
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
