import asyncio
from unittest.mock import AsyncMock, patch
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph.pipeline import _classify_intent_cheap_llm, _pre_processor
from app.graph.state import CAGState


@pytest.mark.asyncio
async def test_classify_empty_string_returns_ambiguous():
    res = await _classify_intent_cheap_llm("   ")
    assert res == "AMBIGUOUS"


@pytest.mark.asyncio
async def test_classify_valid_intents(monkeypatch):
    class FakeLLM:
        def __init__(self, reply):
            self.reply = reply

        async def ainvoke(self, messages, config=None):
            return AIMessage(content=self.reply)

    for intent in ["AMBIGUOUS", "GREETING", "OFF_SCOPE", "TOPIC_LIST", "KNOWLEDGE"]:
        monkeypatch.setattr("app.graph.pipeline.get_intent_llm", lambda intent=intent: FakeLLM(intent))
        res = await _classify_intent_cheap_llm("sample query")
        assert res == intent


@pytest.mark.asyncio
async def test_classify_cleans_token_with_punctuation(monkeypatch):
    class FakeLLM:
        async def ainvoke(self, messages, config=None):
            return AIMessage(content='  "AMBIGUOUS."  \n')

    monkeypatch.setattr("app.graph.pipeline.get_intent_llm", lambda: FakeLLM())
    res = await _classify_intent_cheap_llm("sudah lapar")
    assert res == "AMBIGUOUS"


@pytest.mark.asyncio
async def test_classify_timeout_returns_none(monkeypatch):
    class HangingLLM:
        async def ainvoke(self, messages, config=None):
            await asyncio.sleep(5)
            return AIMessage(content="KNOWLEDGE")

    monkeypatch.setattr("app.graph.pipeline.get_intent_llm", lambda: HangingLLM())
    # patch timeout or allow natural timeout
    with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
        res = await _classify_intent_cheap_llm("sudah lapar")
        assert res is None


@pytest.mark.asyncio
async def test_classify_exception_returns_none(monkeypatch):
    class FailingLLM:
        async def ainvoke(self, messages, config=None):
            raise RuntimeError("OpenRouter 500 error")

    monkeypatch.setattr("app.graph.pipeline.get_intent_llm", lambda: FailingLLM())
    res = await _classify_intent_cheap_llm("sudah lapar")
    assert res is None


@pytest.mark.asyncio
async def test_pre_processor_tier2_routes_to_ambiguous(monkeypatch):
    """Test that a query missing regex (e.g. 'Sudah lapar') is caught by Tier-2 LLM."""
    monkeypatch.setattr("app.graph.pipeline._classify_intent_cheap_llm", AsyncMock(return_value="AMBIGUOUS"))

    state: CAGState = {
        "messages": [HumanMessage(content="Sudah lapar")],
        "conversation_id": "test-convo",
    }
    result = await _pre_processor(state, {})

    assert result["intent"] == "AMBIGUOUS"
    assert result["intent_scores"]["needs_lookup"] == 0.0


@pytest.mark.asyncio
async def test_pre_processor_tier2_routes_to_knowledge(monkeypatch):
    """Test that a real question classified as KNOWLEDGE by Tier-2 initiates retrieval."""
    monkeypatch.setattr("app.graph.pipeline._classify_intent_cheap_llm", AsyncMock(return_value="KNOWLEDGE"))

    state: CAGState = {
        "messages": [HumanMessage(content="Berapa margin keuntungan produk pinjaman?")],
        "conversation_id": "test-convo",
    }
    result = await _pre_processor(state, {})

    assert result["intent"] == "KNOWLEDGE"
    assert result["intent_scores"]["needs_lookup"] == 1.0


@pytest.mark.asyncio
async def test_pre_processor_tier2_fallback_on_none(monkeypatch):
    """Test that when Tier-2 returns None (e.g. timeout), it safely falls back to KNOWLEDGE."""
    monkeypatch.setattr("app.graph.pipeline._classify_intent_cheap_llm", AsyncMock(return_value=None))

    state: CAGState = {
        "messages": [HumanMessage(content="Kmi lagi treaning")],
        "conversation_id": "test-convo",
    }
    result = await _pre_processor(state, {})

    assert result["intent"] == "KNOWLEDGE"
    assert result["intent_scores"]["needs_lookup"] == 1.0
