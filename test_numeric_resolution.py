import pytest
from app.agents.memory import resolve_numeric_query

@pytest.mark.asyncio
async def test_resolve_numeric_query_hit(mocker):
    # Mock get_conversation_history to return a conversation with follow-up questions
    mock_history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Here is info.\n\n**Apa kamu penasaran tentang:**\n1. What is X?\n2. What is Y?\n3. What is Z?"}
    ]
    mocker.patch("app.agents.memory.get_conversation_history", return_value=mock_history)
    
    # Test resolving "1"
    resolved = await resolve_numeric_query("1", "fake_id")
    assert resolved == "What is X?"
    
    # Test resolving "2"
    resolved = await resolve_numeric_query("2", "fake_id")
    assert resolved == "What is Y?"

    # Test resolving "3"
    resolved = await resolve_numeric_query("3", "fake_id")
    assert resolved == "What is Z?"

@pytest.mark.asyncio
async def test_resolve_numeric_query_no_hit(mocker):
    mock_history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Just a normal response without follow-ups."}
    ]
    mocker.patch("app.agents.memory.get_conversation_history", return_value=mock_history)
    
    # Should return original query
    resolved = await resolve_numeric_query("1", "fake_id")
    assert resolved == "1"

@pytest.mark.asyncio
async def test_resolve_numeric_query_non_numeric():
    # Should return original query without checking history
    resolved = await resolve_numeric_query("how does it work?", "fake_id")
    assert resolved == "how does it work?"
