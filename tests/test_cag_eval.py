import pytest


@pytest.mark.asyncio
async def test_eval_uses_active_cag_kb_when_retrieved_context_empty(monkeypatch):
    from app.eval import tasks

    seen = {}

    async def fake_load_active_cag_kb_text():
        return "<knowledge_base>ground truth</knowledge_base>"

    async def fake_judge_faithfulness(query, answer, retrieved_context):
        seen["retrieved_context"] = retrieved_context

        class Result:
            score = 1.0
            unsupported_claims = []
            reasoning = "grounded"

        return Result()

    async def fake_persist(turn_id, score):
        return True

    monkeypatch.setattr(tasks, "_load_active_cag_kb_text", fake_load_active_cag_kb_text)
    monkeypatch.setattr(tasks, "judge_faithfulness", fake_judge_faithfulness)
    monkeypatch.setattr(tasks, "_persist_faithfulness", fake_persist)

    result = await tasks.eval_turn_task(
        query="apa itu modal",
        answer="jawaban",
        retrieved_context=[],
        intent="KNOWLEDGE",
        turn_id="t1",
    )

    assert result["status"] == "annotated"
    assert seen["retrieved_context"] == [{"text": "<knowledge_base>ground truth</knowledge_base>"}]
