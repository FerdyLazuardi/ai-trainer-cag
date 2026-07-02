def test_user_context_makes_answer_cache_user_scoped():
    from app.api.routes.chat import compute_was_personalized

    assert compute_was_personalized(
        ltm_profile={},
        user_pref_dict=None,
        recent_history=[],
        summary="",
        user_context={"name": "Ferdy", "dept": "Learning"},
    ) is True


def test_empty_user_context_stays_globally_cacheable():
    from app.api.routes.chat import compute_was_personalized

    assert compute_was_personalized(
        ltm_profile={},
        user_pref_dict=None,
        recent_history=[],
        summary="",
        user_context={"name": None, "dept": "", "location": None},
    ) is False


import pytest


@pytest.mark.asyncio
async def test_non_stream_cache_hit_does_not_acquire_pipeline_slot(monkeypatch):
    from app.api.routes import chat

    acquired = False

    async def fake_acquire():
        nonlocal acquired
        acquired = True
        raise AssertionError("cache hit should not acquire pipeline slot")

    async def fake_run_chat(request, background_tasks, current_user):
        return "ok"

    class Req:
        query = "apa itu modal"
        conversation_id = "c1"
        course_id = None
        coaching_mode = False

    class User:
        user_id = "u1"
        role = "staff"

    monkeypatch.setattr(chat, "acquire_pipeline_slot", fake_acquire)
    monkeypatch.setattr(chat, "_run_chat", fake_run_chat)

    result = await chat.chat(Req(), background_tasks=None, current_user=User())

    assert result == "ok"
    assert acquired is False
