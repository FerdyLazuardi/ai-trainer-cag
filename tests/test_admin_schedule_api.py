import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi import HTTPException
from app.api.auth import get_admin_or_jwt_user, User


class FakeRequest:
    def __init__(self, headers: dict | None = None):
        self.headers = headers or {}
        self.client = MagicMock(host="127.0.0.1")
        self.query_params = {}


@pytest.mark.asyncio
async def test_get_admin_or_jwt_user_with_valid_api_key(monkeypatch):
    import app.api.auth as auth_mod
    monkeypatch.setattr(auth_mod.settings, "admin_api_key", "secret_admin_key_123")

    req = FakeRequest(headers={"x-api-key": "secret_admin_key_123"})
    user = await get_admin_or_jwt_user(req, credentials=None)

    assert user.user_id == "admin"
    assert user.role == "admin"
    assert user.username == "admin"


@pytest.mark.asyncio
async def test_get_admin_or_jwt_user_with_invalid_api_key_and_no_jwt(monkeypatch):
    import app.api.auth as auth_mod
    monkeypatch.setattr(auth_mod.settings, "admin_api_key", "secret_admin_key_123")
    monkeypatch.setattr(auth_mod.settings, "dev_bypass_enabled", False)

    req = FakeRequest(headers={"x-api-key": "wrong_key"})
    with pytest.raises(HTTPException) as exc_info:
        await get_admin_or_jwt_user(req, credentials=None)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_admin_schedule_get_and_set(monkeypatch):
    from app.api.routes.admin import get_spreadsheet_schedule, update_spreadsheet_schedule

    store = {}

    class FakeRedis:
        async def get(self, key):
            return store.get(key)

        async def set(self, key, val):
            store[key] = val

    monkeypatch.setattr("app.database.redis_client.get_redis_client", lambda: FakeRedis())

    # Initial get (default empty)
    res_default = await get_spreadsheet_schedule(_="valid_key")
    assert res_default["enabled"] is False
    assert res_default["schedule_type"] == "daily"
    assert res_default["hour"] == 2

    # Update schedule
    new_sched = {
        "enabled": True,
        "schedule_type": "weekly",
        "hour": 3,
        "minute": 0,
        "day_of_week": 1,
    }
    res_updated = await update_spreadsheet_schedule(new_sched, _="valid_key")
    assert res_updated["enabled"] is True
    assert res_updated["schedule_type"] == "weekly"
    assert res_updated["hour"] == 3
    assert res_updated["day_of_week"] == 1

    # Verify retrieval
    res_fetched = await get_spreadsheet_schedule(_="valid_key")
    assert res_fetched["enabled"] is True
    assert res_fetched["schedule_type"] == "weekly"
    assert res_fetched["hour"] == 3
