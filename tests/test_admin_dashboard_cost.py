import pytest

from app.api.routes import admin


class Result:
    def __init__(self, *, scalar=None, one=None, all=None):
        self._scalar = scalar
        self._one = one
        self._all = all or []

    def scalar(self):
        return self._scalar

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._all


@pytest.mark.asyncio
async def test_dashboard_cost_uses_openrouter_logged_cost_only(monkeypatch):
    async def forbid_price_fetch():
        raise AssertionError("cost must come from agent_logs.or_cost")

    async def fake_run_one(sql, params=None):
        if "SELECT COUNT(*)" in sql:
            return Result(scalar=2)
        if "AVG(latency_ms)" in sql:
            return Result(scalar=25.0)
        if "SUM(or_prompt_tokens)" in sql and "SUM(or_cost)" in sql:
            return Result(one=(1000, 250, 80, 0.00042))
        if "GROUP BY intent" in sql:
            return Result(all=[("KNOWLEDGE", 2)])
        if "GROUP BY DATE(created_at)" in sql:
            return Result(all=[("2026-07-06", 2)])
        if "SELECT created_at, id" in sql:
            return Result(all=[
                (
                    "2026-07-06 00:00:00+00:00",
                    2,
                    "KNOWLEDGE",
                    10.0,
                    False,
                    "q with cost",
                    "a",
                    "s1",
                    12,
                    0,
                    None,
                    None,
                    None,
                    None,
                    [],
                    100,
                    20,
                    8,
                    "google-vertex",
                    None,
                    0.00042,
                    "gen-with-cost",
                ),
                (
                    "2026-07-05 00:00:00+00:00",
                    1,
                    "KNOWLEDGE",
                    15.0,
                    False,
                    "q without cost",
                    "a",
                    "s1",
                    10,
                    0,
                    None,
                    None,
                    None,
                    None,
                    [],
                    90,
                    0,
                    6,
                    "alibaba",
                    None,
                    None,
                    None,
                ),
            ])
        if "FROM user_profiles" in sql:
            return Result(all=[])
        if "percentile_cont" in sql:
            return Result(one=(10.0, 15.0, None, 0, 0))
        raise AssertionError(sql)

    monkeypatch.setattr(admin, "get_openrouter_prices", forbid_price_fetch, raising=False)
    monkeypatch.setattr(admin, "_run_one", fake_run_one)

    data = await admin.get_dashboard_logs(limit=10, _="ok")

    assert data["kpis"]["total_cost"] == 0.00042
    assert data["logs"][0]["cost"] == 0.00042
    assert data["logs"][1]["cost"] == 0.0
    assert data["logs"][0]["or_provider"] == "google-vertex"
    assert data["logs"][0]["or_generation_id"] == "gen-with-cost"
