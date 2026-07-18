import pytest
from app.database.models import UserKPIData, BranchData
from app.knowledge.sync_spreadsheet import sync_kpi_from_spreadsheet


class FakeScalars:
    def __init__(self, item):
        self.item = item

    def first(self):
        return self.item


class FakeResult:
    def __init__(self, item=None):
        self.item = item

    def scalars(self):
        return FakeScalars(self.item)


class FakeSession:
    def __init__(self):
        self.executed = []
        self.committed = False

    async def execute(self, stmt):
        self.executed.append(stmt)
        return FakeResult()

    async def commit(self):
        self.committed = True


@pytest.mark.asyncio
async def test_sync_kpi_from_spreadsheet_success(monkeypatch):
    # Mock settings
    class FakeSettings:
        spreadsheet_sync_url = "https://example.com/gas"
        spreadsheet_sync_token = "test_token"

    monkeypatch.setattr("app.knowledge.sync_spreadsheet.get_settings", lambda: FakeSettings())

    # Mock httpx response
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {
                    "user_id": "user1",
                    "full_name": "User One",
                    "KPI 2026": "KPI 1",
                    "Jumlah Mitra Lancar": 10,
                    "Jumlah Mitra Nunggak": 5
                },
                {
                    "point": "cabangA",
                    "nama_cabang": "Cabang A",
                    "target_cabang": "Target A",
                    "total_mitra_aktif": 50,
                    "npl_cabang": "1.2%"
                }
            ]

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def get(self, url, params, follow_redirects, timeout=30.0):
            assert url == "https://example.com/gas"
            assert params == {"token": "test_token"}
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeClient)

    session = FakeSession()
    result = await sync_kpi_from_spreadsheet(session)

    assert result["status"] == "success"
    assert result["users_updated"] == 1
    assert result["branches_updated"] == 1
    assert len(session.executed) == 2
    assert session.committed is True


@pytest.mark.asyncio
async def test_sync_kpi_from_spreadsheet_skipped(monkeypatch):
    class FakeSettings:
        spreadsheet_sync_url = ""
        spreadsheet_sync_token = "test_token"

    monkeypatch.setattr("app.knowledge.sync_spreadsheet.get_settings", lambda: FakeSettings())

    session = FakeSession()
    result = await sync_kpi_from_spreadsheet(session)

    assert result["status"] == "skipped"
    assert len(session.executed) == 0
