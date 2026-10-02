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

    async def flush(self):
        pass

    async def commit(self):
        self.committed = True


@pytest.mark.asyncio
async def test_sync_kpi_from_spreadsheet_success(monkeypatch):
    # Mock settings
    class FakeSettings:
        spreadsheet_sync_url = "https://example.com/gas"
        spreadsheet_sync_token = "test_token"

    monkeypatch.setattr("app.knowledge.sync_spreadsheet.get_settings", lambda: FakeSettings())

    USER_ROW = {
        "user_id": "user1",
        "full_name": "User One",
        "KPI 2026": "KPI 1",
        "Jumlah Mitra Lancar": 10,
        "Jumlah Mitra Nunggak": 5
    }
    BRANCH_ROW = {
        "point": "cabangA",
        "nama_cabang": "Cabang A",
        "target_cabang": "Target A",
        "total_mitra_aktif": 50,
        "npl_cabang": "1.2%"
    }

    # Mock httpx response (paginated GAS: per-scope pages + legacy fallback)
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def get(self, url, params, follow_redirects, timeout=30.0):
            assert url == "https://example.com/gas"
            assert params.get("token") == "test_token"
            scope = params.get("scope")
            if scope == "users":
                return FakeResponse([USER_ROW])
            if scope == "branches":
                return FakeResponse([BRANCH_ROW])
            # legacy single-shot fallback: combined flat list
            return FakeResponse([USER_ROW, BRANCH_ROW])

    monkeypatch.setattr("httpx.AsyncClient", FakeClient)

    session = FakeSession()
    result = await sync_kpi_from_spreadsheet(session)

    assert result["status"] == "success"
    assert result["users_updated"] == 1
    assert result["branches_updated"] == 1
    assert result["users_deleted"] == 0
    assert result["branches_deleted"] == 0
    # 2 upserts + 2 orphan deletes
    assert len(session.executed) == 4
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


@pytest.mark.asyncio
async def test_sync_kpi_from_spreadsheet_retry_success(monkeypatch):
    class FakeSettings:
        spreadsheet_sync_url = "https://example.com/gas"
        spreadsheet_sync_token = "test_token"

    monkeypatch.setattr("app.knowledge.sync_spreadsheet.get_settings", lambda: FakeSettings())
    monkeypatch.setattr("app.knowledge.sync_spreadsheet.PAGE_DELAY_SECONDS", 0.0)

    USER_ROW = {
        "username": "user1",
        "full_name": "User One",
    }

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    call_count = {"users": 0}

    class FakeClientWithRetry:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def get(self, url, params, follow_redirects, timeout=90.0):
            scope = params.get("scope")
            if scope == "users":
                call_count["users"] += 1
                if call_count["users"] == 1:
                    # Simulate transient 404 / gateway timeout on first attempt
                    raise httpx.HTTPStatusError("404 Not Found", request=None, response=None)
                return FakeResponse([USER_ROW])
            if scope == "branches":
                return FakeResponse([])
            return FakeResponse([])

    monkeypatch.setattr("httpx.AsyncClient", FakeClientWithRetry)

    session = FakeSession()
    result = await sync_kpi_from_spreadsheet(session)

    assert result["status"] == "success"
    assert result["users_updated"] == 1
    assert call_count["users"] == 2

