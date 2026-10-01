import pytest
from app.database.models import CourseCatalog
from app.knowledge.sync_courses import sync_courses_from_spreadsheet
from app.knowledge.course_search import detect_course_query, search_courses


class FakeScalars:
    def __init__(self, items):
        self._items = items if isinstance(items, list) else ([items] if items is not None else [])

    def first(self):
        return self._items[0] if self._items else None

    def all(self):
        return self._items


class FakeResult:
    def __init__(self, items=None, rowcount=0):
        self._items = items or []
        self.rowcount = rowcount

    def scalars(self):
        return FakeScalars(self._items)

    def fetchall(self):
        return self._items


class FakeSession:
    def __init__(self, existing_courses=None):
        self.executed = []
        self.committed = False
        self.existing_courses = existing_courses or []

    async def execute(self, stmt, params=None):
        self.executed.append((stmt, params))
        # If it's a delete or select
        return FakeResult(items=self.existing_courses, rowcount=len(self.existing_courses))

    async def flush(self):
        pass

    async def commit(self):
        self.committed = True


def test_detect_course_query():
    # Positive triggers (direct turn)
    assert detect_course_query("minta link course kolaborasi analyst dong") == "kolaborasi analyst"
    assert detect_course_query("Ava, link kelas customer first ada?") == "customer first"
    assert detect_course_query("minta link pelatihan digital marketing ya") == "digital marketing"
    assert detect_course_query("tolong url modul risk based thinking") == "risk based thinking"

    # Positive triggers (follow-up contextual turn like 'Link nya', 'kasih linknya')
    assert detect_course_query("kasih linknya", previous_query="bisnis proses") == "bisnis proses"
    assert detect_course_query("bisa kasih linknya?", previous_query="tolong jelaskan proses bisnis") == "proses bisnis"
    assert detect_course_query("share linknya dong", previous_query="Customer First") == "Customer First"
    assert detect_course_query("Link nya", previous_query="Empower growth") == "Empower growth"
    assert detect_course_query("linknya apa?", previous_query="jelaskan tentang risk based thinking") == "risk based thinking"
    assert detect_course_query("minta linknya dong", previous_query="apa itu customer centric") == "customer centric"
    assert detect_course_query("mana linknya", previous_query="Collaborate to influence") == "Collaborate to influence"

    # Negative triggers (should NOT detect course link query)
    assert detect_course_query("jelaskan rumus DPD 1") is None
    assert detect_course_query("halo ava, selamat pagi") is None
    assert detect_course_query("berapa pencapaian cabang saya?") is None
    assert detect_course_query("apa bedanya PAR dan NPL?") is None
    assert detect_course_query("Link nya", previous_query=None) is None
    assert detect_course_query("kasih linknya", previous_query=None) is None
    assert detect_course_query("minta link", previous_query=None) is None


@pytest.mark.asyncio
async def test_sync_courses_from_spreadsheet_success(monkeypatch):
    class FakeSettings:
        course_spreadsheet_url = "https://script.google.com/macros/s/test_course/exec"
        course_spreadsheet_token = "test_token"

    monkeypatch.setattr("app.knowledge.sync_courses.get_settings", lambda: FakeSettings())

    COURSES_PAYLOAD = {
        "courses": [
            {
                "id": 2791,
                "fullname": "[Analyst Class] Collaborate to Influence",
                "category": "STAR Talent Program",
                "status": "Aktif",
                "url": "https://academy.amartha.com/course/view.php?id=2791",
            },
            {
                "id": 2796,
                "fullname": "[General Class] Customer First",
                "category": "STAR Talent Program",
                "status": "Aktif",
                "url": "https://academy.amartha.com/course/view.php?id=2796",
            },
            {
                # Should be discarded defensively if a non-active row slips through
                "id": 504,
                "fullname": "Digital Marketing 2023",
                "category": "Training Eksternal",
                "status": "Tidak Aktif",
                "url": "https://academy.amartha.com/course/view.php?id=504",
            },
        ]
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return COURSES_PAYLOAD

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def get(self, url, params, follow_redirects, timeout=60.0):
            assert url == "https://script.google.com/macros/s/test_course/exec"
            assert params.get("token") == "test_token"
            return FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", FakeClient)

    session = FakeSession()
    result = await sync_courses_from_spreadsheet(session)

    assert result["status"] == "success"
    # Only the 2 'Aktif' courses should be updated
    assert result["courses_updated"] == 2
    assert session.committed is True


@pytest.mark.asyncio
async def test_sync_courses_from_spreadsheet_skipped(monkeypatch):
    class FakeSettings:
        course_spreadsheet_url = ""
        course_spreadsheet_token = ""

    monkeypatch.setattr("app.knowledge.sync_courses.get_settings", lambda: FakeSettings())

    session = FakeSession()
    result = await sync_courses_from_spreadsheet(session)

    assert result["status"] == "skipped"
    assert len(session.executed) == 0


@pytest.mark.asyncio
async def test_search_courses_fallback(monkeypatch):
    sample_courses = [
        CourseCatalog(
            id=2791,
            fullname="[Analyst Class] Collaborate to Influence",
            category="STAR Talent Program",
            status="Aktif",
            url="https://academy.amartha.com/course/view.php?id=2791",
        )
    ]
    session = FakeSession(existing_courses=sample_courses)

    # Calling search_courses with fallback
    results = await search_courses(session, "collaborate", limit=1)
    assert len(results) == 1
    assert results[0]["id"] == 2791
    assert "Collaborate to Influence" in results[0]["fullname"]
