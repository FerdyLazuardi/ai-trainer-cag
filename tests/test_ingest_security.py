import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import User, get_current_user
from app.api.routes import ingest


def test_moodle_sync_requires_admin_api_key(monkeypatch):
    app = FastAPI()
    app.include_router(ingest.router, prefix="/api/v1")
    app.dependency_overrides[get_current_user] = lambda: User(
        user_id="u-1",
        role="moodle_user",
        username="Regular User",
    )

    class FakeTask:
        id = "fake-task"

        def __await__(self):
            if False:
                yield None
            return None

    class FakeSyncTask:
        @staticmethod
        def enqueue(**kwargs):
            return FakeTask()

    worker_module = importlib.import_module("app.worker")

    monkeypatch.setattr(worker_module, "sync_moodle_task", FakeSyncTask)

    with TestClient(app) as client:
        response = client.post("/api/v1/ingest/moodle/sync", json={"course_id": 3})

    assert response.status_code == 401
