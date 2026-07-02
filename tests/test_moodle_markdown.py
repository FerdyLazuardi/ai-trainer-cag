import hashlib

import pytest


class FakeResponse:
    def __init__(self, payload=None, content=b""):
        self.payload = payload
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeMoodleClient:
    def __init__(self):
        self.downloads = []

    async def post(self, url, data):
        assert url == "https://moodle.test/webservice/rest/server.php"
        assert data["wsfunction"] == "core_course_get_contents"
        return FakeResponse([
            {
                "id": 20,
                "name": "Second",
                "modules": [
                    {"contents": [{"filename": "b.pdf", "fileurl": "https://files/b.pdf"}]},
                    {"contents": [{"filename": "b.md", "fileurl": "https://files/b.md"}]},
                ],
            },
            {
                "id": 10,
                "name": "First",
                "modules": [
                    {"contents": [{"filename": "a.md", "fileurl": "https://files/a.md"}]},
                ],
            },
        ])

    async def get(self, url, params):
        self.downloads.append((url, params))
        return FakeResponse(content=f"# {url}\r\nbody".encode())


@pytest.mark.asyncio
async def test_pull_moodle_markdown_downloads_md_files_sorted_and_hashed():
    from app.knowledge.moodle_markdown import pull_moodle_markdown

    client = FakeMoodleClient()

    docs = await pull_moodle_markdown(
        [7],
        client=client,
        api_url="https://moodle.test/",
        token="secret",
    )

    assert [(d.course_id, d.section_id, d.filename) for d in docs] == [
        (7, 10, "a.md"),
        (7, 20, "b.md"),
    ]
    assert docs[0].content == "# https://files/a.md\nbody"
    assert docs[0].content_hash == hashlib.sha256(docs[0].content.encode()).hexdigest()
    assert client.downloads == [
        ("https://files/b.md", {"token": "secret"}),
        ("https://files/a.md", {"token": "secret"}),
    ]
