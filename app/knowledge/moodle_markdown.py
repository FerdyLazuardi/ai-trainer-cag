import hashlib
import html
from collections.abc import Iterable
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class MoodleMarkdownFile:
    course_id: int
    section_id: int
    section_name: str
    filename: str
    content: str
    content_hash: str


async def pull_moodle_markdown(
    course_ids: Iterable[int],
    *,
    client=None,
    api_url: str | None = None,
    token: str | None = None,
    target_sections: list[str] | None = None,
) -> list[MoodleMarkdownFile]:
    if api_url is None or token is None:
        from app.config.settings import get_settings

        settings = get_settings()
        api_url = api_url or settings.moodle_api_url
        token = token or settings.moodle_api_token

    if client is not None:
        return await _pull_with_client(course_ids, client, api_url, token, target_sections)

    async with httpx.AsyncClient(timeout=30.0) as owned_client:
        return await _pull_with_client(course_ids, owned_client, api_url, token, target_sections)


async def _pull_with_client(
    course_ids: Iterable[int],
    client,
    api_url: str,
    token: str,
    target_sections: list[str] | None,
) -> list[MoodleMarkdownFile]:
    docs: list[MoodleMarkdownFile] = []
    endpoint = f"{api_url.rstrip('/')}/webservice/rest/server.php"
    target_names = {s.strip().lower() for s in target_sections or []}

    for course_id in course_ids:
        resp = await client.post(
            endpoint,
            data={
                "wstoken": token,
                "wsfunction": "core_course_get_contents",
                "moodlewsrestformat": "json",
                "courseid": course_id,
            },
        )
        resp.raise_for_status()
        sections = resp.json()
        if isinstance(sections, dict) and "exception" in sections:
            raise ValueError(f"Moodle error: {sections.get('message', 'Unknown error')}")

        for section in sections:
            section_id = int(section.get("id") or section.get("section") or 0)
            section_name = html.unescape(section.get("name", "") or "")
            if target_names and section_name.strip().lower() not in target_names:
                continue
            for module in section.get("modules", []):
                for item in module.get("contents", []):
                    filename = item.get("filename", "")
                    fileurl = item.get("fileurl", "")
                    if not filename.lower().endswith(".md") or not fileurl:
                        continue

                    file_resp = await client.get(fileurl, params={"token": token})
                    file_resp.raise_for_status()
                    content = _normalize_text(file_resp.content.decode("utf-8", errors="replace"))
                    docs.append(
                        MoodleMarkdownFile(
                            course_id=course_id,
                            section_id=section_id,
                            section_name=section_name,
                            filename=filename,
                            content=content,
                            content_hash=hashlib.sha256(content.encode()).hexdigest(),
                        )
                    )

    return sorted(docs, key=lambda d: (d.course_id, d.section_id, d.filename))


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")
