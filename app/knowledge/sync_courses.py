"""Course catalog spreadsheet sync — mirrors active Moodle courses into Postgres.

GAS Web App serves only active courses ({courses: [...]}).
This module fetches the payload, upserts into course_catalog,
and cleans up any courses that are no longer active or deleted.
"""

import re
import httpx
from loguru import logger
from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.database.models import CourseCatalog


async def sync_courses_from_spreadsheet(session: AsyncSession) -> dict:
    """Fetch active courses from GAS Web App and sync to PostgreSQL course_catalog."""
    settings = get_settings()
    url = settings.course_spreadsheet_url
    token = settings.course_spreadsheet_token

    if not url:
        logger.warning("COURSE_SPREADSHEET_URL is not configured. Skipping course sync.")
        return {"status": "skipped", "message": "URL not configured"}

    logger.info(f"Starting course catalog sync from GAS Web App: {url}")
    fetch_url = url
    # If a standard Google Sheets URL is provided, automatically convert to gviz CSV export format
    if "docs.google.com/spreadsheets/d/" in fetch_url:
        import re as _re
        m = _re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", fetch_url)
        if m:
            sheet_id = m.group(1)
            fetch_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
            logger.info(f"Using Google Sheets CSV endpoint: {fetch_url}")

    params = {"token": token} if (token and "docs.google.com" not in fetch_url) else None

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(fetch_url, params=params, follow_redirects=True, timeout=60.0)
            resp.raise_for_status()

            courses_list: list[dict] = []
            text_content = getattr(resp, "text", "")
            url_str = str(getattr(resp, "url", ""))

            # Guard against Google sign-in redirect
            if "accounts.google.com" in url_str or "<title>Sign in - Google Accounts</title>" in text_content:
                msg = (
                    "Akses ditolak: Google meminta login akun. "
                    "Pastikan Apps Script di-deploy dengan akses 'Anyone' (Siapa saja), "
                    "atau gunakan link Google Spreadsheet publik langsung."
                )
                logger.error(msg)
                return {"status": "failed", "message": msg}

            headers = getattr(resp, "headers", {}) or {}
            content_type = headers.get("content-type", "").lower()

            if not text_content and hasattr(resp, "json"):
                payload = resp.json()
                if isinstance(payload, dict):
                    courses_list = payload.get("courses", payload.get("data", []))
                elif isinstance(payload, list):
                    courses_list = payload
            elif "json" in content_type or text_content.strip().startswith(("{", "[")):
                import json
                payload = json.loads(text_content)
                if isinstance(payload, dict):
                    courses_list = payload.get("courses", payload.get("data", []))
                elif isinstance(payload, list):
                    courses_list = payload
            else:
                # Parse CSV format
                import csv, io
                reader = csv.DictReader(io.StringIO(text_content, newline=""))
                for r in reader:
                    status = (r.get("status") or "").strip()
                    if status.lower() != "aktif":
                        continue
                    url_val = (r.get("lihat kelas") or r.get("url") or "").strip()
                    m = re.search(r"id=(\d+)", url_val)
                    cid = int(m.group(1)) if m else None
                    if not cid:
                        continue
                    courses_list.append({
                        "id": cid,
                        "fullname": (r.get("fullname") or "").strip(),
                        "category": (r.get("category") or "").strip(),
                        "status": "Aktif",
                        "url": url_val,
                    })
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}".strip(": ")
        logger.error(f"Failed to fetch course data: {err}")
        return {"status": "failed", "message": err}

    if not courses_list:
        logger.warning("Course catalog payload empty — skipping sync to avoid wipe")
        return {"status": "failed", "message": "empty payload"}

    incoming_ids: set[int] = set()
    courses_updated = 0

    for row in courses_list:
        if not isinstance(row, dict):
            continue

        # Hard guard: Only 'Aktif' courses should enter PostgreSQL
        status = str(row.get("status") or "").strip()
        if status.lower() != "aktif":
            continue

        raw_id = row.get("id") or row.get("course_id")
        url = str(row.get("url") or row.get("link") or "").strip()

        course_id: int | None = None
        if raw_id is not None:
            try:
                course_id = int(raw_id)
            except (ValueError, TypeError):
                course_id = None

        if course_id is None and url:
            m = re.search(r"id=(\d+)", url)
            if m:
                course_id = int(m.group(1))

        if not course_id:
            continue

        fullname = str(row.get("fullname") or row.get("title") or "").strip()
        category = str(row.get("category") or "").strip()

        if not url:
            url = f"https://academy.amartha.com/course/view.php?id={course_id}"

        if not fullname:
            continue

        stmt = insert(CourseCatalog).values(
            id=course_id,
            fullname=fullname,
            category=category,
            status="Aktif",
            url=url,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "fullname": stmt.excluded.fullname,
                "category": stmt.excluded.category,
                "status": stmt.excluded.status,
                "url": stmt.excluded.url,
            },
        )
        await session.execute(stmt)
        incoming_ids.add(course_id)
        courses_updated += 1

    courses_deleted = 0
    if incoming_ids:
        # Orphan deletion: delete any course in DB that is no longer in the active list
        res = await session.execute(
            delete(CourseCatalog).where(CourseCatalog.id.notin_(incoming_ids))
        )
        courses_deleted = int(getattr(res, "rowcount", 0) or 0)

    await session.commit()
    logger.info(
        f"Course catalog sync complete. Updated: {courses_updated}, Deleted (inactive/orphans): {courses_deleted}"
    )

    return {
        "status": "success",
        "courses_updated": courses_updated,
        "courses_deleted": courses_deleted,
    }
