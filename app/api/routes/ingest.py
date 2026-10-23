"""
CAG KB & Spreadsheet Sync Endpoints:
- POST /ingest/moodle/sync — triggers the Moodle AI_Knowledge_Base sync to CAG KB pack.
- POST /ingest/spreadsheet/sync — triggers paginated GAS KPI/branch spreadsheet sync.
- POST /ingest/courses/sync — triggers active Moodle course catalog sync.
- POST /ingest/history/sync — triggers historical KPI sync.
"""
import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from app.api.auth import User, get_admin_or_jwt_user, get_current_user
from app.api.routes.admin import verify_api_key
from app.config.settings import get_settings

router = APIRouter()
settings = get_settings()


class MoodleSyncRequest(BaseModel):
    course_id: int = Field(default=3, description="Moodle Course ID (defaults to 3 = AI_Knowledge_Base)")
    target_sections: list[str] | None = Field(default=None, description="Specific sections to sync (case-insensitive, leave null for all sections)")
    force_reingest: bool = Field(default=True, description="Force re-ingestion even if content hash matches")

    model_config = {
        "json_schema_extra": {
            "example": {
                "course_id": 3,
                "target_sections": None,
                "force_reingest": True
            }
        }
    }


class MoodleSyncResponse(BaseModel):
    message: str


@router.post(
    "/ingest/moodle/sync",
    response_model=MoodleSyncResponse,
    summary="Sync Moodle AI_Knowledge_Base course to the active CAG KB pack",
)
async def moodle_sync(
    request: MoodleSyncRequest,
    current_user: User = Depends(get_current_user),
    _admin_key: str = Depends(verify_api_key),
) -> MoodleSyncResponse:
    """
    Trigger background sync of Moodle course (default: course_id=3 AI_Knowledge_Base) via streaq worker.
    """
    logger.info(f"Moodle sync triggered by user: {current_user.username}")
    from app.worker import sync_moodle_task

    task = sync_moodle_task.enqueue(
        course_id=request.course_id,
        target_sections=request.target_sections,
        force_reingest=request.force_reingest,
    )
    await task

    return MoodleSyncResponse(
        message="Moodle sync task has been successfully enqueued to the persistent background worker."
    )


class SpreadsheetSyncEnqueuedResponse(BaseModel):
    message: str
    job_id: str
    status: str = "queued"


class SpreadsheetSyncStatusResponse(BaseModel):
    job_id: str
    status: str  # queued | started | finished | failed | not_found
    result: dict | None = None


@router.post(
    "/ingest/spreadsheet/sync",
    response_model=SpreadsheetSyncEnqueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue weekly Google Spreadsheet KPI and branch sync (async)",
)
async def spreadsheet_sync(
    current_user: User = Depends(get_admin_or_jwt_user),
) -> SpreadsheetSyncEnqueuedResponse:
    """
    Enqueue the paginated spreadsheet sync to the streaq worker (manual trigger).
    """
    logger.info(f"Spreadsheet sync enqueued by user: {current_user.username}")
    from app.worker import sync_spreadsheet_task

    task = sync_spreadsheet_task.enqueue()
    await task

    return SpreadsheetSyncEnqueuedResponse(
        message="Spreadsheet sync enqueued. Poll GET /ingest/spreadsheet/status/{job_id}.",
        job_id=task.id,
        status="queued",
    )


@router.get(
    "/ingest/spreadsheet/status/{job_id}",
    response_model=SpreadsheetSyncStatusResponse,
    summary="Get the status of an enqueued spreadsheet sync job",
)
async def spreadsheet_sync_status(
    job_id: str,
    current_user: User = Depends(get_admin_or_jwt_user),
) -> SpreadsheetSyncStatusResponse:
    """Poll worker-side status/result for a spreadsheet sync job."""
    from app.worker import worker

    try:
        task_status = await worker.status_by_id(job_id)
        status_str = str(getattr(task_status, "value", task_status))
    except Exception:
        return SpreadsheetSyncStatusResponse(job_id=job_id, status="not_found")

    result: dict | None = None
    if status_str == "done":
        try:
            task_result = await worker.result_by_id(job_id, timeout=10)
            try:
                raw = task_result.result
            except Exception as exc:
                result = {"status": "failed", "error": str(getattr(task_result, 'exception', exc))}
            else:
                if isinstance(raw, dict):
                    result = raw
                else:
                    result = {"result": raw}
        except Exception:
            result = None

    return SpreadsheetSyncStatusResponse(job_id=job_id, status=status_str, result=result)


class CourseSyncEnqueuedResponse(BaseModel):
    message: str
    job_id: str
    status: str


@router.post(
    "/ingest/courses/sync",
    response_model=CourseSyncEnqueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue active course catalog sync from Google Spreadsheet (async)",
)
async def courses_sync(
    current_user: User = Depends(get_admin_or_jwt_user),
) -> CourseSyncEnqueuedResponse:
    """Enqueue course catalog sync from GAS Web App to the background worker."""
    logger.info(f"Course catalog sync enqueued by user: {current_user.username}")
    from app.worker import sync_courses_task

    task = sync_courses_task.enqueue()
    await task

    return CourseSyncEnqueuedResponse(
        message="Course catalog sync enqueued. Poll GET /ingest/courses/status/{job_id}.",
        job_id=task.id,
        status="queued",
    )


@router.get(
    "/ingest/courses/status/{job_id}",
    response_model=SpreadsheetSyncStatusResponse,
    summary="Get status of an enqueued course catalog sync job",
)
async def courses_sync_status(
    job_id: str,
    current_user: User = Depends(get_admin_or_jwt_user),
) -> SpreadsheetSyncStatusResponse:
    """Poll worker-side status/result for a course sync job."""
    from app.worker import worker

    try:
        task_status = await worker.status_by_id(job_id)
        status_str = str(getattr(task_status, "value", task_status))
    except Exception:
        return SpreadsheetSyncStatusResponse(job_id=job_id, status="not_found")

    result: dict | None = None
    if status_str == "done":
        try:
            task_result = await worker.result_by_id(job_id, timeout=10)
            try:
                raw = task_result.result
            except Exception as exc:
                result = {"status": "failed", "error": str(getattr(task_result, 'exception', exc))}
            else:
                if isinstance(raw, dict):
                    result = raw
                else:
                    result = {"result": raw}
        except Exception:
            result = None

    return SpreadsheetSyncStatusResponse(job_id=job_id, status=status_str, result=result)


class HistorySyncEnqueuedResponse(BaseModel):
    message: str
    job_id: str
    status: str


@router.post(
    "/ingest/history/sync",
    response_model=HistorySyncEnqueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue active KPI history sync from Google Spreadsheet (async)",
)
async def history_sync(
    current_user: User = Depends(get_admin_or_jwt_user),
) -> HistorySyncEnqueuedResponse:
    """Enqueue historical KPI sync from GAS Web App to the background worker."""
    logger.info(f"Historical KPI sync enqueued by user: {current_user.username}")
    from app.worker import sync_history_task

    task = sync_history_task.enqueue()
    await task

    return HistorySyncEnqueuedResponse(
        message="Historical KPI sync enqueued. Poll GET /ingest/history/status/{job_id}.",
        job_id=task.id,
        status="queued",
    )


@router.get(
    "/ingest/history/status/{job_id}",
    response_model=SpreadsheetSyncStatusResponse,
    summary="Get status of an enqueued historical KPI sync job",
)
async def history_sync_status(
    job_id: str,
    current_user: User = Depends(get_admin_or_jwt_user),
) -> SpreadsheetSyncStatusResponse:
    """Poll worker-side status/result for a historical KPI sync job."""
    from app.worker import worker

    try:
        task_status = await worker.status_by_id(job_id)
        status_str = str(getattr(task_status, "value", task_status))
    except Exception:
        return SpreadsheetSyncStatusResponse(job_id=job_id, status="not_found")

    result: dict | None = None
    if status_str == "done":
        try:
            task_result = await worker.result_by_id(job_id, timeout=10)
            try:
                raw = task_result.result
            except Exception as exc:
                result = {"status": "failed", "error": str(getattr(task_result, 'exception', exc))}
            else:
                if isinstance(raw, dict):
                    result = raw
                else:
                    result = {"result": raw}
        except Exception:
            result = None

    return SpreadsheetSyncStatusResponse(job_id=job_id, status=status_str, result=result)


@router.get("/moodle/sections", summary="Get Moodle course sections")
async def get_moodle_sections(
    course_id: int = 3,
    current_user: User = Depends(get_current_user),
    _admin_key: str = Depends(verify_api_key),
):
    """
    Fetch sections from a Moodle course.

    Returns list of section names that can be used for selective ingestion.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.moodle_api_url}/webservice/rest/server.php",
                data={
                    "wstoken": settings.moodle_api_token,
                    "wsfunction": "core_course_get_contents",
                    "moodlewsrestformat": "json",
                    "courseid": course_id,
                },
            )
            resp.raise_for_status()
            sections_data = resp.json()

            if isinstance(sections_data, dict) and "exception" in sections_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Moodle error: {sections_data.get('message', 'Unknown error')}"
                )

            sections = []
            for section in sections_data:
                section_name = section.get("name", "").strip()
                if section_name and section_name != "":
                    sections.append({
                        "id": section.get("id"),
                        "name": section_name,
                        "summary": section.get("summary", "")[:100]
                    })

            logger.info(f"Fetched {len(sections)} sections from course {course_id} for user {current_user.username}")
            return sections

    except httpx.HTTPError as exc:
        logger.error(f"Failed to fetch Moodle sections: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch sections: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error fetching sections: {exc}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")
