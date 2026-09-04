"""
POST /ingest endpoint — [DISABLED] accepts raw text and triggers the ingestion pipeline.
POST /ingest/moodle/sync — triggers the Moodle AI_Knowledge_Base sync to CAG KB pack.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from app.api.schemas import IngestEnqueuedResponse, IngestRequest, IngestStatusResponse
from app.config.settings import get_settings
import httpx
from app.api.auth import get_current_user, User
from app.api.routes.admin import verify_api_key

router = APIRouter()
settings = get_settings()

@router.post(
    "/ingest",
    response_model=IngestEnqueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="[DISABLED] Enqueue a document for ingestion (RAG mode only)",
)
async def ingest(
    request: IngestRequest,
    current_user: User = Depends(get_current_user),
) -> IngestEnqueuedResponse:
    """
    **[DISABLED in CAG mode]**

    Raw text ingestion was used in RAG mode. In the current CAG mode, this endpoint is disabled (returns 410 Gone) because the knowledge base is compiled deterministically from Moodle via `/ingest/moodle/sync`.
    """
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Raw RAG ingestion is disabled in CAG mode. Use /ingest/moodle/sync."
    )


@router.get(
    "/ingest/{job_id}",
    response_model=IngestStatusResponse,
    summary="[DISABLED] Get the status of an enqueued ingestion job (RAG mode only)",
)
async def ingest_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> IngestStatusResponse:
    """
    **[DISABLED in CAG mode]**

    Get the status of an enqueued ingestion job. Since raw RAG ingestion is disabled, this endpoint is also disabled.
    """
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail="Raw RAG ingestion is disabled in CAG mode."
    )


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

    **Quick Start:** Just click "Execute" to sync all markdown files from AI_Knowledge_Base course.

    **Parameters:**
    - `course_id`: Moodle course ID (default: 3 = AI_Knowledge_Base)
    - `target_sections`: Optional list of section names to sync (leave empty for all)
    - `force_reingest`: Set to `true` to re-process files even if unchanged

    **What it does:**
    1. Enqueues a persistent job to the streaq worker.
    2. Downloads Moodle `.md` files, assembles one deterministic CAG KB pack.
    3. Saves the active KB pack in PostgreSQL database.
    4. Invalidates query and agent caches if changes are detected.
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
    current_user: User = Depends(get_current_user),
    _admin_key: str = Depends(verify_api_key),
) -> SpreadsheetSyncEnqueuedResponse:
    """
    Enqueue the paginated spreadsheet sync to the streaq worker (manual trigger).

    Returns 202 immediately so the request never hits the Cloudflare 120s
    proxy timeout — the worker fetches GAS page-by-page (limit=1000) with a
    900s task timeout. Poll ``GET /ingest/spreadsheet/status/{job_id}``.
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
    current_user: User = Depends(get_current_user),
    _admin_key: str = Depends(verify_api_key),
) -> SpreadsheetSyncStatusResponse:
    """Poll worker-side status/result for a spreadsheet sync job."""
    from app.worker import worker

    try:
        task_status = await worker.status_by_id(job_id)
        status_str = str(getattr(task_status, "value", task_status))
    except Exception:
        return SpreadsheetSyncStatusResponse(job_id=job_id, status="not_found")

    result: dict | None = None
    if status_str in ("finished", "done", "completed", "success"):
        try:
            task_result = await worker.result_by_id(job_id, timeout=0)
            raw = getattr(task_result, "result", task_result)
            if isinstance(raw, dict):
                result = raw
            else:
                result = {"result": raw}
        except Exception:
            result = None

    return SpreadsheetSyncStatusResponse(job_id=job_id, status=status_str, result=result)



@router.post("/test/dummy-task", summary="Enqueue a dummy task to verify the worker")
async def enqueue_dummy_task(
    name: str = "Tester",
    current_user: User = Depends(get_current_user),
    _admin_key: str = Depends(verify_api_key),
):
    """
    Enqueue a dummy task to verify that the streaq worker is running correctly.
    """
    logger.info(f"Dummy task enqueued by user: {current_user.username}")
    from app.worker import dummy_task

    task = dummy_task.enqueue(name=name)
    await task
    return {"message": f"Dummy task enqueued for {name}", "job_id": task.id}


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
    from app.config.settings import get_settings
    settings = get_settings()
    
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
            
            # Extract section names
            sections = []
            for section in sections_data:
                section_name = section.get("name", "").strip()
                if section_name and section_name != "":
                    sections.append({
                        "id": section.get("id"),
                        "name": section_name,
                        "summary": section.get("summary", "")[:100]  # First 100 chars
                    })
            
            logger.info(f"Fetched {len(sections)} sections from course {course_id} for user {current_user.username}")
            return sections
            
    except httpx.HTTPError as exc:
        logger.error(f"Failed to fetch Moodle sections: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch sections: {exc}")
    except Exception as exc:
        logger.error(f"Unexpected error fetching sections: {exc}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")
