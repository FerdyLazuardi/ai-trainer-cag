"""
POST /ingest endpoint — accepts raw text and triggers the ingestion pipeline.
POST /ingest/moodle/sync — triggers the Moodle AI_Knowledge_Base sync.
"""
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request, status
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from streaq import TaskStatus

from app.api.schemas import IngestRequest, IngestEnqueuedResponse, IngestStatusResponse
from app.database.postgres import get_db, AsyncSessionLocal
from app.ingestion.pipeline import ingest_document
from app.ingestion.moodle_sync import sync_moodle_knowledge_base
from app.config.settings import get_settings
import httpx
from app.api.auth import get_current_user, User
from app.worker import (
    worker as _api_worker,  # shared Worker instance for status_by_id (read-only)
    ingest_text_task,
    sync_moodle_task,
    dummy_task,
)

router = APIRouter()
settings = get_settings()

# ── streaq note ──────────────────────────────────────────────────────────────
# streaq Task objects MUST be awaited for the enqueue to actually publish to
# Redis (Task.__await__ → _chain → publish_task). Unlike arq's enqueue_job()
# which is fire-and-forget, you need:
#     task = some_task.enqueue(...)
#     await task   # ← this is what writes to Redis
#
# For status polling, the shared `worker` instance is opened in the FastAPI
# lifespan (`app/main.py:lifespan`) so `worker.status_by_id()` /
# `worker.result_by_id()` are safe to call from any request handler without
# raising `StreaqError: Worker not initialized`. See
# https://streaq.readthedocs.io/en/latest/worker.html#task-related-functions.
@router.post(
    "/ingest",
    response_model=IngestEnqueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue a document for ingestion into the RAG knowledge base",
)
async def ingest(
    request: IngestRequest,
    current_user: User = Depends(get_current_user),
) -> IngestEnqueuedResponse:
    """
    Enqueue raw text for background ingestion. Returns 202 with a job_id
    that can be polled via GET /ingest/{job_id}. The job is processed by
    the streaq worker so a 200 KB body no longer ties up the API process
    or a Postgres session.
    """
    logger.info(
        "Ingestion request received",
        title=request.title,
        source=request.source,
        text_len=len(request.text),
        user=current_user.username,
    )

    task = ingest_text_task.enqueue(
        text=request.text,
        title=request.title,
        source=request.source,
        metadata=request.metadata,
    )
    await task
    logger.info("Ingestion enqueued", job_id=task.id, text_len=len(request.text))
    return IngestEnqueuedResponse(job_id=task.id, status="queued")


@router.get(
    "/ingest/{job_id}",
    response_model=IngestStatusResponse,
    summary="Get the status of an enqueued ingestion job",
)
async def ingest_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
) -> IngestStatusResponse:
    """Poll an ingestion job's status. 'complete' carries the document_id."""
    # The shared worker instance is initialized by the FastAPI lifespan
    # (`async with worker:` in app/main.py), so status_by_id/result_by_id
    # are safe to call here without a per-request async-context enter.
    streaq_status = await _api_worker.status_by_id(job_id)
    if streaq_status == TaskStatus.DONE:
        # Result is already in Redis (DONE state proved it). timeout=0 skips
        # the pubsub wait. If we hit a TTL race, fall through to "queued".
        try:
            result = await _api_worker.result_by_id(job_id, timeout=timedelta(0))
        except TimeoutError:
            return IngestStatusResponse(job_id=job_id, status="queued")
        if result.success:
            result_dict = result.result if isinstance(result.result, dict) else None
            return IngestStatusResponse(
                job_id=job_id,
                status="complete",
                document_id=result_dict.get("document_id") if result_dict else None,
                chunks_count=result_dict.get("chunks_count") if result_dict else None,
                total_tokens=result_dict.get("total_tokens") if result_dict else None,
            )
        return IngestStatusResponse(
            job_id=job_id,
            status="failed",
            error=str(result.exception),
        )
    if streaq_status == TaskStatus.RUNNING:
        return IngestStatusResponse(job_id=job_id, status="in_progress")
    # QUEUED, SCHEDULED, or NOT_FOUND (TTL'd/never-existed) — same caller
    # response as the arq path: report "queued" optimistically.
    return IngestStatusResponse(job_id=job_id, status="queued")


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
    summary="Sync Moodle AI_Knowledge_Base course to Qdrant",
)
async def moodle_sync(
    request: MoodleSyncRequest,
    current_user: User = Depends(get_current_user),
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
    2. Downloads all `.md` files from the specified Moodle course
    3. Parses YAML frontmatter for metadata (department, topic, course_id, course_name)
    4. Splits content by markdown headers
    5. Embeds and upserts to Qdrant `Knowledge_Base` collection
    6. Skips unchanged files (unless force_reingest=true)
    """
    logger.info(f"Moodle sync triggered by user: {current_user.username}")

    task = sync_moodle_task.enqueue(
        course_id=request.course_id,
        target_sections=request.target_sections,
        force_reingest=request.force_reingest,
    )
    await task

    return MoodleSyncResponse(
        message="Moodle sync task has been successfully enqueued to the persistent background worker."
    )


@router.post("/test/dummy-task", summary="Enqueue a dummy task to verify the worker")
async def enqueue_dummy_task(
    name: str = "Tester",
    current_user: User = Depends(get_current_user),
):
    """
    Enqueue a dummy task to verify that the streaq worker is running correctly.
    """
    logger.info(f"Dummy task enqueued by user: {current_user.username}")
    task = dummy_task.enqueue(name=name)
    await task
    return {"message": f"Dummy task enqueued for {name}", "job_id": task.id}


@router.get("/moodle/sections", summary="Get Moodle course sections")
async def get_moodle_sections(
    course_id: int = 3,
    current_user: User = Depends(get_current_user),
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
