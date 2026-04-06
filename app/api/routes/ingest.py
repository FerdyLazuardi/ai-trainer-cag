"""
POST /ingest endpoint — accepts raw text and triggers the ingestion pipeline.
POST /ingest/moodle/sync — triggers the Moodle AI_Knowledge_Base sync.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from arq import create_pool
from arq.connections import RedisSettings, ArqRedis

from app.api.schemas import IngestRequest, IngestResponse
from app.database.postgres import get_db, AsyncSessionLocal
from app.ingestion.pipeline import ingest_document
from app.ingestion.moodle_sync import sync_moodle_knowledge_base
from app.config.settings import get_settings
import httpx
from app.api.auth import get_current_user, User

router = APIRouter()
settings = get_settings()

# ── ARQ Singleton Pool ────────────────────────────────────────────────────────
# A single, long-lived pool shared across all requests instead of creating a
# new TCP connection per enqueue call (prevents Redis connection exhaustion).
_arq_pool: ArqRedis | None = None

async def get_arq_redis() -> ArqRedis:
    """Return the shared singleton ARQ Redis pool, creating it on first call."""
    global _arq_pool
    if _arq_pool is None:
        _arq_pool = await create_pool(RedisSettings(
            host=settings.redis_host,
            port=settings.redis_port,
            database=settings.redis_db,
            password=settings.redis_password if settings.redis_password else None,
        ))
        logger.info("ARQ Redis pool created (singleton)")
    return _arq_pool

# ... (keep existing ingest endpoint) ...
@router.post("/ingest", response_model=IngestResponse, summary="Ingest a document into the RAG system")
async def ingest(
    request: IngestRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> IngestResponse:
    """
    Ingest raw text into the RAG knowledge base.
    """
    logger.info(
        "Ingestion request received",
        title=request.title,
        source=request.source,
        text_len=len(request.text),
        user=current_user.username,
    )

    try:
        result = await ingest_document(
            text=request.text,
            session=db,
            metadata=request.metadata,
            title=request.title,
            source=request.source,
        )
    except Exception as exc:
        logger.error("Ingestion failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    logger.info(
        "Ingestion complete",
        document_id=result.document_id,
        chunks=result.chunks_count,
        tokens=result.total_tokens,
    )

    return IngestResponse(
        document_id=result.document_id,
        chunks_count=result.chunks_count,
        total_tokens=result.total_tokens,
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
    summary="Sync Moodle AI_Knowledge_Base course to Qdrant",
)
async def moodle_sync(
    request: MoodleSyncRequest,
    current_user: User = Depends(get_current_user),
) -> MoodleSyncResponse:
    """
    Trigger background sync of Moodle course (default: course_id=3 AI_Knowledge_Base) via arq worker.

    **Quick Start:** Just click "Execute" to sync all markdown files from AI_Knowledge_Base course.

    **Parameters:**
    - `course_id`: Moodle course ID (default: 3 = AI_Knowledge_Base)
    - `target_sections`: Optional list of section names to sync (leave empty for all)
    - `force_reingest`: Set to `true` to re-process files even if unchanged

    **What it does:**
    1. Enqueues a persistent job to the arq worker.
    2. Downloads all `.md` files from the specified Moodle course
    3. Parses YAML frontmatter for metadata (department, topic, course_id, course_name)
    4. Splits content by markdown headers
    5. Embeds and upserts to Qdrant `Knowledge_Base` collection
    6. Skips unchanged files (unless force_reingest=true)
    """
    logger.info(f"Moodle sync triggered by user: {current_user.username}")

    redis = await get_arq_redis()
    await redis.enqueue_job(
        'sync_moodle_task',
        course_id=request.course_id,
        target_sections=request.target_sections,
        force_reingest=request.force_reingest
    )
    # Note: do NOT close the pool — it's a shared singleton

    return MoodleSyncResponse(
        message="Moodle sync task has been successfully enqueued to the persistent background worker."
    )


@router.post("/test/dummy-task", summary="Enqueue a dummy task to verify the worker")
async def enqueue_dummy_task(
    name: str = "Tester",
    current_user: User = Depends(get_current_user),
):
    """
    Enqueue a dummy task to verify that the arq worker is running correctly.
    """
    logger.info(f"Dummy task enqueued by user: {current_user.username}")
    redis = await get_arq_redis()
    job = await redis.enqueue_job('dummy_task', name=name)
    # Note: do NOT close the pool — it's a shared singleton
    return {"message": f"Dummy task enqueued for {name}", "job_id": job.job_id}


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
