"""
POST /ingest endpoint — accepts raw text and triggers the ingestion pipeline.
POST /ingest/moodle/sync — triggers the Moodle AI_Knowledge_Base sync.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import IngestRequest, IngestResponse
from app.database.postgres import get_db, AsyncSessionLocal
from app.ingestion.pipeline import ingest_document
from app.ingestion.moodle_sync import sync_moodle_knowledge_base

router = APIRouter()

# ... (keep existing ingest endpoint) ...

@router.post("/ingest", response_model=IngestResponse, summary="Ingest a document into the RAG system")
async def ingest(
    request: IngestRequest,
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    """
    Ingest raw text into the RAG knowledge base.
    """
    logger.info(
        "Ingestion request received",
        title=request.title,
        source=request.source,
        text_len=len(request.text),
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


async def run_moodle_sync_background(course_id: int | None, target_sections: list[str] | None, force_reingest: bool):
    """Background task to run the moodle sync with its own database session."""
    logger.info("Starting background Moodle sync task", force_reingest=force_reingest)
    try:
        async with AsyncSessionLocal() as session:
            summary = await sync_moodle_knowledge_base(
                session=session,
                course_id=course_id,
                target_sections=target_sections,
                force_reingest=force_reingest
            )
            # Commit any changes made by the sync
            await session.commit()
            
            logger.info(f"Background Moodle sync completed: {summary}")
    except Exception as exc:
        logger.error(f"Background Moodle sync failed: {exc}")


@router.post(
    "/ingest/moodle/sync",
    response_model=MoodleSyncResponse,
    summary="Sync Moodle AI_Knowledge_Base course to Qdrant",
)
async def moodle_sync(
    request: MoodleSyncRequest,
    background_tasks: BackgroundTasks,
) -> MoodleSyncResponse:
    """
    Trigger background sync of Moodle course (default: course_id=3 AI_Knowledge_Base).
    
    **Quick Start:** Just click "Execute" to sync all markdown files from AI_Knowledge_Base course.
    
    **Parameters:**
    - `course_id`: Moodle course ID (default: 3 = AI_Knowledge_Base)
    - `target_sections`: Optional list of section names to sync (leave empty for all)
    - `force_reingest`: Set to `true` to re-process files even if unchanged
    
    **What it does:**
    1. Downloads all `.md` files from the specified Moodle course
    2. Parses YAML frontmatter for metadata (department, topic, course_id, course_name)
    3. Splits content by markdown headers
    4. Embeds and upserts to Qdrant `Knowledge_Base` collection
    5. Skips unchanged files (unless force_reingest=true)
    """
    
    background_tasks.add_task(
        run_moodle_sync_background,
        course_id=request.course_id,
        target_sections=request.target_sections,
        force_reingest=request.force_reingest
    )

    return MoodleSyncResponse(
        message="Moodle sync task has been successfully enqueued and is running in the background."
    )
