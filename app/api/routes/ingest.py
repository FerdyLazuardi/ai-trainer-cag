"""
POST /ingest endpoint — accepts raw text and triggers the ingestion pipeline.
POST /ingest/moodle/sync — triggers the Moodle AI_Knowledge_Base sync.
"""
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import IngestRequest, IngestResponse
from app.database.postgres import get_db
from app.ingestion.pipeline import ingest_document
from app.ingestion.moodle_sync import sync_moodle_knowledge_base

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse, summary="Ingest a document into the RAG system")
async def ingest(
    request: IngestRequest,
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    """
    Ingest raw text into the RAG knowledge base.

    Steps:
    1. Chunk the document (512 tokens / 50 overlap)
    2. Generate embeddings via LlamaIndex OpenAI
    3. Upsert to Qdrant
    4. Store metadata in PostgreSQL
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


class MoodleSyncResponse(BaseModel):
    files_processed: int
    chunks_ingested: int
    files_skipped: int
    errors: list[str]


@router.post(
    "/ingest/moodle/sync",
    response_model=MoodleSyncResponse,
    summary="Sync specific sections from AI_Knowledge_Base (Course 3) into Qdrant",
)
async def moodle_sync(
    db: AsyncSession = Depends(get_db),
) -> MoodleSyncResponse:
    """
    Trigger a targeted sync of the Moodle course (Course ID 3).
    Specifically syncs 'training client protection' and 'anti harassment' sections.

    - Downloads all `.md` files from targeted sections.
    - Parses Markdown frontmatter for metadata (department, topic, course_id, course_name).
    - Splits content by headers (H1/H2/H3) and upserts to `Knowledge_Base` Qdrant collection.
    - Skips files whose content has not changed since last sync.
    """
    try:
        summary = await sync_moodle_knowledge_base(
            session=db,
            course_id=3,
            target_sections=["client protection", "anti harassment"]
        )
    except Exception as exc:
        logger.error("Moodle sync failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Moodle sync failed: {exc}") from exc

    return MoodleSyncResponse(**summary)
