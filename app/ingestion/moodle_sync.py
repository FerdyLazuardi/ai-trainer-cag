"""
Moodle → Qdrant  Knowledge-Base Synchronisation Script
========================================================
1. Fetches all sections/modules from the Moodle course `AI_Knowledge_Base`.
2. Downloads every `.md` file attached to each module.
3. Parses YAML frontmatter (department / topic / course_id / course_name).
4. Splits the document body by Markdown headers (MarkdownNodeParser).
5. Embeds each node and upserts it to the dedicated Qdrant `Knowledge_Base`
   collection, including the frontmatter payload.
6. Tracks `content_hash` in PostgreSQL so unchanged files are skipped.
"""
import hashlib
import io
import re
import uuid
from typing import Any

import httpx
import yaml
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.database.models import Chunk, Document
from app.database.qdrant_client import get_qdrant_client
from app.ingestion.pipeline import _init_llamaindex_settings

from llama_index.core import Document as LlamaDocument, Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore

settings = get_settings()

# ─── Moodle Course ─────────────────────────────────────────────────────────
MOODLE_COURSE_NAME = "AI_Knowledge_Base"


# ─── Internal helpers ───────────────────────────────────────────────────────

def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """
    Split a Markdown document into frontmatter (dict) and body (str).
    Returns (metadata_dict, body_text). If no frontmatter, metadata is empty.
    """
    frontmatter: dict = {}
    body = raw

    stripped = raw.lstrip()
    if stripped.startswith("---"):
        end = stripped.find("---", 3)
        if end != -1:
            yaml_block = stripped[3:end].strip()
            body = stripped[end + 3:].lstrip()
            try:
                frontmatter = yaml.safe_load(yaml_block) or {}
            except yaml.YAMLError as exc:
                logger.warning("YAML frontmatter parse error", error=str(exc))

    return frontmatter, body


async def _find_kb_course_id(client: httpx.AsyncClient) -> int | None:
    """Search Moodle for the AI_Knowledge_Base course and return its id."""
    resp = await client.post(
        f"{settings.moodle_api_url}/webservice/rest/server.php",
        data={
            "wstoken": settings.moodle_api_token,
            "wsfunction": "core_course_get_courses_by_field",
            "moodlewsrestformat": "json",
            "field": "shortname",
            "value": MOODLE_COURSE_NAME,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    courses = data.get("courses", [])
    if courses:
        return courses[0]["id"]
    logger.error("Course not found on Moodle", course=MOODLE_COURSE_NAME)
    return None


async def _get_course_contents(client: httpx.AsyncClient, course_id: int) -> list[dict]:
    """Fetch all sections and their modules from Moodle."""
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
    return resp.json()


async def _download_file(client: httpx.AsyncClient, url: str) -> bytes:
    """Download a file from Moodle (token injected as query param)."""
    resp = await client.get(url, params={"token": settings.moodle_api_token})
    resp.raise_for_status()
    return resp.content


# ─── Core ingestion per document ────────────────────────────────────────────

async def _ingest_markdown(
    raw_content: str,
    filename: str,
    session: AsyncSession,
) -> int:
    """
    Parse frontmatter, split by headers, embed, upsert to Knowledge_Base.
    Returns number of chunks ingested (0 if file was unchanged).
    """
    content_hash = hashlib.sha256(raw_content.encode()).hexdigest()

    # Check if already ingested and unchanged
    from sqlalchemy import select
    result = await session.execute(
        select(Document).where(Document.source == filename)
    )
    existing: Document | None = result.scalars().first()

    if existing and existing.content_hash == content_hash:
        logger.info("Skipping unchanged file", source=filename)
        return 0

    # ── 1. Parse frontmatter + body ──────────────────────────────────────
    frontmatter, body = _parse_frontmatter(raw_content)
    metadata = {
        "department": frontmatter.get("department", "Global"),
        "topic": frontmatter.get("topic", ""),
        "course_id": frontmatter.get("course_id"),
        "course_name": frontmatter.get("course_name", ""),
        "source": filename,
    }

    document_id = str(existing.id) if existing else str(uuid.uuid4())
    title = frontmatter.get("course_name", filename)

    if existing:
        # Mark old chunks as being replaced
        existing.ingestion_state = "processing"
        existing.content_hash = content_hash
    else:
        doc = Document(
            id=document_id,
            title=title,
            source=filename,
            content_hash=content_hash,
            metadata_=metadata,
            ingestion_state="processing",
        )
        session.add(doc)

    await session.flush()

    # ── 2. Build LlamaDocument ───────────────────────────────────────────
    llama_doc = LlamaDocument(
        text=body,
        doc_id=document_id,
        metadata={
            "document_id": document_id,
            "title": title,
            **metadata,
        },
    )

    # ── 3. Split by Markdown Headers (H1 / H2 / H3) ─────────────────────
    _init_llamaindex_settings(chunk_size=512, chunk_overlap=50)
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents([llama_doc])

    if not nodes:
        logger.warning("No nodes produced from Markdown", source=filename)
        return 0

    # Attach frontmatter metadata to every node
    for node in nodes:
        node.metadata.update(metadata)

    logger.info(
        "Markdown parsed into nodes",
        source=filename,
        nodes=len(nodes),
        department=metadata["department"],
        topic=metadata["topic"],
    )

    # ── 4. Embed & Upsert to Knowledge_Base collection ───────────────────
    qdrant = get_qdrant_client()
    await qdrant.ensure_kb_collection()  # Create if it doesn't exist yet

    vector_store = QdrantVectorStore(
        aclient=qdrant.client,
        collection_name=settings.qdrant_kb_collection,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        dense_vector_name="text-dense",
        sparse_vector_name="text-sparse",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=[], storage_context=storage_context, show_progress=False)
    await index.ainsert_nodes(nodes)

    logger.info(
        "Ingested to Knowledge_Base (Hybrid Search Active)",
        source=filename,
        collection=settings.qdrant_kb_collection,
        points=len(nodes),
    )

    # ── 5. Update PostgreSQL chunks ──────────────────────────────────────
    total_tokens = 0
    for i, node in enumerate(nodes):
        tokens = len(node.text) // 4
        total_tokens += tokens
        session.add(
            Chunk(
                id=node.node_id,
                document_id=document_id,
                chunk_index=i,
                text=node.text,
                token_count=tokens,
                qdrant_point_id=node.node_id,
                metadata_={**metadata, "header_path": node.metadata.get("Header_1", "")},
            )
        )

    # Update document state
    if existing:
        existing.ingestion_state = "completed"
        existing.total_chunks = len(nodes)
    else:
        doc.ingestion_state = "completed"
        doc.total_chunks = len(nodes)

    await session.flush()
    logger.info("Ingestion complete", source=filename, chunks=len(nodes), tokens=total_tokens)
    return len(nodes)


# ─── Main sync entrypoint ────────────────────────────────────────────────────

async def sync_moodle_knowledge_base(
    session: AsyncSession,
    course_id: int | None = None,
    target_sections: list[str] | None = None,
) -> dict[str, Any]:
    """
    Full sync of Moodle course `AI_Knowledge_Base` into Qdrant `Knowledge_Base`.

    Args:
        session: Database session.
        course_id: Explicit Moodle Course ID to sync (defaults to lookup by MOODLE_COURSE_NAME).
        target_sections: Optional list of specific section names to sync (case-insensitive).

    Returns summary dict with counts.
    """
    logger.info("Starting Moodle Knowledge Base sync", course_id=course_id, target_sections=target_sections)
    summary = {"files_processed": 0, "chunks_ingested": 0, "files_skipped": 0, "errors": []}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Find the course ID if not provided
        if course_id is None:
            course_id = await _find_kb_course_id(client)
            if course_id is None:
                summary["errors"].append(f"Course '{MOODLE_COURSE_NAME}' not found on Moodle.")
                return summary

        logger.info("Found course on Moodle to sync", course_id=course_id)

        # Normalize target sections for case-insensitive matching if provided
        target_sections_lower = [ts.lower().strip() for ts in target_sections] if target_sections else None

        # 2. Get all sections and modules
        sections = await _get_course_contents(client, course_id)

        for section in sections:
            section_name = section.get("name", "").lower().strip()
            
            # Filter by target_sections if provided
            if target_sections_lower and section_name not in target_sections_lower:
                logger.debug("Skipping section not in targets", section=section_name)
                continue

            for module in section.get("modules", []):
                for content in module.get("contents", []):
                    filename: str = content.get("filename", "")
                    file_url: str = content.get("fileurl", "")

                    if not filename.lower().endswith(".md"):
                        continue

                    try:
                        logger.info("Downloading file", section=section_name, filename=filename)
                        raw_bytes = await _download_file(client, file_url)
                        raw_text = raw_bytes.decode("utf-8", errors="replace")

                        chunks = await _ingest_markdown(raw_text, filename, session)
                        if chunks == 0:
                            summary["files_skipped"] += 1
                        else:
                            summary["files_processed"] += 1
                            summary["chunks_ingested"] += chunks

                    except Exception as exc:
                        logger.error("Failed to process file", filename=filename, error=str(exc))
                        summary["errors"].append(f"{filename}: {exc}")

    logger.info("Moodle KB sync complete", **summary)
    return summary
