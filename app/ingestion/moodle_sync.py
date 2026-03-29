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
7. On re-ingest (content changed), deletes stale chunks from Qdrant + PG first.
"""
import hashlib
import uuid
from typing import Any

import httpx
import yaml
from loguru import logger
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import models as qdrant_models

from app.config.settings import get_settings
from app.config.embedding_config import ensure_llamaindex_configured
from app.database.models import Chunk, Document
from app.database.qdrant_client import get_qdrant_client
from app.utils.token_counter import count_tokens

from llama_index.core import Document as LlamaDocument, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore

settings = get_settings()

# ─── Moodle Course ─────────────────────────────────────────────────────────
MOODLE_COURSE_NAME = "ai-kb"


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
    # Try searching by fullname first (as it's usually what users think of)
    for field in ["fullname", "shortname"]:
        logger.info(f"Searching for course by {field}: {MOODLE_COURSE_NAME}")
        resp = await client.post(
            f"{settings.moodle_api_url}/webservice/rest/server.php",
            data={
                "wstoken": settings.moodle_api_token,
                "wsfunction": "core_course_get_courses_by_field",
                "moodlewsrestformat": "json",
                "field": field,
                "value": MOODLE_COURSE_NAME,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"Moodle response for {field}: {data}")
        if isinstance(data, dict) and "exception" in data:
            logger.warning(f"Moodle search error for {field}: {data['message']}")
            continue
            
        courses = data.get("courses", [])
        if courses:
            cid = courses[0]["id"]
            logger.info(f"Found course via {field}", course_id=cid, name=MOODLE_COURSE_NAME)
            return cid

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
    data = resp.json()
    if isinstance(data, dict) and "exception" in data:
        raise ValueError(f"Moodle error: {data.get('message', 'Unknown error')} ({data.get('exception')})")
    return data


async def _download_file(client: httpx.AsyncClient, url: str) -> bytes:
    """Download a file from Moodle (token injected as query param)."""
    resp = await client.get(url, params={"token": settings.moodle_api_token})
    resp.raise_for_status()
    return resp.content


# ─── Core ingestion per document ────────────────────────────────────────────

async def _delete_stale_documents(session: AsyncSession, course_id: int, current_filenames: list[str]):
    """
    Delete documents in DB/Qdrant for this course that are no longer present in Moodle.
    This ensures that when files are renamed or deleted in Moodle, they are removed from our RAG.
    """
    from sqlalchemy import select
    from sqlalchemy.sql import text

    # Find documents for this course_id (from metadata JSON)
    # Note: we check both string and int versions to be safe with different JSON encoders
    from sqlalchemy import or_
    stmt = select(Document).where(
        or_(
            text("metadata->>'course_id' = :cid_str"),
            text("metadata->>'course_id' = :cid_int")
        )
    ).params(cid_str=str(course_id), cid_int=str(course_id))
    
    result = await session.execute(stmt)
    docs = result.scalars().all()

    for doc in docs:
        if doc.source not in current_filenames:
            logger.info("Deleting stale document (not in Moodle anymore)", source=doc.source, course_id=course_id)
            
            # 1. Delete from Qdrant
            qdrant = get_qdrant_client()
            try:
                await qdrant.client.delete(
                    collection_name=settings.qdrant_kb_collection,
                    points_selector=qdrant_models.FilterSelector(
                        filter=qdrant_models.Filter(
                            must=[qdrant_models.FieldCondition(
                                key="document_id",
                                match=qdrant_models.MatchValue(value=str(doc.id)),
                            )]
                        )
                    ),
                )
            except Exception as e:
                logger.warning(f"Failed to delete stale Qdrant points for {doc.source}", error=str(e))

            # 2. Delete from PG (chunks will be deleted by FK cascade)
            await session.delete(doc)
            
            # 3. Flush course cache
            from app.utils.cache import flush_cache_by_course
            await flush_cache_by_course(course_id)
    
    await session.flush()


async def _ingest_markdown(
    raw_content: str,
    filename: str,
    session: AsyncSession,
    moodle_course_id: int,
    force_reingest: bool = False,
) -> int:
    """
    Parse frontmatter, split by headers, embed, upsert to Knowledge_Base.
    Returns number of chunks ingested (0 if file was unchanged and not forced).
    """
    content_hash = hashlib.sha256(raw_content.encode()).hexdigest()

    # Check if already ingested and unchanged
    from sqlalchemy import select
    result = await session.execute(
        select(Document).where(Document.source == filename)
    )
    existing: Document | None = result.scalars().first()

    if existing and existing.content_hash == content_hash and not force_reingest:
        logger.info("Skipping unchanged file", source=filename)
        return 0
    
    if existing and force_reingest:
        logger.info("Force re-ingesting file (force_reingest=True)", source=filename)

    # ── 1. Parse frontmatter + body ──────────────────────────────────────
    frontmatter, body = _parse_frontmatter(raw_content)
    
    # Use course_id from frontmatter, fallback to moodle_course_id if not present
    metadata = {
        "department": frontmatter.get("department", "Global"),
        "topic": frontmatter.get("topic", ""),
        "course_id": frontmatter.get("course_id", moodle_course_id),
        "course_name": frontmatter.get("course_name", ""),
        "source": filename,
    }

    document_id = str(existing.id) if existing else str(uuid.uuid4())
    title = frontmatter.get("course_name", filename)

    if existing:
        # ── Delete stale chunks from Qdrant + PostgreSQL ─────────────────
        qdrant = get_qdrant_client()
        try:
            await qdrant.client.delete(
                collection_name=settings.qdrant_kb_collection,
                points_selector=qdrant_models.FilterSelector(
                    filter=qdrant_models.Filter(
                        must=[qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=document_id),
                        )]
                    )
                ),
            )
            logger.info("Deleted stale Qdrant points", document_id=document_id)
        except Exception as e:
            logger.warning("Failed to delete stale Qdrant points", error=str(e))

        await session.execute(
            delete(Chunk).where(Chunk.document_id == document_id)
        )
        await session.flush()
        logger.info("Deleted stale PostgreSQL chunks", document_id=document_id)

        # Update existing document record
        existing.ingestion_state = "processing"
        existing.content_hash = content_hash
        existing.metadata_ = metadata # Ensure metadata is updated with new course_id if it changed
        
        # Flush course cache
        from app.utils.cache import flush_cache_by_course
        await flush_cache_by_course(moodle_course_id)
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
    ensure_llamaindex_configured(chunk_size=512, chunk_overlap=50)
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents([llama_doc])

    # Filter out empty or whitespace-only nodes
    original_node_count = len(nodes)
    nodes = [n for n in nodes if n.text and n.text.strip()]
    if len(nodes) < original_node_count:
        logger.info(f"Filtered out {original_node_count - len(nodes)} empty nodes", source=filename)

    if not nodes:
        logger.warning("No nodes produced from Markdown after filtering", source=filename)
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

    # Log collection info before insertion
    try:
        info = await qdrant.client.get_collection(settings.qdrant_kb_collection)
        logger.info(f"Knowledge_Base collection status BEFORE insertion: {info.points_count} points")
    except Exception as e:
        logger.warning(f"Could not get collection info: {e}")

    vector_store = QdrantVectorStore(
        aclient=qdrant.client,
        collection_name=settings.qdrant_kb_collection,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        dense_vector_name="text-dense",
        sparse_vector_name="text-sparse",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logger.info("Starting Qdrant node insertion", points=len(nodes), collection=settings.qdrant_kb_collection)
    print(f"DEBUG: Starting Qdrant node insertion into {settings.qdrant_kb_collection} with {len(nodes)} nodes", flush=True)
    
    try:
        index = VectorStoreIndex(nodes=[], storage_context=storage_context, show_progress=False)
        await index.ainsert_nodes(nodes)
        logger.info("Qdrant ainsert_nodes completed successfully")
        print("DEBUG: Qdrant ainsert_nodes completed successfully", flush=True)
        
        # Verify immediately
        info = await qdrant.client.get_collection(settings.qdrant_kb_collection)
        logger.info(f"Verified point count in {settings.qdrant_kb_collection}: {info.points_count}")
        print(f"DEBUG: Verified point count: {info.points_count}", flush=True)
    except Exception as e:
        logger.error(f"Failed to insert into Qdrant: {e}")
        print(f"DEBUG ERROR: Failed to insert into Qdrant: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise e

    logger.info(
        "Ingested to Knowledge_Base (Hybrid Search Active)",
        source=filename,
        collection=settings.qdrant_kb_collection,
        points=len(nodes),
    )

    # ── 5. Update PostgreSQL chunks ──────────────────────────────────────
    total_tokens = 0
    for i, node in enumerate(nodes):
        tokens = count_tokens(node.text)
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
    force_reingest: bool = False,
) -> dict[str, Any]:
    """
    Full sync of Moodle course `AI_Knowledge_Base` into Qdrant `Knowledge_Base`.

    Args:
        session: Database session.
        course_id: Explicit Moodle Course ID to sync (defaults to lookup by MOODLE_COURSE_NAME).
        target_sections: Optional list of specific section names to sync (case-insensitive).
        force_reingest: Force re-ingestion even if content hash matches.

    Returns summary dict with counts.
    """
    logger.info("Starting Moodle Knowledge Base sync", course_id=course_id, target_sections=target_sections, force_reingest=force_reingest)
    summary = {"files_processed": 0, "chunks_ingested": 0, "files_skipped": 0, "errors": []}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Find the course ID(s)
        if course_id is None:
            cid = await _find_kb_course_id(client)
            if cid is None:
                summary["errors"].append("Could not find AI_Knowledge_Base course on Moodle")
                return summary
            course_ids = [cid]
        else:
            course_ids = [course_id]

        logger.info("Found courses on Moodle to sync", course_count=len(course_ids))

        # Normalize target sections for case-insensitive matching if provided
        target_sections_lower = [ts.lower().strip() for ts in target_sections] if target_sections else None

        # 2. Get all sections and modules
        for cid in course_ids:
            try:
                sections = await _get_course_contents(client, cid)
            except Exception as e:
                logger.warning(f"Failed to get contents for course_id {cid}", error=str(e))
                continue

            # --- CLEANUP STEP ---
            # Collect all .md filenames currently in Moodle for this course
            moodle_filenames = []
            for section in sections:
                for module in section.get("modules", []):
                    for content in module.get("contents", []):
                        fname = content.get("filename", "")
                        if fname.lower().endswith(".md"):
                            moodle_filenames.append(fname)
            
            # Delete documents in our DB that are no longer in Moodle
            await _delete_stale_documents(session, cid, moodle_filenames)
            # --------------------

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

                            chunks = await _ingest_markdown(raw_text, filename, session, cid, force_reingest)
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
