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
    Batched: one Qdrant delete + one PG bulk delete + one cache flush per course
    instead of N round-trips per stale document.

    Scoping: a KB doc is identified by `metadata->>'moodle_course_id'` (the
    actual synced course), NOT `course_id` (which comes from frontmatter and is
    usually unrelated — 10/2205/etc — so the old filter almost never matched and
    cleanup silently never ran, letting renamed files accumulate as duplicates).
    A `.md`-source fallback catches legacy docs ingested before the marker
    existed. Askfer/portfolio docs are safe: their sources are http(s)://
    or portfolio:// — never `.md` — and they carry no moodle_course_id.
    """
    from sqlalchemy import select
    from sqlalchemy.sql import text

    # Scope to docs synced from THIS Moodle course (marker), plus legacy KB docs
    # (source ends in .md) that predate the marker. The .md filter can never
    # match a portfolio doc (http/portfolio:// sources), so Askfer stays untouched.
    stmt = select(Document).where(
        text(
            "(metadata->>'moodle_course_id' = :cid"
            " OR (metadata->>'moodle_course_id' IS NULL AND lower(source) LIKE '%.md'))"
        )
    ).params(cid=str(course_id))

    result = await session.execute(stmt)
    docs = result.scalars().all()

    stale_docs = [d for d in docs if d.source not in current_filenames]
    if not stale_docs:
        return

    stale_ids = [str(d.id) for d in stale_docs]
    logger.info(
        "Deleting stale documents in batch (not in Moodle anymore)",
        course_id=course_id,
        count=len(stale_ids),
        sources=[d.source for d in stale_docs],
    )

    # 1. Single Qdrant delete via MatchAny — one round-trip for all stale docs.
    qdrant = get_qdrant_client()
    try:
        await qdrant.client.delete(
            collection_name=settings.qdrant_kb_collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[qdrant_models.FieldCondition(
                        key="document_id",
                        match=qdrant_models.MatchAny(any=stale_ids),
                    )]
                )
            ),
        )
    except Exception as e:
        logger.warning(f"Failed to batch-delete stale Qdrant points for course {course_id}", error=str(e))

    # 2. PG: delete docs in one go (chunks cascade).
    for d in stale_docs:
        await session.delete(d)
    await session.flush()

    # 3. Single cache flush for the course (deduplicated).
    from app.utils.cache import flush_cache_by_course
    await flush_cache_by_course(course_id)


async def _ingest_markdown(
    raw_content: str,
    filename: str,
    session: AsyncSession,
    moodle_course_id: int,
    force_reingest: bool = False,
    section_name: str = "",
) -> int:
    """
    Parse frontmatter, split by headers, embed, upsert to Knowledge_Base.
    Returns number of chunks ingested (0 if file was unchanged and not forced).

    `section_name` is the Moodle section the file lives in (e.g. "Product
    Amartha"). It is stored in metadata so TOPIC_LIST can group multiple files
    in one section under a single topic — manual section structure in Moodle
    becomes the topic taxonomy. Empty for files in unnamed/section-0; the
    TOPIC_LIST query falls back to course_name in that case.
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
    # ponytail: only keep fields that belong in Qdrant payload.
    # department/topic/course_id/moodle_course_id are legacy cruft that
    # leaked from Moodle frontmatter and caused metadata bloat + confusion.
    metadata = {
        "course_name": frontmatter.get("course_name", ""),
        "section_name": (section_name or "").strip(),
        "source": filename,
        # Keep moodle_course_id for stale-cleanup scoping (line 247-270)
        "moodle_course_id": moodle_course_id,
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
        # Cache flush moved to end of function — after all DB writes are staged — to
        # avoid flushing cache mid-transaction when a later step could still fail.
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
    # ponytail: only pass metadata fields we actually need in Qdrant payload
    # `source` MUST be here — without it the filename never reaches the Qdrant
    # payload, so _extract_sources (chat.py) sees "Unknown" for every KB chunk
    # and the UI's source list renders empty for all Moodle-sourced answers.
    _SAFE_META_KEYS = {"course_name", "keywords", "section_name", "source"}
    _clean_meta = {k: v for k, v in metadata.items() if k in _SAFE_META_KEYS}
    llama_doc = LlamaDocument(
        text=body,
        doc_id=document_id,
        metadata={
            "document_id": document_id,
            "title": title,
            **_clean_meta,
        },
    )

    # ── 3. Split by Markdown Headers (H1 / H2 / H3) ─────────────────────
    ensure_llamaindex_configured(chunk_size=512, chunk_overlap=50)
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents([llama_doc])

    # Filter out empty or whitespace-only nodes
    original_node_count = len(nodes)
    nodes = [n for n in nodes if n.text and n.text.strip()]  # type: ignore[attr-defined]  # TextNode at runtime
    if len(nodes) < original_node_count:
        logger.info(f"Filtered out {original_node_count - len(nodes)} empty nodes", source=filename)

    if not nodes:
        logger.warning("No nodes produced from Markdown after filtering", source=filename)
        return 0

    # Attach frontmatter metadata to every node
    for node in nodes:
        node.metadata.update(_clean_meta)

    # ── Hierarchical linking (H3 ↔ H2 parent/children) ────────────────────
    # Mirrors app/ingestion/pipeline.py. Without this, moodle-ingested chunks
    # carry no parent_chunk_id/child_chunk_ids, so the retriever's hierarchical
    # expansion (sibling/parent fetch) never fires — multi-topic retrieval stays
    # flat at top-k. MarkdownNodeParser splits by H1/H2/H3 already; here we link.
    import re as _re_hier
    _last_h2_id = None
    _h2_children: dict[str, list[str]] = {}
    for node in nodes:
        text = getattr(node, "text", "") or ""
        first_line = text.split("\n", 1)[0].strip()
        is_h3 = bool(_re_hier.match(r"^###\s+", first_line))
        is_h2 = bool(_re_hier.match(r"^##\s+", first_line))
        if is_h2:
            _last_h2_id = node.node_id
            _h2_children.setdefault(node.node_id, [])
        elif is_h3 and _last_h2_id:
            node.metadata["parent_chunk_id"] = _last_h2_id
            _h2_children.setdefault(_last_h2_id, []).append(node.node_id)
    for node in nodes:
        if node.node_id in _h2_children and _h2_children[node.node_id]:
            node.metadata["child_chunk_ids"] = _h2_children[node.node_id]

    logger.info(
        "Markdown parsed into nodes",
        source=filename,
        nodes=len(nodes),
        course_name=metadata.get("course_name", ""),
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

    vector_store = qdrant.get_vector_store(settings.qdrant_kb_collection, enable_hybrid=True)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logger.info("Starting Qdrant node insertion", points=len(nodes), collection=settings.qdrant_kb_collection)
    
    try:
        index = VectorStoreIndex(nodes=[], storage_context=storage_context, show_progress=False)
        await index.ainsert_nodes(nodes)
        logger.info("Qdrant ainsert_nodes completed successfully")
        
        # Verify immediately
        info = await qdrant.client.get_collection(settings.qdrant_kb_collection)
        logger.info(f"Verified point count in {settings.qdrant_kb_collection}: {info.points_count}")
    except Exception as e:
        logger.error(f"Failed to insert into Qdrant: {e}")
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
        tokens = count_tokens(node.text)  # type: ignore[attr-defined]  # TextNode at runtime
        total_tokens += tokens
        session.add(
            Chunk(
                id=node.node_id,
                document_id=document_id,
                chunk_index=i,
                text=node.text,  # type: ignore[attr-defined]  # TextNode at runtime
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

    # Flush the query cache AFTER all DB writes are staged (post-flush, pre-commit).
    # This is safer than mid-function: if anything above raised an exception,
    # we never reach here, so the cache is only invalidated when ingestion succeeded.
    # The commit itself happens in the caller (sync_moodle_task / worker.py).
    try:
        from app.utils.cache import flush_cache_by_course
        await flush_cache_by_course(moodle_course_id)
    except Exception as _cache_exc:
        logger.warning(f"Cache flush after ingest failed (non-fatal): {_cache_exc}")

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
    summary: dict[str, Any] = {"files_processed": 0, "chunks_ingested": 0, "files_skipped": 0, "errors": []}

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
                # Moodle returns section names HTML-encoded (e.g. "A &amp; B").
                # Decode so "&amp;" is stored as "&" — otherwise the entity leaks
                # into TOPIC_LIST labels and source citations.
                import html as _html
                section_name_raw = _html.unescape(section.get("name", "") or "")
                section_name = section_name_raw.lower().strip()

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

                            chunks = await _ingest_markdown(
                                raw_text, filename, session, cid, force_reingest,
                                section_name=section_name_raw,
                            )
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
