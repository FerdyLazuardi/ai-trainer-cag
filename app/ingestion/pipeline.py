"""
Full document ingestion pipeline using LlamaIndex.
Orchestrates: load → chunk → embed → store Qdrant → store PostgreSQL metadata.
"""
import hashlib
import uuid
from dataclasses import dataclass
import frontmatter

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.config.embedding_config import ensure_llamaindex_configured
from app.database.models import Chunk, Document
from app.database.qdrant_client import get_qdrant_client
from app.utils.token_counter import count_tokens

from llama_index.core import Document as LlamaDocument
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore

settings = get_settings()


@dataclass
class IngestionResult:
    document_id: str
    chunks_count: int
    total_tokens: int


async def ingest_document(
    text: str,
    session: AsyncSession,
    metadata: dict | None = None,
    title: str = "",
    source: str = "",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> IngestionResult:
    """
    Full ingestion pipeline for a single document using LlamaIndex.
    """
    meta = metadata or {}
    
    # Extract YAML Frontmatter if present
    try:
        parsed = frontmatter.loads(text)
        if parsed.metadata:
            # Flatten keywords into a comma-separated string if it's a list
            # so LlamaIndex metadata doesn't complain about list types
            extracted_meta = parsed.metadata
            if "keywords" in extracted_meta and isinstance(extracted_meta["keywords"], list):
                extracted_meta["keywords"] = ", ".join(extracted_meta["keywords"])
            meta.update(extracted_meta)
            logger.info("Extracted YAML frontmatter", keys=list(extracted_meta.keys()))
            # Use clean text without YAML for hashing and chunking
            text = parsed.content
    except Exception as e:
        logger.warning("Failed to parse frontmatter", error=str(e))
        
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    document_id = str(uuid.uuid4())

    # ── 1. Create Document record ──────────────────────────────────────────
    doc = Document(
        id=document_id,
        title=title or source or "Untitled",
        source=source,
        content_hash=content_hash,
        metadata_=meta,
        ingestion_state="processing",
    )
    session.add(doc)
    await session.flush()
    logger.info("Document created", document_id=document_id, source=source)

    try:
        # ── 2. Configure LlamaIndex ───────────────────────────────────────
        ensure_llamaindex_configured(chunk_size, chunk_overlap)
        
        # Build Document
        # ponytail: only pass metadata fields we actually need in Qdrant payload
        # (avoids leaking admin-request baggage like department/topic/course_id)
        _SAFE_META_KEYS = {"course_name", "keywords", "section_name"}
        _clean_meta = {k: v for k, v in meta.items() if k in _SAFE_META_KEYS}
        llama_doc = LlamaDocument(
            text=text,
            doc_id=document_id,
            metadata={
                "document_id": document_id,
                "source": source,
                "title": title,
                **_clean_meta
            }
        )
        
        # ── 2.1 Build Nodes (Chunks) ─────────────────────────────────────
        # If it's markdown, we use MarkdownNodeParser to split by headers (H1, H2, H3)
        # Otherwise, we fallback to the global Settings.text_splitter
        is_markdown = source.lower().endswith(".md") or text.strip().startswith("#")

        if is_markdown:
            parser = MarkdownNodeParser()
            nodes = parser.get_nodes_from_documents([llama_doc])
            # ── Hierarchical linking (H3 ↔ H2 parent/children) ────────────
            # After MarkdownNodeParser splits by headers, link H3 children to
            # their H2 parent and vice versa. This enables the retriever to
            # expand search results hierarchically: when an H3 is retrieved,
            # its H2 parent and sibling H3s are also fetched, and when an H2
            # is retrieved, its H3 children are included.
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
            # Back-fill child_chunk_ids on each H2 node
            for node in nodes:
                if node.node_id in _h2_children and _h2_children[node.node_id]:
                    node.metadata["child_chunk_ids"] = _h2_children[node.node_id]
            logger.info("Text parsed via MarkdownNodeParser (Headers)", document_id=document_id)
        else:
            nodes = Settings.text_splitter.get_nodes_from_documents([llama_doc])
            logger.info("Text chunked via TokenTextSplitter", document_id=document_id)

        if not nodes:
            doc.ingestion_state = "failed"
            logger.warning("No chunks produced", document_id=document_id)
            return IngestionResult(document_id=document_id, chunks_count=0, total_tokens=0)
            
        logger.info("Text chunked via LlamaIndex", document_id=document_id, chunks=len(nodes))

        # ── 3. Initialize Qdrant as LlamaIndex VectorStore ────────────────
        qdrant = get_qdrant_client()
        vector_store = qdrant.get_vector_store(settings.qdrant_kb_collection, enable_hybrid=True)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # ── 4. Embed & Upsert (via LlamaIndex async integration) ──────────
        # Instead of VectorStoreIndex.from_documents which is sync, we construct async index
        index = VectorStoreIndex(
            nodes=[], 
            storage_context=storage_context,
            show_progress=False
        )
        
        # Async generation of embeddings and insertion into Qdrant natively
        await index.ainsert_nodes(nodes)
        
        logger.info("Qdrant upserted (Hybrid Search Active)", document_id=document_id, points=len(nodes))

        # ── 5. Store Chunk metadata in PostgreSQL ─────────────────────────
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
                    metadata_={**meta, "source": source},
                )
            )

        # ── 6. Update Document state ──────────────────────────────────────
        doc.ingestion_state = "completed"
        doc.total_chunks = len(nodes)
        await session.flush()

        logger.info(
            "Ingestion complete",
            document_id=document_id,
            chunks=len(nodes),
            total_tokens=total_tokens,
        )
        return IngestionResult(
            document_id=document_id,
            chunks_count=len(nodes),
            total_tokens=total_tokens,
        )

    except Exception as exc:
        doc.ingestion_state = "failed"
        await session.flush()
        logger.error("Ingestion failed", document_id=document_id, error=str(exc))
        raise exc
