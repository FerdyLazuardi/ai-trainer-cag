"""
Full document ingestion pipeline using LlamaIndex.
Orchestrates: load → chunk → embed → store Qdrant → store PostgreSQL metadata.
"""
import hashlib
import uuid
from dataclasses import dataclass

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import get_settings
from app.database.models import Chunk, Document
from app.database.qdrant_client import get_qdrant_client

from llama_index.core import Document as LlamaDocument
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import TokenTextSplitter, MarkdownNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding

settings = get_settings()


@dataclass
class IngestionResult:
    document_id: str
    chunks_count: int
    total_tokens: int


def _init_llamaindex_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Initialize LlamaIndex global settings."""
    # Embedding Configuration
    kwargs = {
        "model": settings.embedding_model,
        "api_key": settings.openrouter_api_key,
    }
    if settings.openrouter_embedding_url:
        base = settings.openrouter_embedding_url
        if base.endswith("/embeddings"):
            base = base[:-11]
        kwargs["api_base"] = base
        
    if "text-embedding-3" in settings.embedding_model and settings.embedding_dim:
        kwargs["dimensions"] = settings.embedding_dim

    Settings.embed_model = OpenAIEmbedding(**kwargs)

    # Chunker / Splitter Configuration
    Settings.text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


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
        _init_llamaindex_settings(chunk_size, chunk_overlap)
        
        # Build Document
        llama_doc = LlamaDocument(
            text=text,
            doc_id=document_id,
            metadata={
                "document_id": document_id,
                "source": source,
                "title": title,
                **meta
            }
        )
        
        # ── 2.1 Build Nodes (Chunks) ─────────────────────────────────────
        # If it's markdown, we use MarkdownNodeParser to split by headers (H1, H2, H3)
        # Otherwise, we fallback to the global Settings.text_splitter
        is_markdown = source.lower().endswith(".md") or text.strip().startswith("#")
        
        if is_markdown:
            parser = MarkdownNodeParser()
            nodes = parser.get_nodes_from_documents([llama_doc])
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
        vector_store = QdrantVectorStore(
            aclient=qdrant.client,     # use async client
            collection_name=qdrant.collection,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
            dense_vector_name="text-dense",
            sparse_vector_name="text-sparse",
        )
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
            # approximate token count
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
