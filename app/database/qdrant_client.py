"""
Qdrant client singleton.
Handles collection creation/recreation and provides the shared async client.

On startup, ensure_collection() checks whether the existing collection was built
with a different embedding dimension than the one currently configured. If there
is a mismatch it deletes the stale collection and creates a fresh one using the
correct dimension (1536 for text-embedding-3-small).

WARNING: A dimension-mismatch rebuild will erase all existing vectors.
         Re-ingest your documents after any such rebuild.
"""
from functools import lru_cache

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    VectorParams,
)
from qdrant_client import models

from app.config.settings import get_settings

settings = get_settings()


class QdrantManager:
    """Wraps AsyncQdrantClient with collection management."""

    def __init__(self) -> None:
        self.client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
            prefer_grpc=False,
        )
        self.collection = settings.qdrant_collection
        self.dim = settings.embedding_dim

    async def _create_collection(self) -> None:
        """Create the Qdrant collection with the configured dimension and payload indexes."""
        await self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "text-dense": VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
            on_disk_payload=True,
        )
        # Index frequently-filtered metadata fields for fast pre-filtering
        await self.client.create_payload_index(
            collection_name=self.collection,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        await self.client.create_payload_index(
            collection_name=self.collection,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info(
            "Qdrant collection created",
            collection=self.collection,
            dim=self.dim,
        )

    async def ensure_collection(self) -> None:
        """
        Ensure the Qdrant collection exists and has the correct vector dimension.

        Logic:
        - Collection does not exist  → create it.
        - Collection exists, correct dim → nothing to do.
        - Collection exists, WRONG dim  → delete it and recreate (vectors must be
          re-ingested after this).
        """
        collections = await self.client.get_collections()
        existing_names = {c.name for c in collections.collections}

        if self.collection not in existing_names:
            # Fresh install — just create.
            await self._create_collection()
            return

        # Collection exists — verify the stored vector dimension matches config.
        info = await self.client.get_collection(self.collection)
        stored_dim: int | None = None

        # Support both named-vector and single-vector collection layouts.
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, VectorParams):
            stored_dim = vectors_config.size
        elif isinstance(vectors_config, dict):
            # Named-vector layout: pick any entry (all should share the same dim)
            first_cfg = next(iter(vectors_config.values()), None)
            if first_cfg is not None:
                stored_dim = first_cfg.size

        if stored_dim is not None and stored_dim != self.dim:
            logger.warning(
                "Qdrant collection has WRONG dimension — recreating",
                collection=self.collection,
                stored_dim=stored_dim,
                required_dim=self.dim,
            )
            await self.client.delete_collection(self.collection)
            await self._create_collection()
        else:
            logger.info(
                "Qdrant collection OK",
                collection=self.collection,
                dim=stored_dim or self.dim,
            )


    async def _create_kb_collection(self) -> None:
        """Create the Knowledge_Base Qdrant collection with hybrid search and metadata indexes."""
        kb_col = settings.qdrant_kb_collection
        await self.client.create_collection(
            collection_name=kb_col,
            vectors_config={
                "text-dense": VectorParams(
                    size=self.dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
            on_disk_payload=True,
        )
        # Index metadata fields used for fast pre-filtering
        for field, schema in [
            ("department", PayloadSchemaType.KEYWORD),
            ("topic", PayloadSchemaType.KEYWORD),
            ("course_id", PayloadSchemaType.INTEGER),
            ("course_name", PayloadSchemaType.KEYWORD),
            ("document_id", PayloadSchemaType.KEYWORD),
            ("source", PayloadSchemaType.KEYWORD),
        ]:
            await self.client.create_payload_index(
                collection_name=kb_col,
                field_name=field,
                field_schema=schema,
            )
        logger.info("Qdrant Knowledge_Base collection created", collection=kb_col, dim=self.dim)

    async def ensure_kb_collection(self) -> None:
        """Ensure the Knowledge_Base collection exists and has the correct vector dimension."""
        kb_col = settings.qdrant_kb_collection
        collections = await self.client.get_collections()
        existing_names = {c.name for c in collections.collections}

        if kb_col not in existing_names:
            await self._create_kb_collection()
            return

        info = await self.client.get_collection(kb_col)
        stored_dim: int | None = None
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, VectorParams):
            stored_dim = vectors_config.size
        elif isinstance(vectors_config, dict):
            first_cfg = next(iter(vectors_config.values()), None)
            if first_cfg is not None:
                stored_dim = first_cfg.size

        if stored_dim is not None and stored_dim != self.dim:
            logger.warning("Knowledge_Base collection has WRONG dimension — recreating", stored_dim=stored_dim)
            await self.client.delete_collection(kb_col)
            await self._create_kb_collection()
        else:
            logger.info("Qdrant Knowledge_Base collection OK", collection=kb_col, dim=stored_dim or self.dim)

    # ── Long-Term Memory Collection ───────────────────────────────────────────

    _LTM_COLLECTION = "user_ltm_memories"

    async def _create_ltm_collection(self) -> None:
        """
        Create the user_ltm_memories collection for semantic long-term memory.

        Design choices:
        - Dense-only vectors (no sparse): semantic similarity is the primary retrieval mechanism.
        - Payload indexed on user_id (keyword) for fast per-user pre-filtering.
        - Payload indexed on created_at (float epoch) for time-based ordering.
        - on_disk_payload=True for memory efficiency at scale.
        """
        await self.client.create_collection(
            collection_name=self._LTM_COLLECTION,
            vectors_config=VectorParams(
                size=self.dim,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
            on_disk_payload=True,
        )
        # CRITICAL: user_id index ensures each query filters ONLY this user's memories
        await self.client.create_payload_index(
            collection_name=self._LTM_COLLECTION,
            field_name="user_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        # created_at as float epoch for time-based sorting / recency scoring
        await self.client.create_payload_index(
            collection_name=self._LTM_COLLECTION,
            field_name="created_at",
            field_schema=PayloadSchemaType.FLOAT,
        )
        logger.info(
            "Qdrant LTM collection created",
            collection=self._LTM_COLLECTION,
            dim=self.dim,
        )

    async def ensure_ltm_collection(self) -> None:
        """
        Ensure the user_ltm_memories collection exists with the correct vector dimension.
        Safe to call on every startup — idempotent.
        """
        collections = await self.client.get_collections()
        existing_names = {c.name for c in collections.collections}

        if self._LTM_COLLECTION not in existing_names:
            await self._create_ltm_collection()
            return

        # Validate dimension matches current embedding config
        info = await self.client.get_collection(self._LTM_COLLECTION)
        stored_dim: int | None = None
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, VectorParams):
            stored_dim = vectors_config.size
        elif isinstance(vectors_config, dict):
            first_cfg = next(iter(vectors_config.values()), None)
            if first_cfg is not None:
                stored_dim = first_cfg.size

        if stored_dim is not None and stored_dim != self.dim:
            logger.warning(
                "LTM collection has WRONG dimension — recreating",
                stored_dim=stored_dim,
                required_dim=self.dim,
            )
            await self.client.delete_collection(self._LTM_COLLECTION)
            await self._create_ltm_collection()
        else:
            logger.info(
                "Qdrant LTM collection OK",
                collection=self._LTM_COLLECTION,
                dim=stored_dim or self.dim,
            )


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantManager:
    """Return the singleton QdrantManager."""
    return QdrantManager()
