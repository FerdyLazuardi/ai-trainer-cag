"""
Semantic Long-Term Memory Service — Qdrant-backed, per-user, per-episode.

Architecture:
  Each chat session → ONE memory point in Qdrant (user_ltm_memories collection).
  Point layout:
    vector  : embedding(session_summary)          ← enables semantic retrieval
    payload : {user_id, session_summary, topics,
               session_id, created_at (epoch)}

Load strategy (per-request, query-aware):
  1. Embed the user's current query.
  2. Search Qdrant filtered by user_id → top-2 most relevant past episodes.
  3. Concatenate & summarise them into a compact context string (token-efficient).
  4. Inject into system prompt as <user_history>.

Update strategy (AFK 10-second worker):
  1. Receive session summary from get_or_summarize_history().
  2. Extract topics via LLM (cheap model).
  3. Embed the summary.
  4. Upsert a new point (one per session — no merging needed).
"""
import time
import uuid

from loguru import logger
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.api.user_utils import is_real_user
from app.config.embedding_config import ensure_llamaindex_configured
from app.database.qdrant_client import get_qdrant_client

_LTM_COLLECTION = "user_ltm_memories"
_LTM_TOP_K = 2          # retrieve at most 2 past episodes per request
_MAX_COURSE_NAMES = 3        # cap course names stored per episode


class QdrantLTMService:
    """Semantic long-term memory backed by Qdrant vector store."""

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _embed(self, text: str) -> list[float]:
        """
        Embed text using the same OpenAI embedding model as the KB pipeline.
        Returns a flat list of floats (1536-dim for text-embedding-3-small).
        """
        ensure_llamaindex_configured()
        from llama_index.core import Settings as LISettings
        embed_model = LISettings.embed_model
        result = await embed_model.aget_text_embedding(text)
        return result

    def _build_user_filter(self, user_id: str) -> Filter:
        """Qdrant filter that restricts search to a single user's memories."""
        return Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            ]
        )

    # ── Public API ────────────────────────────────────────────────────────────

    async def load(self, user_id: str, query: str, query_embedding: list[float] | None = None) -> dict:
        """
        Semantic retrieval of the most relevant past episodes for this user.

        Args:
            user_id: Moodle user identifier.
            query:   The user's current query (used as the retrieval key).
            query_embedding: Optional pre-computed embedding to avoid duplicate API call.

        Returns:
            {
                "summary": "<compact multi-episode context>",
                "course_names":  ["Course A", "Course B", ...]
            }
        """
        if not is_real_user(user_id=user_id, role="moodle_user"):
            return {"summary": "", "course_names": []}

        if query_embedding is None:
            try:
                query_vector = await self._embed(query)
            except Exception as exc:
                logger.warning("LTM: failed to embed query for retrieval", error=str(exc))
                return {"summary": "", "course_names": []}
        else:
            query_vector = query_embedding

        qdrant = get_qdrant_client()
        try:
            response = await qdrant.client.query_points(
                collection_name=_LTM_COLLECTION,
                query=query_vector,
                query_filter=self._build_user_filter(user_id),
                limit=_LTM_TOP_K,
                with_payload=True,
            )
            results = response.points
        except Exception as exc:
            logger.warning("LTM: Qdrant search failed", user_id=user_id, error=str(exc))
            return {"summary": "", "course_names": []}

        if not results:
            logger.debug("LTM: no past episodes found", user_id=user_id)
            return {"summary": "", "course_names": []}

        # Aggregate episodes into compact context — one sentence per episode
        episode_lines: list[str] = []
        all_course_names: list[str] = []

        for hit in results:
            payload = hit.payload or {}
            ep_summary = payload.get("session_summary", "").strip()
            ep_course_names: list[str] = payload.get("course_names", [])

            if ep_summary:
                episode_lines.append(f"- {ep_summary}")
            all_course_names.extend(ep_course_names)

        if not episode_lines:
            return {"summary": "", "course_names": []}

        # Deduplicate course names while preserving order
        seen: set[str] = set()
        unique_course_names: list[str] = []
        for t in all_course_names:
            if t not in seen:
                seen.add(t)
                unique_course_names.append(t)

        combined_summary = "\n".join(episode_lines)
        logger.info(
            "LTM: loaded semantic episodes",
            user_id=user_id,
            episodes=len(episode_lines),
            course_names=len(unique_course_names),
        )
        return {
            "summary": combined_summary,
            "course_names": unique_course_names[:_MAX_COURSE_NAMES],
        }

    async def update(
        self,
        user_id: str,
        session_summary: str,
        new_course_names: list[str],
        session_id: str,
        llm=None,
    ) -> None:
        """
        Persist a new memory episode for this user in Qdrant.

        Each call creates ONE new vector point (per-session episode).
        No merging / overwriting of previous episodes — the collection
        grows one point per session, enabling granular semantic retrieval.

        Args:
            user_id:         Moodle user identifier.
            session_summary: Summarised text of the session (from STM worker).
            new_course_names:Course names extracted by the caller (or empty list).
            session_id:      conversation_id for deduplication & audit.
            llm:             Optional cheap LLM for course name extraction fallback.
        """
        if not is_real_user(user_id=user_id, role="moodle_user"):
            return
        if not session_summary or not session_summary.strip():
            return

        # Extract course names if not supplied
        course_names = new_course_names or []
        if not course_names and llm:
            course_names = await self._extract_course_names(session_summary, llm)

        # Embed the session summary
        try:
            vector = await self._embed(session_summary)
        except Exception as exc:
            logger.warning("LTM: failed to embed session summary", user_id=user_id, error=str(exc))
            return

        payload = {
            "user_id": user_id,
            "session_summary": session_summary[:2000],   # hard cap for safety
            "course_names": course_names[:_MAX_COURSE_NAMES],
            "session_id": session_id,
            "created_at": time.time(),                   # float epoch for index
        }

        qdrant = get_qdrant_client()
        try:
            from qdrant_client.models import PointStruct
            # Use deterministic UUID based on session_id so that multiple syncs 
            # for the same session overwrite the single point.
            deterministic_id = str(uuid.uuid5(uuid.NAMESPACE_OID, session_id))
            point = PointStruct(
                id=deterministic_id,
                vector=vector,
                payload=payload,
            )
            await qdrant.client.upsert(
                collection_name=_LTM_COLLECTION,
                points=[point],
                wait=False,   # fire-and-forget — non-blocking in background worker
            )
            logger.info(
                "LTM: episode persisted to Qdrant",
                user_id=user_id,
                session_id=session_id,
                course_names=len(course_names),
            )
        except Exception as exc:
            logger.warning("LTM: Qdrant upsert failed", user_id=user_id, error=str(exc))

    async def _extract_course_names(self, summary: str, llm) -> list[str]:
        """
        Use the cheap LLM to extract a short list of Course Names from the summary.
        Returns a list of short strings (max 3 course names).
        Falls back to empty list on error.
        """
        try:
            from langchain_core.messages import HumanMessage
            from langfuse.langchain import CallbackHandler
            lf_handler = CallbackHandler()
            prompt = (
                "Ekstrak Nama Materi (Course Name) yang dibahas dari ringkasan percakapan berikut. "
                "Contoh Course Name yang valid: 'Product Knowledge Amartha', 'Client Protection', dsb. "
                "Kembalikan HANYA maksimal 3 Nama Materi yang dipisahkan dengan koma. "
                "Tanpa penjelasan, tanpa nomor, tulis nama aslinya saja.\n\n"
                f"Ringkasan:\n{summary}\n\nNama Materi:"
            )
            resp = await llm.ainvoke(
                [HumanMessage(content=prompt)],
                config={"callbacks": [lf_handler], "run_name": "a-pedi-ltm-extract-courses"}
            )
            raw = resp.content.strip()
            course_names = [t.strip() for t in raw.split(",") if t.strip()]
            return course_names[:_MAX_COURSE_NAMES]
        except Exception as exc:
            logger.warning("LTM: course name extraction failed", error=str(exc))
            return []


# Singleton
qdrant_ltm = QdrantLTMService()
