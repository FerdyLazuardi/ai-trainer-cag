"""
Semantic Long-Term Memory Service — Qdrant-backed, per-user, per-episode.

Architecture:
  Each chat session → ONE memory point in Qdrant (user_ltm_memories collection).
  Point layout:
    vector  : embedding(session_summary)          ← enables semantic retrieval
    payload : {user_id, session_summary, course_names, unanswered_questions,
               session_id, created_at (epoch)}

Load strategy (per-request, query-aware):
  1. Embed the user's current query.
  2. Search Qdrant filtered by user_id → top-N candidates.
  3. Re-rank candidates by `cosine * exp(-age_days / decay_days)` so recent
     episodes outrank stale-but-similar ones.
  4. Concatenate the top-K survivors into a compact context string.
  5. Inject into system prompt as <user_history>.

Update strategy (AFK 10-second worker):
  1. Receive session summary + course_names + unanswered_questions from worker.
  2. Embed the summary.
  3. Upsert a deterministic point keyed by session_id (idempotent re-syncs).
"""
import math
import time
import uuid

from loguru import logger
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.api.user_utils import is_real_user
from app.config.embedding_config import ensure_llamaindex_configured
from app.database.qdrant_client import get_qdrant_client

_LTM_COLLECTION = "user_ltm_memories"
_LTM_TOP_K = 2          # number of episodes injected into the prompt
_LTM_CANDIDATES = 6     # over-fetch this many before time-decay re-ranking
_LTM_DECAY_DAYS = 60.0  # half-weight age — older episodes get exp-decayed
_MAX_COURSE_NAMES = 3        # cap course names stored per episode
_MAX_UNANSWERED = 3          # cap unanswered questions stored per episode


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

        Over-fetches `_LTM_CANDIDATES`, then re-ranks with a time-decay so
        recent episodes outrank stale-but-similar ones, and trims to top-K.

        Args:
            user_id: Moodle user identifier.
            query:   The user's current query (used as the retrieval key).
            query_embedding: Optional pre-computed embedding to avoid duplicate API call.

        Returns:
            {
                "summary":              "<compact multi-episode context>",
                "course_names":         ["Course A", "Course B", ...],
                "unanswered_questions": ["...", ...]
            }
        """
        empty = {"summary": "", "course_names": [], "unanswered_questions": []}
        if not is_real_user(user_id=user_id, role="moodle_user"):
            return empty

        if query_embedding is None:
            try:
                query_vector = await self._embed(query)
            except Exception as exc:
                logger.warning("LTM: failed to embed query for retrieval", error=str(exc))
                return empty
        else:
            query_vector = query_embedding

        qdrant = get_qdrant_client()
        try:
            response = await qdrant.client.query_points(
                collection_name=_LTM_COLLECTION,
                query=query_vector,
                query_filter=self._build_user_filter(user_id),
                limit=_LTM_CANDIDATES,
                with_payload=True,
            )
            results = response.points
        except Exception as exc:
            logger.warning("LTM: Qdrant search failed", user_id=user_id, error=str(exc))
            return empty

        if not results:
            logger.debug("LTM: no past episodes found", user_id=user_id)
            return empty

        # Time-decay re-rank: cosine * exp(-age_days / decay_days)
        now = time.time()
        scored: list[tuple[float, object]] = []
        for hit in results:
            payload = hit.payload or {}
            created_at = float(payload.get("created_at") or now)
            age_days = max(0.0, (now - created_at) / 86400.0)
            decay = math.exp(-age_days / _LTM_DECAY_DAYS)
            final_score = float(hit.score or 0.0) * decay
            scored.append((final_score, hit))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [h for _, h in scored[:_LTM_TOP_K]]

        # Aggregate episodes into compact context — one sentence per episode
        episode_lines: list[str] = []
        all_course_names: list[str] = []
        all_unanswered: list[str] = []

        for hit in top:
            payload = hit.payload or {}
            ep_summary = payload.get("session_summary", "").strip()
            ep_course_names: list[str] = payload.get("course_names", []) or []
            ep_unanswered: list[str] = payload.get("unanswered_questions", []) or []

            if ep_summary:
                episode_lines.append(f"- {ep_summary}")
            all_course_names.extend(ep_course_names)
            all_unanswered.extend(ep_unanswered)

        if not episode_lines:
            return empty

        # Deduplicate while preserving order
        def _dedup(items: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for x in items:
                if x and x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        unique_course_names = _dedup(all_course_names)
        unique_unanswered = _dedup(all_unanswered)

        combined_summary = "\n".join(episode_lines)
        logger.info(
            "LTM: loaded semantic episodes",
            user_id=user_id,
            episodes=len(episode_lines),
            course_names=len(unique_course_names),
            unanswered=len(unique_unanswered),
        )
        return {
            "summary": combined_summary,
            "course_names": unique_course_names[:_MAX_COURSE_NAMES],
            "unanswered_questions": unique_unanswered[:_MAX_UNANSWERED],
        }

    async def update(
        self,
        user_id: str,
        session_summary: str,
        new_course_names: list[str],
        session_id: str,
        unanswered_questions: list[str] | None = None,
        llm=None,
    ) -> None:
        """
        Persist a new memory episode for this user in Qdrant.

        Each call upserts ONE vector point per session (deterministic UUID5
        derived from session_id), so repeated syncs of the same session
        overwrite the single point instead of creating duplicates.

        Args:
            user_id:              Moodle user identifier.
            session_summary:      Summarised text of the session (from STM worker).
            new_course_names:     Course names extracted by the caller.
            session_id:           conversation_id for deduplication & audit.
            unanswered_questions: Questions the AI failed to answer this session.
            llm:                  Kept for backwards-compat — no longer used here
                                   since course_names are produced inline by the
                                   structured LTM summarization in worker.py.
        """
        if not is_real_user(user_id=user_id, role="moodle_user"):
            return
        if not session_summary or not session_summary.strip():
            return

        course_names = (new_course_names or [])[:_MAX_COURSE_NAMES]
        unanswered = (unanswered_questions or [])[:_MAX_UNANSWERED]

        # Embed the session summary
        try:
            vector = await self._embed(session_summary)
        except Exception as exc:
            logger.warning("LTM: failed to embed session summary", user_id=user_id, error=str(exc))
            return

        payload = {
            "user_id": user_id,
            "session_summary": session_summary[:800],   # hard cap — LLM is told 15 words; this is the safety net
            "course_names": course_names,
            "unanswered_questions": unanswered,
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
                unanswered=len(unanswered),
            )
        except Exception as exc:
            logger.warning("LTM: Qdrant upsert failed", user_id=user_id, error=str(exc))


# Singleton
qdrant_ltm = QdrantLTMService()
