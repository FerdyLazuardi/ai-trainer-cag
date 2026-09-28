"""Helper functions for chat routes (user context enrichment, cache privacy, evaluation sampling, log formatting, stream helpers)."""
import asyncio
from collections import Counter
import random
import re
from typing import Any, Optional
import uuid

from loguru import logger
from sqlalchemy import select

from app.api.auth import User
from app.config.settings import get_settings
from app.database.models import BranchData, UserKPIData
from app.database.postgres import AsyncSessionLocal

settings = get_settings()

_REDUNDANT_KPI_KEYS = {
    "full_name", "jabatan", "role", "nama_cabang", "username",
    "user_id", "nik", "user_name", "updated_at", "periode", "periode_kpi",
    "point", "cabang", "area", "wilayah", "regional", "region", "pulau", "island",
}

_CACHE_SKIP_INTENTS = {"GREETING", "AMBIGUOUS", "MALICIOUS", "TOPIC_LIST", "COACHING", "OFF_SCOPE"}


async def _build_enriched_user_context(current_user: User) -> dict[str, Any]:
    """Build Moodle user profile enriched with UserKPIData and BranchData from Postgres."""
    kpi_data = None
    branch_data = None
    try:
        async with AsyncSessionLocal() as db_session:
            kpi_stmt = select(UserKPIData).where(UserKPIData.username == current_user.username)
            kpi_data = (await db_session.execute(kpi_stmt)).scalars().first()

            point_from_kpi = ""
            if kpi_data:
                if isinstance(kpi_data.data, dict):
                    for k, v in kpi_data.data.items():
                        if str(k).lower().strip() in ("point", "cabang") and v:
                            point_from_kpi = str(v).strip()
                            break
                if not point_from_kpi and getattr(kpi_data, "point_norm", None):
                    point_from_kpi = kpi_data.point_norm

            point_val = point_from_kpi or str(current_user.point or "").strip()
            if point_val:
                branch_stmt = select(BranchData).where(BranchData.point == point_val)
                branch_data = (await db_session.execute(branch_stmt)).scalars().first()
                if not branch_data:
                    norm_point = point_val.lower().replace(" ", "")
                    all_branches = (await db_session.execute(select(BranchData))).scalars().all()
                    for b in all_branches:
                        if (b.point or "").lower().replace(" ", "") == norm_point:
                            branch_data = b
                            break
    except Exception as exc:
        logger.warning(f"Failed to load spreadsheet data from database: {exc}")

    user_context: dict[str, Any] = {
        "name": current_user.fullname or current_user.username,
        "dept": current_user.dept,
        "location": current_user.location,
        "position": current_user.position,
        "grade": current_user.grade,
        "point": current_user.point,
        "gender": current_user.gender,
        "area": current_user.area,
        "regional": current_user.regional,
        "pulau": "",
    }

    if kpi_data and kpi_data.full_name:
        user_context["name"] = kpi_data.full_name

    if kpi_data and isinstance(kpi_data.data, dict):
        for k, v in kpi_data.data.items():
            k_lower = str(k).lower().strip()
            val_str = str(v).strip() if v is not None else ""
            if val_str:
                if k_lower in ("point", "cabang"):
                    user_context["point"] = val_str
                elif k_lower in ("area", "wilayah"):
                    user_context["area"] = val_str
                elif k_lower in ("regional", "region"):
                    user_context["regional"] = val_str
                elif k_lower in ("pulau", "island"):
                    user_context["pulau"] = val_str

    if branch_data and isinstance(branch_data.data, dict):
        for k, v in branch_data.data.items():
            k_lower = str(k).lower().strip()
            val_str = str(v).strip() if v is not None else ""
            if val_str:
                if not user_context.get("area") and k_lower in ("area", "wilayah"):
                    user_context["area"] = val_str
                elif not user_context.get("regional") and k_lower in ("regional", "region"):
                    user_context["regional"] = val_str
                elif not user_context.get("pulau") and k_lower in ("pulau", "island"):
                    user_context["pulau"] = val_str

    pos_upper = str(user_context.get("position") or "").strip().upper()
    grade_upper = str(user_context.get("grade") or "").strip().upper()
    is_mt = "MANAGEMENT TRAINEE" in pos_upper or pos_upper == "MT" or "MT" in grade_upper

    if kpi_data and isinstance(kpi_data.data, dict):
        sp_role = str(kpi_data.data.get("Role") or kpi_data.data.get("Jabatan") or "").strip()
        if is_mt and sp_role:
            user_context["position"] = f"Management Trainee ({sp_role.upper()})"
            user_context["role"] = sp_role.upper()

        for k, v in kpi_data.data.items():
            k_clean = str(k).strip()
            if k_clean.lower() not in _REDUNDANT_KPI_KEYS and k_clean not in user_context and v is not None:
                user_context[k_clean] = v

    if branch_data and isinstance(branch_data.data, dict):
        user_role = str(user_context.get("role") or "").strip().upper()
        if not user_role and user_context.get("position"):
            pos_str = str(user_context["position"]).upper()
            if "BM" in pos_str or "MANAGER" in pos_str:
                user_role = "BM"
            elif "BP" in pos_str or "PARTNER" in pos_str:
                user_role = "BP"

        for k, v in branch_data.data.items():
            k_clean = str(k).strip()
            if k_clean.lower() in _REDUNDANT_KPI_KEYS or v is None:
                continue
            if k_clean.lower().endswith(("updated_at", "updated at")):
                continue
            if user_role and ("Point BP - " in k_clean or "Point BM - " in k_clean):
                opposite_role = "BM" if user_role == "BP" else "BP"
                if f"Point {opposite_role} - " in k_clean:
                    continue
            if k_clean not in user_context:
                user_context[k_clean] = v

    return user_context


class StreamTextDeduper:
    """Suppresses duplicate prefix restarts if an upstream stream replays initial chunks."""

    def __init__(self) -> None:
        self.full_answer = ""
        self._restart_buffer = ""

    def feed(self, text: str) -> str:
        if not text:
            return ""
        if self._restart_buffer:
            self._restart_buffer += text
            if self.full_answer.startswith(self._restart_buffer):
                return ""
            if self._restart_buffer.startswith(self.full_answer):
                emit = self._restart_buffer[len(self.full_answer):]
                self.full_answer += emit
                self._restart_buffer = ""
                return emit
            emit = self._restart_buffer
            self.full_answer += emit
            self._restart_buffer = ""
            return emit
        if self.full_answer and self.full_answer.startswith(text):
            self._restart_buffer = text
            return ""
        self.full_answer += text
        return text


def _is_transient_stream_error(exc: BaseException) -> bool:
    """Return True if a stream exception is a transient network/timeout error."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    try:
        import httpx as _httpx_mod

        if isinstance(
            exc,
            (
                _httpx_mod.ReadError,
                _httpx_mod.ReadTimeout,
                _httpx_mod.WriteError,
                _httpx_mod.ConnectError,
                _httpx_mod.RemoteProtocolError,
                _httpx_mod.PoolTimeout,
            ),
        ):
            return True
    except Exception:
        pass
    return "httpx" in type(exc).__module__.lower()


def _should_write_response_cache(
    *,
    intent: Optional[str],
    is_low_relevance: bool,
    query: str,
    skip_cache: bool,
) -> bool:
    """Return True if a turn's response is eligible to be stored in Redis cache."""
    return (
        intent not in _CACHE_SKIP_INTENTS
        and not is_low_relevance
        and not _is_bare_affirmation(query)
        and not skip_cache
    )


def compute_was_personalized(
    *,
    ltm_profile: Optional[dict],
    recent_history: list,
    summary: str,
    user_context: Optional[dict],
) -> bool:
    """Return True if prompt contains user-specific payload (LTM, history, or Moodle profile)."""
    _ltm = ltm_profile or {}
    has_ltm = bool((_ltm.get("learning_summary") or "").strip())
    has_history = bool(recent_history) or bool((summary or "").strip())
    ctx = user_context or {}
    has_user_ctx = bool(ctx) and any(
        bool(str(v).strip()) for v in ctx.values() if v
    )
    return has_ltm or has_history or has_user_ctx


def cache_namespace_for(*, was_personalized: bool, user_id) -> str:
    """Resolve the cache namespace. User-scoped when personalized, else global."""
    return f"rag_user_{user_id}" if was_personalized else "rag"


_AFFIRMATION_TOKENS = frozenset({
    "iya", "ya", "yaa", "iyaa", "yoi", "yup", "yep", "yes", "yess",
    "boleh", "bole", "oleh", "oke", "okay", "ok", "oce", "sip", "siap",
    "mau", "lanjut", "lanjutkan", "lanjutin", "terus", "next", "gas", "gaskeun",
    "sure", "yuk", "ayo", "ayok", "dong", "deh", "aja", "nih", "kuy",
    "go", "ahead", "please", "tolong", "monggo", "silakan", "silahkan",
})


def _is_bare_affirmation(query: str) -> bool:
    """True if the message is ONLY affirmation/continuation tokens (<=4 words)."""
    toks = re.findall(r"[a-zA-Z]+", query.lower())
    if not toks or len(toks) > 4:
        return False
    return all(t in _AFFIRMATION_TOKENS for t in toks)


_EVAL_SKIP_INTENTS = _CACHE_SKIP_INTENTS


def _should_eval_turn(
    *,
    intent: str | None,
    intent_scores: dict | None,
    max_dense_score: float | None,
    answer: str | None,
    is_low_relevance: bool = False,
) -> bool:
    """Sampling decision for post-hoc evaluation."""
    if not settings.eval_enabled:
        return False
    if not answer or not answer.strip():
        return False
    if intent in _EVAL_SKIP_INTENTS:
        return False
    if is_low_relevance:
        return False

    scores = intent_scores or {}
    empathy = float(scores.get("needs_empathy") or 0.0)
    if empathy >= settings.eval_always_if_empathy_above:
        return True

    if (
        max_dense_score is not None
        and max_dense_score < settings.eval_always_if_dense_below
    ):
        return True

    return random.random() < settings.eval_sample_rate


async def _enqueue_eval(
    *,
    turn_id: Optional[str],
    query: str,
    answer: str,
    retrieved_context: list,
    intent: Optional[str],
    intent_scores: Optional[dict],
) -> None:
    """Push the turn onto the streaq queue for async LLM-as-judge evaluation."""
    if not turn_id:
        return
    try:
        from app.worker import eval_turn_task

        await eval_turn_task.enqueue(
            query=query,
            answer=answer,
            retrieved_context=retrieved_context or [],
            intent=intent,
            intent_scores=intent_scores or {},
            turn_id=turn_id,
        ).start(priority="high")
    except Exception as e:
        logger.warning(f"Failed to enqueue eval task: {e}")


def _user_log_fields(current_user: User, context: Optional[dict] = None) -> dict:
    """Extract user & branch hierarchy columns."""
    ctx = context or {}
    uctx = ctx.get("user_context") or (ctx.get("initial_state") or {}).get("user_context") or {}

    def _s(val: object, max_len: int) -> Optional[str]:
        s = str(val or "").strip()
        return s[:max_len] if s else None

    return {
        "username": _s(current_user.username, 64),
        "full_name": _s(uctx.get("name") or current_user.fullname, 255),
        "position": _s(uctx.get("position") or current_user.position, 128),
        "point": _s(uctx.get("point") or current_user.point, 64),
        "area": _s(uctx.get("area") or current_user.area, 64),
        "regional": _s(uctx.get("regional") or current_user.regional, 64),
        "pulau": _s(uctx.get("pulau"), 64),
    }


def _quality_log_fields(
    intent: Optional[str],
    intent_scores: Optional[dict],
    max_dense_score: Optional[float],
    gate_score: Optional[dict] = None,
) -> dict:
    """Build the durable quality-signal columns for an agent_logs row."""
    scores = intent_scores or {}

    def _f(key):
        v = scores.get(key)
        return float(v) if isinstance(v, (int, float)) else None

    def _gf(key, caster=float):
        if not gate_score:
            return None
        v = gate_score.get(key)
        return caster(v) if v is not None else None

    return {
        "intent": intent,
        "needs_lookup": _f("needs_lookup"),
        "needs_reasoning": _f("needs_reasoning"),
        "needs_empathy": _f("needs_empathy"),
        "max_dense_score": float(max_dense_score) if isinstance(max_dense_score, (int, float)) else None,
        "gate_decision": _gf("decision", str),
        "gate_intent": _gf("best_intent", str),
        "gate_best_cosine": _gf("best_cosine"),
        "gate_second_cosine": _gf("second_cosine"),
        "gate_margin": _gf("margin"),
    }


def _serialize_gate_score(gs) -> Optional[dict]:
    """Convert a GateScore dataclass (or None / dict) to a plain dict."""
    if gs is None:
        return None
    if isinstance(gs, dict):
        return gs
    return {
        "decision": getattr(gs, "decision", None),
        "committed": getattr(gs, "committed", None),
        "best_intent": getattr(gs, "best_intent", None),
        "best_cosine": getattr(gs, "best_cosine", None),
        "second_intent": getattr(gs, "second_intent", None),
        "second_cosine": getattr(gs, "second_cosine", None),
        "margin": getattr(gs, "margin", None),
    }


def _extract_sources(retrieved_context: list) -> list:
    sources = []
    if retrieved_context:
        for c in retrieved_context:
            if c.get("source") and c.get("source") != "Unknown":
                sources.append({
                    "chunk_id": c.get("chunk_id") or str(uuid.uuid4()),
                    "document_id": c.get("document_id") or "Unknown",
                    "source": c.get("source"),
                    "title": c.get("course_name") or c.get("title") or "Unknown",
                    "chunk_index": c.get("chunk_index") or 0,
                    "score": c.get("score") or 0.0,
                })
    return sources


def _auto_detect_course_id(retrieved_context: list, request_course_id: Optional[int]) -> Optional[int]:
    effective_course_id = request_course_id
    if effective_course_id in (None, 0) and retrieved_context:
        cids = [c.get("course_id") for c in retrieved_context if c.get("course_id") not in (None, "", 0)]
        if cids:
            try:
                effective_course_id = int(Counter(cids).most_common(1)[0][0])
            except (ValueError, TypeError):
                pass
    return effective_course_id
