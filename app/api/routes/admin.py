import asyncio
import base64
import secrets
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy import text

from app.config.settings import get_settings
from app.database.postgres import engine

settings = get_settings()

router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    settings = get_settings()
    # Constant-time compare to avoid a timing side-channel that could let an
    # attacker recover the admin key byte-by-byte. `secrets.compare_digest`
    # requires str (not None), so reject the missing-header case first —
    # APIKeyHeader(auto_error=False) yields None when the header is absent.
    if not api_key or not secrets.compare_digest(api_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


# Opaque cursor for Recent Logs pagination: base64('<created_at_iso>|<id>').
# Keyset on (created_at, id) avoids duplicates when multiple rows share a
# second-resolution timestamp and is index-friendly (created_at DESC, id DESC).
def _encode_cursor(created_at: str, row_id: int) -> str:
    raw = f"{created_at}|{row_id}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[datetime, int] | None:
    # ponytail: SQLAlchemy text() with asyncpg passes positional params typed —
    # passing the raw ISO string raises DataError("expected datetime"). Parse
    # to a real datetime so the bound parameter survives the driver roundtrip.
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded.encode()).decode()
        ts, rid = raw.split("|", 1)
        # Accept "YYYY-MM-DD HH:MM:SS.fff+ZZ:ZZ" (Postgres timestamptz default
        # format from text()) and "YYYY-MM-DDTHH:MM:SS.fff+ZZ:ZZ" (ISO).
        ts_norm = ts.replace("T", " ")
        return datetime.fromisoformat(ts_norm), int(rid)
    except Exception:
        return None


async def _run_one(sql: str, params: dict | None = None):
    """Acquire a fresh connection, run one query, return Result, close.

    The original code reused a single connection across 7 awaits, so the
    queries ran serially (each await blocks until the previous query's
    network round-trip finished). Each `_run_one` opens its own connection
    so asyncio.gather can fan them out — Postgres pool (8 base + 12 overflow)
    absorbs the burst on the X-API-Key gated admin endpoint.
    """
    async with engine.connect() as conn:
        return await conn.execute(text(sql), params or {})


@router.get("/logs", summary="Get aggregated logs for Dashboard")
async def get_dashboard_logs(
    limit: int = Query(default=100, ge=1, le=500),
    cursor: str | None = Query(default=None, description="Opaque pagination cursor from a prior response's next_cursor"),
    _=Depends(verify_api_key),
) -> Dict[str, Any]:
    decoded = _decode_cursor(cursor) if cursor else None

    # cache_lookup rows are observability events (intentional, see audit
    # 5.11) — counted in KPIs/intents/trends but NOT in Recent Logs (would
    # show the same query 3-4× as a fake routing bug).
    chat_where = "(endpoint != 'cache_lookup' OR endpoint IS NULL)"

    # Keyset pagination: strict (created_at, id) less-than to avoid duplicates
    # on shared-second timestamps. OR form keeps it valid SQLAlchemy text()
    # (no row-value tuple binding needed).
    cursor_clause = ""
    cursor_params: dict[str, Any] = {}
    if decoded:
        cursor_clause = (
            "AND (created_at < :cursor_ts "
            "OR (created_at = :cursor_ts AND id < :cursor_id)) "
        )
        cursor_params = {"cursor_ts": decoded[0], "cursor_id": decoded[1]}

    # Fetch limit+1 to detect has_more without a separate COUNT(*).
    recent_limit = limit + 1

    total_q, avg_lat, cache_hits, intents_q, trends_q, logs_q, users_q, perf_q = await asyncio.gather(
        _run_one(f"SELECT COUNT(*) FROM agent_logs WHERE {chat_where}"),
        _run_one(f"SELECT AVG(latency_ms) FROM agent_logs WHERE latency_ms IS NOT NULL AND {chat_where}"),
        _run_one(f"SELECT SUM(or_prompt_tokens), SUM(or_cached_tokens), SUM(or_completion_tokens), SUM(or_cost) FROM agent_logs WHERE {chat_where}"),
        _run_one(f"SELECT intent, COUNT(*) AS count FROM agent_logs WHERE {chat_where} GROUP BY intent"),
        _run_one(f"""
            SELECT DATE(created_at) AS date, COUNT(*) AS queries
            FROM agent_logs
            WHERE {chat_where}
              AND created_at > NOW() - INTERVAL '30 days'
            GROUP BY DATE(created_at)
            ORDER BY date ASC
        """),
        _run_one(f"""
            SELECT created_at, id, intent, latency_ms, cache_hit, query, answer,
                   conversation_id, llm_tokens_used, chunks_retrieved,
                   faithfulness_score, needs_empathy, needs_reasoning, needs_lookup,
                   retrieved_context, or_prompt_tokens, or_cached_tokens, or_completion_tokens, or_provider,
                   rewritten_query, or_cost, or_generation_id
            FROM agent_logs
            WHERE {chat_where}
              {cursor_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT :limit
        """, {"limit": recent_limit, **cursor_params}),
        _run_one("""
            SELECT user_id, learning_summary, updated_at
            FROM user_ltm_memories
            ORDER BY updated_at DESC
        """),
        # Rolling-7d perf + quality. Window matters: an all-time AVG/percentile
        # is dragged for weeks by a single bad day (e.g. a cold-start/retry-storm
        # day with p95=90s), hiding that recent traffic is ~4s p95. 7d tracks
        # "how is it NOW". Latency excludes cache_lookup (no LLM); faithfulness
        # is the sampled judge score (NULL for un-evaluated turns).
        _run_one(f"""
            SELECT
              percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
                FILTER (WHERE latency_ms IS NOT NULL) AS p95,
              percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms)
                FILTER (WHERE latency_ms IS NOT NULL) AS p99,
              AVG(faithfulness_score) FILTER (WHERE faithfulness_score IS NOT NULL) AS faith_avg,
              COUNT(*) FILTER (WHERE faithfulness_score IS NOT NULL) AS faith_n,
              COUNT(*) FILTER (WHERE faithfulness_score IS NOT NULL
                AND faithfulness_score < :faith_min) AS faith_fail
            FROM agent_logs
            WHERE {chat_where} AND created_at > NOW() - INTERVAL '7 days'
        """, {"faith_min": settings.faithfulness_min}),
    )

    total = int(total_q.scalar() or 0)
    avg_latency = float(avg_lat.scalar() or 0.0)
    or_stats = cache_hits.fetchone()
    or_prompt = int(or_stats[0] or 0) if or_stats else 0
    or_cached = int(or_stats[1] or 0) if or_stats else 0
    or_completion = int(or_stats[2] or 0) if or_stats else 0
    total_cost = float(or_stats[3] or 0.0) if or_stats else 0.0
    hit_rate = (or_cached / or_prompt * 100.0) if or_prompt > 0 else 0.0

    perf = perf_q.fetchone()
    p95_7d = float(perf[0]) if perf and perf[0] is not None else 0.0
    p99_7d = float(perf[1]) if perf and perf[1] is not None else 0.0
    faith_avg_7d = float(perf[2]) if perf and perf[2] is not None else None
    faith_n_7d = int(perf[3]) if perf and perf[3] is not None else 0
    faith_fail_7d = int(perf[4]) if perf and perf[4] is not None else 0

    intents = [
        {"intent": str(row[0]) if row[0] else "UNKNOWN", "count": int(row[1])}
        for row in intents_q.fetchall()
    ]

    trends = [
        {"date": str(row[0]), "queries": int(row[1])}
        for row in trends_q.fetchall()
    ]

    log_rows = logs_q.fetchall()
    has_more = len(log_rows) > limit
    if has_more:
        log_rows = log_rows[:limit]

    logs = [
        {
            "created_at": str(row[0]),
            "intent": str(row[2]) if row[2] else "UNKNOWN",
            "latency_ms": float(row[3]) if row[3] is not None else 0.0,
            "cache_hit": bool(row[4]),
            "query": str(row[5]),
            "answer": str(row[6]) if row[6] else "",
            "session_id": str(row[7]) if row[7] else "Unknown",
            "tokens": (
                f"{int(row[8]):,}" if row[8] else "—"
            ) if row[8] is not None else "—",
            "tokens_raw": int(row[8]) if row[8] is not None else 0,
            "retrieved": int(row[9]) if row[9] is not None else 0,
            "faithfulness": float(row[10]) if row[10] is not None else None,
            "empathy": float(row[11]) if row[11] is not None else None,
            "reasoning": float(row[12]) if row[12] is not None else None,
            "lookup": float(row[13]) if row[13] is not None else None,
            "retrieved_context": row[14] if row[14] is not None else [],
            "or_prompt_tokens": int(row[15]) if row[15] is not None else 0,
            "or_cached_tokens": int(row[16]) if row[16] is not None else 0,
            "or_completion_tokens": int(row[17]) if row[17] is not None else 0,
            "or_provider": str(row[18]) if row[18] else "",
            "rewritten_query": str(row[19]) if len(row) > 19 and row[19] else None,
            "cost": float(row[20]) if len(row) > 20 and row[20] is not None else 0.0,
            "or_generation_id": str(row[21]) if len(row) > 21 and row[21] else None,
        }
        for row in log_rows
    ]

    next_cursor = None
    if has_more and log_rows:
        last = log_rows[-1]
        next_cursor = _encode_cursor(str(last[0]), int(last[1]))

    users = [
        {
            "user_id": str(row[0]),
            "learning_summary": str(row[1]) if row[1] else "",
            "updated_at": str(row[2]),
        }
        for row in users_q.fetchall()
    ]

    return {
        "kpis": {
            "total_queries": total,
            "avg_latency": avg_latency,
            "hit_rate": hit_rate,
            "or_prompt_tokens": or_prompt,
            "or_cached_tokens": or_cached,
            "or_completion_tokens": or_completion,
            "total_cost": total_cost,
            "p95_latency_7d": p95_7d,
            "p99_latency_7d": p99_7d,
            "faithfulness_avg_7d": faith_avg_7d,
            "faithfulness_n_7d": faith_n_7d,
            "faithfulness_fail_7d": faith_fail_7d,
        },
        "intents": intents,
        "trends": trends,
        "logs": logs,
        "next_cursor": next_cursor,
        "users": users,
    }


@router.get("/kb", summary="Get active KB text")
async def get_active_kb(
    _=Depends(verify_api_key),
) -> Dict[str, str]:
    from app.graph.pipeline import _load_active_cag_kb_text
    content = await _load_active_cag_kb_text()
    return {"content": content}


@router.get("/spreadsheet/schedule", summary="Get spreadsheet auto-sync schedule config")
async def get_spreadsheet_schedule(
    _=Depends(verify_api_key),
) -> Dict[str, Any]:
    from app.database.redis_client import get_redis_client
    import json
    redis = get_redis_client()
    raw = await redis.get("cag:spreadsheet:schedule")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {
        "enabled": False,
        "schedule_type": "daily",
        "hour": 2,
        "minute": 0,
        "day_of_week": 1,
        "last_run_at": None,
        "last_status": None,
        "last_result": None,
    }


@router.post("/spreadsheet/schedule", summary="Update spreadsheet auto-sync schedule config")
async def update_spreadsheet_schedule(
    payload: Dict[str, Any],
    _=Depends(verify_api_key),
) -> Dict[str, Any]:
    from app.database.redis_client import get_redis_client
    import json
    redis = get_redis_client()

    raw = await redis.get("cag:spreadsheet:schedule")
    existing = json.loads(raw) if raw else {}

    data = {
        "enabled": bool(payload.get("enabled", False)),
        "schedule_type": str(payload.get("schedule_type", "daily")),
        "hour": int(payload.get("hour", 2)),
        "minute": int(payload.get("minute", 0)),
        "day_of_week": int(payload.get("day_of_week", 1)),
        "last_run_at": payload.get("last_run_at") or existing.get("last_run_at"),
        "last_status": payload.get("last_status") or existing.get("last_status"),
        "last_result": payload.get("last_result") or existing.get("last_result"),
    }
    await redis.set("cag:spreadsheet:schedule", json.dumps(data))
    return data


@router.get("/spreadsheet/users", summary="Get spreadsheet user KPI records from PostgreSQL")
async def get_spreadsheet_users(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    search: str | None = Query(None),
    role: str | None = Query(None),
    _=Depends(verify_api_key),
) -> Dict[str, Any]:
    offset = (page - 1) * limit
    params: dict = {"limit": limit, "offset": offset}
    where_clauses = []

    if search:
        s = f"%{search.strip()}%"
        params["search"] = s
        where_clauses.append("(username ILIKE :search OR full_name ILIKE :search OR point_norm ILIKE :search)")

    if role:
        params["role"] = role.strip()
        where_clauses.append("role = :role")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    count_sql = f"SELECT COUNT(*) FROM user_kpi_data {where_sql}"
    data_sql = f"""
        SELECT username, full_name, periode, role, point_norm, data, updated_at
        FROM user_kpi_data
        {where_sql}
        ORDER BY updated_at DESC, username ASC
        LIMIT :limit OFFSET :offset
    """

    async with engine.connect() as conn:
        total_res = await conn.execute(text(count_sql), params)
        total = total_res.scalar() or 0

        rows_res = await conn.execute(text(data_sql), params)
        rows = rows_res.mappings().all()

        roles_res = await conn.execute(text("SELECT COALESCE(role, 'UNKNOWN'), COUNT(*) FROM user_kpi_data GROUP BY role ORDER BY COUNT(*) DESC"))
        role_counts = {str(r[0]): int(r[1]) for r in roles_res.fetchall()}

    users = []
    for r in rows:
        d = dict(r.get("data") or {})
        d["username"] = r["username"]
        d["full_name"] = r["full_name"]
        d["periode_kpi"] = r["periode"] or d.get("periode_kpi") or d.get("periode")
        d["role"] = r["role"] or d.get("role") or d.get("position")
        d["updated_at"] = r["updated_at"].isoformat() if r["updated_at"] else None
        users.append(d)

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "users_total": total,
        "role_counts": role_counts,
        "users": users,
    }


@router.get("/spreadsheet/branches", summary="Get branch records from PostgreSQL")
async def get_spreadsheet_branches(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    search: str | None = Query(None),
    _=Depends(verify_api_key),
) -> Dict[str, Any]:
    offset = (page - 1) * limit
    params: dict = {"limit": limit, "offset": offset}
    where_clauses = []

    if search:
        s = f"%{search.strip()}%"
        params["search"] = s
        where_clauses.append("(point ILIKE :search OR nama_cabang ILIKE :search OR point_norm ILIKE :search)")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    count_sql = f"SELECT COUNT(*) FROM branch_data {where_sql}"
    data_sql = f"""
        SELECT point, nama_cabang, point_norm, periode, data, updated_at
        FROM branch_data
        {where_sql}
        ORDER BY point ASC
        LIMIT :limit OFFSET :offset
    """

    async with engine.connect() as conn:
        total_res = await conn.execute(text(count_sql), params)
        total = total_res.scalar() or 0

        rows_res = await conn.execute(text(data_sql), params)
        rows = rows_res.mappings().all()

    branches = []
    for r in rows:
        d = dict(r.get("data") or {})
        d["point"] = r["point"]
        d["nama_cabang"] = r["nama_cabang"]
        d["periode"] = r["periode"] or d.get("periode")
        d["updated_at"] = r["updated_at"].isoformat() if r["updated_at"] else None
        branches.append(d)

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "branches_total": total,
        "branches": branches,
    }


@router.delete("/spreadsheet/clean", summary="Truncate all spreadsheet KPI and branch records from PostgreSQL")
@router.post("/spreadsheet/clean", summary="Truncate all spreadsheet KPI and branch records from PostgreSQL (POST alternative)")
async def clean_spreadsheet_data(
    _=Depends(verify_api_key),
) -> Dict[str, Any]:
    """Truncate user_kpi_data and branch_data in PostgreSQL matching the Proxmox admin command."""
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE TABLE user_kpi_data, branch_data RESTART IDENTITY;"))

    return {
        "status": "success",
        "message": "Tables user_kpi_data and branch_data truncated successfully.",
    }


@router.delete("/spreadsheet/users", summary="Clean all user KPI records from PostgreSQL")
@router.post("/spreadsheet/users", summary="Clean all user KPI records from PostgreSQL (POST alternative)")
async def clean_spreadsheet_users(
    _=Depends(verify_api_key),
) -> Dict[str, Any]:
    return await clean_spreadsheet_data(_=_)


@router.get("/prompts", summary="Get active production system prompts and modular blocks")
async def get_system_prompts(
    _=Depends(verify_api_key),
) -> Dict[str, Any]:
    """Return active production system prompts and modular XML blocks directly from cag-lms-agent."""
    from app.llm.cag_client import CAG_SYSTEM_PROMPT as CAG_BASELINE_PROMPT
    from app.llm.prompts import (
        CHIT_CHAT_PROMPT,
        CONVERSATIONAL_PROMPT,
        DISAMBIG,
        GROUNDING,
        LTM_LEARNING_SUMMARY_PROMPT,
        MENTORING_VOICE,
        OUTPUT_CONTRACT,
        PERSONA,
        RESPONSE_GUIDELINES,
        SOCRATIC_MODE,
        SOCRATIC_OUTPUT_CONTRACT,
        SOCRATIC_PROMPT,
        SOCRATIC_RESPONSE_GUIDELINES,
        STM_SUMMARY_PROMPT,
    )

    blocks = [
        {
            "id": "role",
            "tag": "<role>",
            "title": "Persona & Role Tailoring",
            "actAs": "Senior Learning & Development Trainer at Amartha (Digital Learning team)",
            "description": "Defines persona, peer-to-peer tone, Field Office (FO) vs Head Office (HO) answer tailoring, and language mirroring.",
            "content": PERSONA,
        },
        {
            "id": "output_contract",
            "tag": "<output_contract>",
            "title": "Output Contract & Safety Rules",
            "actAs": "Senior colleague speaking from verified internalized memory",
            "description": "Mandatory format constraints: direct opening, no markdown headings, no Chinese characters (Hanzi), no em-dashes.",
            "content": OUTPUT_CONTRACT,
        },
        {
            "id": "grounding",
            "tag": "<grounding>",
            "title": "Closed-Book Grounding & Truthfulness",
            "actAs": "Strict closed-book assistant bounded exclusively by Amarthapedia Knowledge Base",
            "description": "Enforces absolute truthfulness, zero guessing on metrics, and template for unknown terms.",
            "content": GROUNDING,
        },
        {
            "id": "response_guidelines",
            "tag": "<response_guidelines>",
            "title": "Response Guidelines (Caveman/Ponytail Style)",
            "actAs": "Direct trainer who values extreme brevity and zero filler",
            "description": "Brevity enforcement: 1-3 sentences max (<50 words) for lookups, bulleted lists for multi-point answers.",
            "content": RESPONSE_GUIDELINES,
        },
        {
            "id": "mentoring_voice",
            "tag": "<mentoring_voice>",
            "title": "Andragogy & Adult Learning Voice",
            "actAs": "Mentor to adult learners using workplace Andragogy principles",
            "description": "Adult learning principles: explain the 'why', anchor to field reality, decisive direct answers.",
            "content": MENTORING_VOICE,
        },
        {
            "id": "socratic_mode",
            "tag": "<mode>",
            "title": "Socratic Dialogue Engine",
            "actAs": "Pure Socratic Coach (facilitator who guides users to construct answers themselves)",
            "description": "5-stage diagnostic arc with explicit escape hatches for frustration or stalled states.",
            "content": SOCRATIC_MODE,
        },
        {
            "id": "disambig",
            "tag": "<disambiguate>",
            "title": "Disambiguation Gate",
            "actAs": "Active listener who clarifies vague queries with 1 focused question",
            "description": "Asks exactly one clarifying question when query is genuinely underspecified.",
            "content": DISAMBIG,
        },
    ]

    prompts = [
        {
            "id": "conversational",
            "title": "Conversational QA Prompt",
            "actAs": "Senior Learning & Development Trainer at Amartha (Digital Learning Team)",
            "category": "core_generation",
            "intentTrigger": "KNOWLEDGE • TOPIC_LIST • SECTION_DRILLDOWN • GENERAL",
            "pipelineStage": "Production Graph _generate_node (Primary LLM Turn)",
            "description": "Primary generation prompt for factual, policy, SOP, and operational questions.",
            "tokensEst": len(CONVERSATIONAL_PROMPT) // 4,
            "openRouterCached": True,
            "components": ["<role>", "<output_contract>", "<grounding>", "<response_guidelines>", "<mentoring_voice>", "<disambiguate>"],
            "content": CONVERSATIONAL_PROMPT,
        },
        {
            "id": "socratic",
            "title": "Socratic Coaching Prompt",
            "actAs": "Socratic Coach (Facilitator who never states answers directly)",
            "category": "core_generation",
            "intentTrigger": "COACHING (Coaching Mode Active)",
            "pipelineStage": "Production Graph _generate_node (Socratic Branch)",
            "description": "Interactive coaching prompt guiding learners via targeted inferential questions.",
            "tokensEst": len(SOCRATIC_PROMPT) // 4,
            "openRouterCached": True,
            "components": ["<role>", "<output_contract (socratic)>", "<grounding>", "<response_guidelines (socratic)>", "<disambiguate>", "<mode>"],
            "content": SOCRATIC_PROMPT,
        },
        {
            "id": "chit_chat",
            "title": "Chit-Chat & Guardrail Prompt",
            "actAs": "Friendly Peer Colleague with Strict Scope Guardrails",
            "category": "core_generation",
            "intentTrigger": "GREETING • AMBIGUOUS • OFF_SCOPE (~30% of Traffic)",
            "pipelineStage": "Production Graph _generate_node (Zero-KB Fast Path)",
            "description": "Handles informal greetings, vague turns, and declines off-topic queries with [OFFSCOPE].",
            "tokensEst": len(CHIT_CHAT_PROMPT) // 4,
            "openRouterCached": True,
            "components": ["<role>", "<output_contract>", "<instructions>"],
            "content": CHIT_CHAT_PROMPT,
        },
        {
            "id": "stm_summary",
            "title": "Short-Term Memory (STM) Dialogue Summarizer",
            "actAs": "Dialogue Compression Engine",
            "category": "memory_summarization",
            "intentTrigger": "Turn threshold exceeded (Rolling Window)",
            "pipelineStage": "Conversation State Maintenance (Async / Turn Boundary)",
            "description": "Compresses running conversation history into 2-4 English bullets (max 60 words).",
            "tokensEst": len(STM_SUMMARY_PROMPT) // 4,
            "openRouterCached": False,
            "components": ["STM Compression Directives"],
            "content": STM_SUMMARY_PROMPT,
        },
        {
            "id": "ltm_analyst",
            "title": "Long-Term Memory (LTM) Learning Profile Analyst",
            "actAs": "AI Learning Analyst (Employee Competency Profile Evaluator)",
            "category": "memory_summarization",
            "intentTrigger": "Post-Conversation Background Task",
            "pipelineStage": "Celery / Streaq Worker (Async user_ltm_memories Table Update)",
            "description": "Analyzes conversation session to maintain long-term competencies (Mastered vs Needs Practice).",
            "tokensEst": len(LTM_LEARNING_SUMMARY_PROMPT) // 4,
            "openRouterCached": False,
            "components": ["LTM Analysis Directives", "JSON Schema Output"],
            "content": LTM_LEARNING_SUMMARY_PROMPT,
        },
        {
            "id": "cag_baseline",
            "title": "Direct CAG Fallback System Prompt",
            "actAs": "Ava (Amartha LMS Assistant)",
            "category": "baseline",
            "intentTrigger": "Direct CAG Client Fallback / Standalone Benchmark",
            "pipelineStage": "app.llm.cag_client (Standalone Pipeline)",
            "description": "Baseline minimal prompt for direct CAG client fallback.",
            "tokensEst": len(CAG_BASELINE_PROMPT) // 4,
            "openRouterCached": True,
            "components": ["Ava Minimal Directives"],
            "content": CAG_BASELINE_PROMPT,
        },
    ]

    return {
        "success": True,
        "source": "backend_live",
        "total_prompts": len(prompts),
        "total_blocks": len(blocks),
        "prompts": prompts,
        "blocks": blocks,
    }




