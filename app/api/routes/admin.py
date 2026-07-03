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


def calculate_cost(or_provider: str, prompt: int, cached: int, completion: int) -> float:
    # Normalize model name
    m = (or_provider or "").lower()
    
    # Defaults (DeepSeek V3/V4 flash pricing as baseline fallback: $0.14 / $0.28 per M)
    prompt_rate = 0.14 / 1_000_000
    cached_rate = 0.014 / 1_000_000
    completion_rate = 0.28 / 1_000_000
    
    if "gemini" in m:
        # Gemini 2.5 Flash pricing: $0.075 / $0.30 per M
        prompt_rate = 0.075 / 1_000_000
        cached_rate = 0.01875 / 1_000_000
        completion_rate = 0.30 / 1_000_000
    elif "deepseek-chat" in m or "deepseek-v3" in m or "deepseek-v4" in m:
        prompt_rate = 0.14 / 1_000_000
        cached_rate = 0.014 / 1_000_000
        completion_rate = 0.28 / 1_000_000

    # Non-cached prompt tokens are charged at prompt_rate
    non_cached_prompt = max(0, prompt - cached)
    
    cost = (non_cached_prompt * prompt_rate) + (cached * cached_rate) + (completion * completion_rate)
    return round(cost, 8)


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
    non_askfer = "(endpoint != 'askfer' OR endpoint IS NULL)"
    chat_where = f"{non_askfer} AND endpoint != 'cache_lookup'"

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
        _run_one(f"SELECT SUM(or_prompt_tokens), SUM(or_cached_tokens), SUM(or_completion_tokens) FROM agent_logs WHERE {chat_where}"),
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
                   rewritten_query
            FROM agent_logs
            WHERE {chat_where}
              {cursor_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT :limit
        """, {"limit": recent_limit, **cursor_params}),
        _run_one("""
            SELECT user_id, role, preferred_tone, formatting_pref, custom_instructions, updated_at
            FROM user_profiles
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
    total_cost = calculate_cost(settings.llm_model, or_prompt, or_cached, or_completion)
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
            "cost": calculate_cost(
                str(row[18]) if row[18] else "",
                int(row[15]) if row[15] is not None else 0,
                int(row[16]) if row[16] is not None else 0,
                int(row[17]) if row[17] is not None else 0,
            ),
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
            "role": str(row[1]) if row[1] else "",
            "preferred_tone": str(row[2]) if row[2] else "",
            "formatting_pref": str(row[3]) if row[3] else "",
            "custom_instructions": str(row[4]) if row[4] else "",
            "updated_at": str(row[5]),
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
