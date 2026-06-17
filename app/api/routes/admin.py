import secrets
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy import text

from app.config.settings import get_settings
from app.database.postgres import engine

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

@router.get("/logs", summary="Get aggregated logs for Dashboard")
async def get_dashboard_logs(limit: int = 100, _=Depends(verify_api_key)) -> Dict[str, Any]:
    async with engine.connect() as conn:
        # Total Interactions
        total_q_res = await conn.execute(text("SELECT COUNT(*) FROM agent_logs WHERE endpoint != 'askfer' OR endpoint IS NULL"))
        total_q = total_q_res.scalar() or 0
        
        # Average Latency
        avg_lat_res = await conn.execute(text("SELECT AVG(latency_ms) FROM agent_logs WHERE latency_ms IS NOT NULL AND (endpoint != 'askfer' OR endpoint IS NULL)"))
        avg_lat = avg_lat_res.scalar() or 0.0
        
        # Cache Hit Rate
        cache_hits_res = await conn.execute(text("SELECT SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) FROM agent_logs WHERE endpoint != 'askfer' OR endpoint IS NULL"))
        cache_hits = cache_hits_res.scalar() or 0
        hit_rate = (float(cache_hits) / float(total_q) * 100) if total_q > 0 else 0.0

        # Intents
        intents_result_exec = await conn.execute(text("SELECT intent, COUNT(*) as count FROM agent_logs WHERE endpoint != 'askfer' OR endpoint IS NULL GROUP BY intent"))
        intents_result = intents_result_exec.fetchall()
        intents = [{"intent": str(row[0]), "count": int(row[1])} for row in intents_result]

        # Trends
        trends_result_exec = await conn.execute(text("""
            SELECT DATE(created_at) as date, COUNT(*) as queries
            FROM agent_logs
            WHERE endpoint != 'askfer' OR endpoint IS NULL
            GROUP BY DATE(created_at)
            ORDER BY date ASC
        """))
        trends_result = trends_result_exec.fetchall()
        trends = [{"date": str(row[0]), "queries": int(row[1])} for row in trends_result]

        # Recent Logs
        # Exclude `cache_lookup` rows — they're cache observability events
        # (no chat happened, intent is NULL, query is the same as the chat
        # turn that triggered the lookup). Surfacing them alongside real
        # turns makes the dashboard show the same query 3-4× with intent=None,
        # which looks like a routing bug. They live in agent_logs for
        # admin/cache-monitoring — not for Recent Logs.
        logs_result_exec = await conn.execute(text("""
            SELECT created_at, intent, latency_ms, cache_hit, query, answer, conversation_id, llm_tokens_used, chunks_retrieved,
                   faithfulness_score, needs_empathy, needs_reasoning, needs_lookup, retrieved_context
            FROM agent_logs
            WHERE (endpoint != 'askfer' OR endpoint IS NULL)
              AND endpoint != 'cache_lookup'
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"limit": limit})
        logs_result = logs_result_exec.fetchall()

        logs = [
            {
                "created_at": str(row[0]),
                "intent": str(row[1]) if row[1] else "UNKNOWN",
                "latency_ms": float(row[2]) if row[2] is not None else 0.0,
                "cache_hit": bool(row[3]),
                "query": str(row[4]),
                "answer": str(row[5]),
                "session_id": str(row[6]) if row[6] else "Unknown",
                # `tokens` is the raw LLM token count (e.g. 24, 2124). It is
                # NOT in thousands. Pre-format with thousands separator so
                # the dashboard can display "1,234" instead of "1234", and
                # "0" turns into "—" so empty rows don't look like a count
                # of zero. Keep `tokens_raw` for any numeric use.
                "tokens": (
                    f"{int(row[7]):,}" if row[7] else "—"
                ) if row[7] is not None else "—",
                "tokens_raw": int(row[7]) if row[7] is not None else 0,
                "retrieved": int(row[8]) if row[8] is not None else 0,
                "faithfulness": float(row[9]) if row[9] is not None else None,
                "empathy": float(row[10]) if row[10] is not None else None,
                "reasoning": float(row[11]) if row[11] is not None else None,
                "lookup": float(row[12]) if row[12] is not None else None,
                "retrieved_context": row[13] if row[13] is not None else []
            }
            for row in logs_result
        ]
        
        # Users (LTM)
        users_result_exec = await conn.execute(text("""
            SELECT user_id, role, preferred_tone, formatting_pref, custom_instructions, updated_at
            FROM user_profiles
            ORDER BY updated_at DESC
        """))
        users_result = users_result_exec.fetchall()
        
        users = [
            {
                "user_id": str(row[0]),
                "role": str(row[1]) if row[1] else "",
                "preferred_tone": str(row[2]) if row[2] else "",
                "formatting_pref": str(row[3]) if row[3] else "",
                "custom_instructions": str(row[4]) if row[4] else "",
                "updated_at": str(row[5])
            }
            for row in users_result
        ]

    return {
        "kpis": {
            "total_queries": int(total_q),
            "avg_latency": float(avg_lat),
            "hit_rate": float(hit_rate)
        },
        "intents": intents,
        "trends": trends,
        "logs": logs,
        "users": users
    }
