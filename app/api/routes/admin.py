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
    if api_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

@router.get("/logs", summary="Get aggregated logs for Dashboard")
async def get_dashboard_logs(limit: int = 100, _=Depends(verify_api_key)) -> Dict[str, Any]:
    async with engine.connect() as conn:
        # Total Interactions
        total_q_res = await conn.execute(text("SELECT COUNT(*) FROM agent_logs"))
        total_q = total_q_res.scalar() or 0
        
        # Average Latency
        avg_lat_res = await conn.execute(text("SELECT AVG(latency_ms) FROM agent_logs WHERE latency_ms IS NOT NULL"))
        avg_lat = avg_lat_res.scalar() or 0.0
        
        # Cache Hit Rate
        cache_hits_res = await conn.execute(text("SELECT SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) FROM agent_logs"))
        cache_hits = cache_hits_res.scalar() or 0
        hit_rate = (float(cache_hits) / float(total_q) * 100) if total_q > 0 else 0.0

        # Intents
        intents_result_exec = await conn.execute(text("SELECT intent, COUNT(*) as count FROM agent_logs GROUP BY intent"))
        intents_result = intents_result_exec.fetchall()
        intents = [{"intent": str(row[0]), "count": int(row[1])} for row in intents_result]

        # Trends
        trends_result_exec = await conn.execute(text("""
            SELECT DATE(created_at) as date, COUNT(*) as queries
            FROM agent_logs
            GROUP BY DATE(created_at)
            ORDER BY date ASC
        """))
        trends_result = trends_result_exec.fetchall()
        trends = [{"date": str(row[0]), "queries": int(row[1])} for row in trends_result]

        # Recent Logs
        logs_result_exec = await conn.execute(text("""
            SELECT created_at, intent, latency_ms, cache_hit, query, answer, conversation_id, llm_tokens_used, chunks_retrieved
            FROM agent_logs
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"limit": limit})
        logs_result = logs_result_exec.fetchall()
        
        logs = [
            {
                "created_at": str(row[0]),
                "intent": str(row[1]),
                "latency_ms": float(row[2]) if row[2] is not None else 0.0,
                "cache_hit": bool(row[3]),
                "query": str(row[4]),
                "answer": str(row[5]),
                "session_id": str(row[6]) if row[6] else "Unknown",
                "tokens": int(row[7]) if row[7] is not None else 0,
                "retrieved": int(row[8]) if row[8] is not None else 0
            }
            for row in logs_result
        ]

    return {
        "kpis": {
            "total_queries": int(total_q),
            "avg_latency": float(avg_lat),
            "hit_rate": float(hit_rate)
        },
        "intents": intents,
        "trends": trends,
        "logs": logs
    }
