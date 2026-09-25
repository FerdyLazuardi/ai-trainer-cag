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


def _extract_geo_from_json(data: dict | None) -> dict[str, str]:
    out = {"point": "", "area": "", "regional": "", "pulau": ""}
    if not isinstance(data, dict):
        return out
    for k, v in data.items():
        if v in (None, ""):
            continue
        kl = str(k).lower().strip()
        vs = str(v).strip()
        if not vs:
            continue
        if kl in ("point", "cabang") and not out["point"]:
            out["point"] = vs
        elif kl in ("area", "wilayah") and not out["area"]:
            out["area"] = vs
        elif kl in ("regional", "region") and not out["regional"]:
            out["regional"] = vs
        elif kl in ("pulau", "island") and not out["pulau"]:
            out["pulau"] = vs
    return out


def _parse_session_id_meta(session_id: str) -> dict[str, str]:
    """Parse {user_id}_{fullname}_{location}_{position}_{point} from session_id."""
    if not session_id or session_id in ("Unknown", "dev_user_123") or "_" not in session_id:
        return {"full_name": "", "position": "", "point": "", "point_norm": ""}
    parts = session_id.split("_")
    role_markers = {
        "fo", "ho", "admin", "bm", "bp", "am", "rm", "hmb",
        "area", "manager", "staff", "lead", "business", "officer",
    }
    marker_idx = -1
    for i in range(1, len(parts)):
        if parts[i].lower() in role_markers:
            marker_idx = i
            break
    if marker_idx > 1:
        name_parts = [p for p in parts[1:marker_idx] if p.lower() not in ("na", "n/a")]
        full_name = " ".join(p.capitalize() for p in name_parts)
        remainder = [p for p in parts[marker_idx:] if p.lower() not in ("na", "n/a")]
        terminal_words = {
            "manager", "partner", "officer", "leader", "staff", "admin",
            "coordinator", "specialist", "analyst", "head", "lead", "trainee",
        }
        term_idx = -1
        for j, tok in enumerate(remainder):
            if tok.lower() in terminal_words:
                term_idx = j
                break
        if term_idx != -1:
            role_tokens = remainder[: term_idx + 1]
            point_tokens = remainder[term_idx + 1 :]
        else:
            role_tokens = remainder[:3]
            point_tokens = remainder[3:]
    else:
        name_parts = [p for p in parts[1:] if p.lower() not in ("na", "n/a")]
        full_name = " ".join(p.capitalize() for p in name_parts)
        role_tokens = []
        point_tokens = []

    acronyms = {"fo", "ho", "bm", "bp", "am", "rm", "hmb"}
    position = " ".join(
        t.upper() if t.lower() in acronyms else t.capitalize()
        for t in role_tokens
    )
    point = " ".join(t.upper() for t in point_tokens)
    point_norm = "".join(t.lower() for t in point_tokens)
    return {
        "full_name": full_name,
        "position": position,
        "point": point,
        "point_norm": point_norm,
    }


async def _backfill_agent_logs(updates: list[dict[str, Any]]) -> None:
    if not updates:
        return
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("""
                    UPDATE agent_logs
                    SET username = COALESCE(username, :username),
                        full_name = COALESCE(full_name, :full_name),
                        position = COALESCE(position, :position),
                        point = COALESCE(point, :point),
                        area = COALESCE(area, :area),
                        regional = COALESCE(regional, :regional),
                        pulau = COALESCE(pulau, :pulau)
                    WHERE id = :id
                """),
                updates,
            )
    except Exception:
        pass


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
                   rewritten_query, or_cost, or_generation_id,
                   username, full_name, position, point, area, regional, pulau
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

    # Enrich historical logs missing geo/spreadsheet fields by matching
    # session_id -> user_kpi_data (by username or full_name) & branch_data (by point_norm).
    parsed_by_row: list[dict[str, str]] = []
    lookup_names: set[str] = set()
    lookup_usernames: set[str] = set()
    lookup_points: set[str] = set()

    for row in log_rows:
        sid = str(row[7]) if row[7] else ""
        meta = _parse_session_id_meta(sid)
        parsed_by_row.append(meta)
        db_username = str(row[22]).strip() if len(row) > 22 and row[22] else ""
        db_fullname = str(row[23]).strip() if len(row) > 23 and row[23] else ""
        db_point = str(row[25]).strip() if len(row) > 25 and row[25] else ""
        db_area = str(row[26]).strip() if len(row) > 26 and row[26] else ""
        db_reg = str(row[27]).strip() if len(row) > 27 and row[27] else ""
        if not db_area or not db_reg or not db_username:
            if db_username:
                lookup_usernames.add(db_username)
            name_cand = (db_fullname or meta["full_name"]).lower().strip()
            if name_cand:
                lookup_names.add(name_cand)
            pt_cand = (db_point or meta["point"]).lower().replace(" ", "").replace("_", "")
            if pt_cand:
                lookup_points.add(pt_cand)

    user_by_username: dict[str, dict[str, str]] = {}
    user_by_name: dict[str, dict[str, str]] = {}
    branch_by_point: dict[str, dict[str, str]] = {}

    if lookup_usernames or lookup_names or lookup_points:
        try:
            async with engine.connect() as conn:
                if lookup_usernames or lookup_names:
                    u_res = await conn.execute(
                        text("""
                            SELECT username, full_name, role, point_norm, data
                            FROM user_kpi_data
                            WHERE username = ANY(:unames)
                               OR LOWER(full_name) = ANY(:fnames)
                        """),
                        {
                            "unames": list(lookup_usernames) or [""],
                            "fnames": list(lookup_names) or [""],
                        },
                    )
                    for ur in u_res.fetchall():
                        geo = _extract_geo_from_json(ur[4])
                        u_info = {
                            "username": str(ur[0] or ""),
                            "full_name": str(ur[1] or ""),
                            "role": str(ur[2] or ""),
                            "point_norm": str(ur[3] or ""),
                            **geo,
                        }
                        if u_info["username"]:
                            user_by_username[u_info["username"]] = u_info
                        if u_info["full_name"]:
                            user_by_name[u_info["full_name"].lower().strip()] = u_info
                        if u_info["point_norm"]:
                            lookup_points.add(u_info["point_norm"])

                if lookup_points:
                    b_res = await conn.execute(
                        text("""
                            SELECT point, point_norm, data
                            FROM branch_data
                            WHERE point_norm = ANY(:pts)
                        """),
                        {"pts": list(lookup_points)},
                    )
                    for br in b_res.fetchall():
                        geo = _extract_geo_from_json(br[2])
                        pnorm = str(br[1] or "").strip() or str(br[0] or "").lower().replace(" ", "")
                        if pnorm:
                            branch_by_point[pnorm] = {
                                "point": str(br[0] or "") or geo["point"],
                                "area": geo["area"],
                                "regional": geo["regional"],
                                "pulau": geo["pulau"],
                            }
        except Exception:
            pass

    logs = []
    backfill_updates: list[dict[str, Any]] = []

    for row, meta in zip(log_rows, parsed_by_row):
        row_id = int(row[1])
        db_username = str(row[22]).strip() if len(row) > 22 and row[22] else ""
        db_fullname = str(row[23]).strip() if len(row) > 23 and row[23] else ""
        db_position = str(row[24]).strip() if len(row) > 24 and row[24] else ""
        db_point = str(row[25]).strip() if len(row) > 25 and row[25] else ""
        db_area = str(row[26]).strip() if len(row) > 26 and row[26] else ""
        db_regional = str(row[27]).strip() if len(row) > 27 and row[27] else ""
        db_pulau = str(row[28]).strip() if len(row) > 28 and row[28] else ""

        u_match = (
            user_by_username.get(db_username)
            or user_by_name.get((db_fullname or meta["full_name"]).lower().strip())
            or {}
        )
        pt_norm = (
            (db_point or u_match.get("point") or meta["point"])
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )
        b_match = branch_by_point.get(pt_norm) or {}

        final_username = db_username or u_match.get("username") or ""
        final_fullname = db_fullname or u_match.get("full_name") or meta["full_name"] or ""
        final_position = db_position or meta["position"] or u_match.get("role") or ""
        final_point = db_point or u_match.get("point") or b_match.get("point") or meta["point"] or ""
        final_area = db_area or u_match.get("area") or b_match.get("area") or ""
        final_regional = db_regional or u_match.get("regional") or b_match.get("regional") or ""
        final_pulau = db_pulau or u_match.get("pulau") or b_match.get("pulau") or ""

        if (
            (final_point and not db_point)
            or (final_area and not db_area)
            or (final_regional and not db_regional)
            or (final_username and not db_username)
        ):
            backfill_updates.append({
                "id": row_id,
                "username": final_username[:64] or None,
                "full_name": final_fullname[:255] or None,
                "position": final_position[:128] or None,
                "point": final_point[:64] or None,
                "area": final_area[:64] or None,
                "regional": final_regional[:64] or None,
                "pulau": final_pulau[:64] or None,
            })

        logs.append({
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
            "username": final_username,
            "full_name": final_fullname,
            "position": final_position,
            "point": final_point,
            "area": final_area,
            "regional": final_regional,
            "pulau": final_pulau,
        })

    if backfill_updates:
        asyncio.create_task(_backfill_agent_logs(backfill_updates))

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


@router.get("/prompts", summary="Get active system prompt strings from prompts.py")
async def get_system_prompts(
    _=Depends(verify_api_key),
) -> Dict[str, Any]:
    """Return active production system prompts and modular XML blocks directly from cag-lms-agent."""
    from app.llm import prompts

    return {
        "success": True,
        "source": "backend_live",
        "prompts": {
            "conversational": prompts.CONVERSATIONAL_PROMPT,
            "socratic": prompts.SOCRATIC_PROMPT,
            "chit_chat": prompts.CHIT_CHAT_PROMPT,
            "stm_summary": prompts.STM_SUMMARY_PROMPT,
            "ltm_analyst": prompts.LTM_LEARNING_SUMMARY_PROMPT,
        },
        "blocks": {
            "role": prompts.PERSONA,
            "output_contract": prompts.OUTPUT_CONTRACT,
            "grounding": prompts.GROUNDING,
            "response_guidelines": prompts.RESPONSE_GUIDELINES,
            "mentoring_voice": prompts.MENTORING_VOICE,
            "socratic_mode": prompts.SOCRATIC_MODE,
            "disambig": prompts.DISAMBIG,
        },
    }





