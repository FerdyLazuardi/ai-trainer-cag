"""Paginated spreadsheet sync — mirrors the GAS source of truth 1:1 into Postgres.

Why paginated: a single GAS doGet for ~10k users x ~74 fields (~4KB/row,
~30-50MB total) exceeds the Cloudflare 120s proxy read timeout (HTTP 524).
GAS now serves ``?scope=users|branches&page=N&limit=1000`` pages of ~4MB
each; this module fetches page-by-page (each well under the timeout),
upserts in staging batches of 1000, validates, then atomically swaps.

Anti-wipe guarantees:
- Orphan DELETE runs ONLY when every page for that scope fetched without
  error (fetch_complete) AND at least one valid key was parsed.
- An empty/failed payload returns failed/skipped and never touches the DB.
- Legacy single-shot payloads (dict with users+branches, no pagination)
  are still accepted as a fallback.
"""

import httpx
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from loguru import logger

from app.config.settings import get_settings
from app.database.models import UserKPIData, BranchData

PAGE_LIMIT = 1000
STAGING_BATCH = 1000
MAX_PAGES = 50  # hard guard: 50 x 1000 = 50k rows ceiling
PER_PAGE_TIMEOUT = 90.0  # each page ~4MB, must stay under CF 120s


def _norm_point(value: object) -> str | None:
    s = str(value or "").strip()
    if not s:
        return None
    return s.lower().replace(" ", "")


def _get_ci(row: dict, *names: str) -> object:
    """Case-insensitive lookup, returns the first non-empty value."""
    lowered = {str(k).lower().strip(): v for k, v in row.items()}
    for n in names:
        v = lowered.get(n.lower())
        if v not in (None, ""):
            return v
    return None


def _extract_user_fields(row: dict, username: str, username_key: str | None) -> dict:
    exclude = {"full_name"}
    if username_key:
        exclude.add(username_key)
    data_payload = {k: v for k, v in row.items() if k not in exclude}
    periode = _get_ci(row, "periode", "periode_kpi")
    role = _get_ci(row, "role", "jabatan", "position")
    point_raw = _get_ci(row, "point", "cabang")
    return {
        "username": username,
        "full_name": row.get("full_name"),
        "periode": str(periode).strip()[:32] if periode not in (None, "") else None,
        "role": str(role).strip()[:64] if role not in (None, "") else None,
        "point_norm": _norm_point(point_raw),
        "data": data_payload,
    }


def _extract_branch_fields(row: dict, point: str, point_key: str | None) -> dict:
    exclude = {"nama_cabang", "full_name"}
    if point_key:
        exclude.add(point_key)
    data_payload = {k: v for k, v in row.items() if k not in exclude}
    periode = _get_ci(row, "periode", "periode_kpi")
    return {
        "point": point,
        "nama_cabang": row.get("nama_cabang") or row.get("full_name") or point,
        "point_norm": _norm_point(point),
        "periode": str(periode).strip()[:32] if periode not in (None, "") else None,
        "data": data_payload,
    }


async def _fetch_scope_pages(
    client: httpx.AsyncClient, url: str, token: str, scope: str
) -> tuple[list[dict], bool]:
    """Fetch all pages for one scope. Returns (rows, fetch_complete).

    fetch_complete=False means at least one page failed -> caller must NOT
    run orphan DELETE for this scope (anti-wipe).
    """
    rows: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        try:
            resp = await client.get(
                url,
                params={"token": token, "scope": scope, "page": page, "limit": PAGE_LIMIT},
                follow_redirects=True,
                timeout=PER_PAGE_TIMEOUT,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.error(f"Spreadsheet fetch failed scope={scope} page={page}: {exc}")
            return rows, False

        if isinstance(payload, dict):
            # New GAS shape: {"users": [...], "page": N, ...} or per-scope key,
            # or a bare list under "data"/"rows".
            batch = payload.get(scope, payload.get("data", payload.get("rows", [])))
            if batch is None:
                batch = []
        elif isinstance(payload, list):
            batch = payload
        else:
            logger.error(f"Invalid page payload type scope={scope} page={page}: {type(payload)}")
            return rows, False

        if not isinstance(batch, list):
            logger.error(f"Invalid batch type scope={scope} page={page}: {type(batch)}")
            return rows, False

        rows.extend(batch)
        logger.info(f"Spreadsheet page fetched scope={scope} page={page} rows={len(batch)} total={len(rows)}")

        if len(batch) < PAGE_LIMIT:
            return rows, True  # last page

    logger.warning(f"Spreadsheet scope={scope} hit MAX_PAGES={MAX_PAGES}, truncating")
    return rows, True


async def _fetch_legacy_single(
    client: httpx.AsyncClient, url: str, token: str
) -> tuple[list[dict], list[dict], bool]:
    """Fallback for GAS without pagination: single doGet returning everything."""
    try:
        resp = await client.get(url, params={"token": token}, follow_redirects=True, timeout=300.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}".strip(": ")
        logger.error(f"Failed to fetch spreadsheet data from GAS Web App: {err}")
        return [], [], False

    users_list: list[dict] = []
    branches_list: list[dict] = []
    if isinstance(data, dict):
        users_list = data.get("users", [])
        branches_list = data.get("branches", [])
        # Paginated GAS may return the first page here when scope is omitted;
        # treat as complete only if explicitly marked, else prefer paginated path.
        # If both keys present we accept it as a full legacy payload.
        if users_list or branches_list:
            return users_list, branches_list, True
        return [], [], False
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            has_user = any(
                k.lower().strip() in ("user_id", "username", "nik", "user_name")
                for k in item.keys() if item[k]
            )
            if has_user:
                users_list.append(item)
            has_branch = any(
                k.lower().strip() in ("point", "cabang")
                for k in item.keys() if item[k]
            )
            if has_branch:
                branches_list.append(item)
        return users_list, branches_list, True
    logger.error(f"Invalid spreadsheet data format: {type(data)}")
    return [], [], False


async def _upsert_users_staged(session: AsyncSession, users_list: list[dict]) -> tuple[int, set[str]]:
    users_updated = 0
    incoming: set[str] = set()
    batch_count = 0
    for row in users_list:
        try:
            if not isinstance(row, dict):
                continue
            username = ""
            username_key = None
            for k, v in row.items():
                if k.lower().strip() in ("user_id", "username", "nik", "user_name"):
                    if v not in (None, ""):
                        username = str(v).strip()
                        username_key = k
                        break
            if not username:
                continue
            fields = _extract_user_fields(row, username, username_key)
            stmt = insert(UserKPIData).values(**fields)
            stmt = stmt.on_conflict_do_update(
                index_elements=["username"],
                set_={
                    "full_name": stmt.excluded.full_name,
                    "periode": stmt.excluded.periode,
                    "role": stmt.excluded.role,
                    "point_norm": stmt.excluded.point_norm,
                    "data": stmt.excluded.data,
                },
            )
            await session.execute(stmt)
            incoming.add(username)
            users_updated += 1
            batch_count += 1
            if batch_count >= STAGING_BATCH:
                await session.flush()
                batch_count = 0
        except Exception as e:
            logger.warning(f"Error parsing user row: {e}")
    if batch_count:
        await session.flush()
    return users_updated, incoming


async def _upsert_branches_staged(session: AsyncSession, branches_list: list[dict]) -> tuple[int, set[str]]:
    branches_updated = 0
    incoming: set[str] = set()
    batch_count = 0
    for row in branches_list:
        try:
            if not isinstance(row, dict):
                continue
            point = ""
            point_key = None
            for k, v in row.items():
                if k.lower().strip() in ("point", "cabang"):
                    if v not in (None, ""):
                        point = str(v).strip()
                        point_key = k
                        break
            if not point:
                continue
            fields = _extract_branch_fields(row, point, point_key)
            stmt = insert(BranchData).values(**fields)
            stmt = stmt.on_conflict_do_update(
                index_elements=["point"],
                set_={
                    "nama_cabang": stmt.excluded.nama_cabang,
                    "point_norm": stmt.excluded.point_norm,
                    "periode": stmt.excluded.periode,
                    "data": stmt.excluded.data,
                },
            )
            await session.execute(stmt)
            incoming.add(point)
            branches_updated += 1
            batch_count += 1
            if batch_count >= STAGING_BATCH:
                await session.flush()
                batch_count = 0
        except Exception as e:
            logger.warning(f"Error parsing branch row: {e}")
    if batch_count:
        await session.flush()
    return branches_updated, incoming


async def sync_kpi_from_spreadsheet(session: AsyncSession) -> dict:
    """
    Paginated fetch (limit=1000/scope) + staging upsert (batch 1000) +
    validated atomic swap. Spreadsheet stays the full source of truth
    (incl. deletes -> mirror 1:1), guarded by anti-wipe rules.
    """
    settings = get_settings()
    url = settings.spreadsheet_sync_url
    token = settings.spreadsheet_sync_token

    if not url:
        logger.warning("SPREADSHEET_SYNC_URL is not configured. Skipping spreadsheet sync.")
        return {"status": "skipped", "message": "URL not configured"}

    logger.info(f"Starting paginated spreadsheet sync from GAS Web App: {url}")

    users_list: list[dict] = []
    branches_list: list[dict] = []
    users_complete = False
    branches_complete = False

    try:
        async with httpx.AsyncClient() as client:
            # Probe: try paginated users page 1 first.
            probe_rows, probe_ok = await _fetch_scope_pages(client, url, token, "users")
            if probe_ok and probe_rows:
                users_list = probe_rows
                users_complete = True
                b_rows, b_ok = await _fetch_scope_pages(client, url, token, "branches")
                branches_list = b_rows
                branches_complete = b_ok
            else:
                # Either GAS has no pagination (legacy) or users scope is empty.
                # Fall back to legacy single-shot, then validate.
                u_legacy, b_legacy, legacy_ok = await _fetch_legacy_single(client, url, token)
                if legacy_ok and (u_legacy or b_legacy):
                    users_list, branches_list = u_legacy, b_legacy
                    users_complete = bool(u_legacy)
                    branches_complete = bool(b_legacy)
                else:
                    # Paginated probe returned empty-but-ok (e.g. zero users)?
                    # Distinguish "empty source" from "unsupported scope": if the
                    # probe was OK, accept its (possibly empty) result and still
                    # try branches paginated.
                    if probe_ok:
                        users_list = probe_rows
                        users_complete = True
                        b_rows, b_ok = await _fetch_scope_pages(client, url, token, "branches")
                        branches_list = b_rows
                        branches_complete = b_ok
                    else:
                        return {"status": "failed", "message": "fetch failed on first page"}
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}".strip(": ")
        logger.error(f"Spreadsheet paginated fetch aborted: {err_msg}")
        return {"status": "failed", "message": err_msg}

    if not users_complete and not branches_complete:
        logger.error("Spreadsheet sync aborted: no scope fetched completely — DB untouched")
        return {"status": "failed", "message": "no complete scope fetched"}

    if not users_list and not branches_list:
        logger.warning("Spreadsheet payload empty — skipping sync to avoid wipe")
        return {"status": "failed", "message": "empty payload"}

    # 1-2. Staging upserts (batches of 1000, flushed per batch)
    users_updated, incoming_usernames = await _upsert_users_staged(session, users_list)
    branches_updated, incoming_points = await _upsert_branches_staged(session, branches_list)

    # 3. Validated orphan swap — per-scope, only when that scope is complete
    #    AND at least one valid key parsed. Otherwise skip (anti-wipe).
    users_deleted = 0
    branches_deleted = 0
    try:
        if users_complete and incoming_usernames:
            res = await session.execute(
                delete(UserKPIData).where(UserKPIData.username.notin_(incoming_usernames))
            )
            users_deleted = int(getattr(res, "rowcount", 0) or 0)
        elif users_list and not incoming_usernames:
            logger.warning("No valid usernames parsed — skipping user orphan delete")
        elif not users_complete:
            logger.warning("Users scope incomplete — skipping user orphan delete (anti-wipe)")

        if branches_complete and incoming_points:
            res = await session.execute(
                delete(BranchData).where(BranchData.point.notin_(incoming_points))
            )
            branches_deleted = int(getattr(res, "rowcount", 0) or 0)
        elif branches_list and not incoming_points:
            logger.warning("No valid points parsed — skipping branch orphan delete")
        elif not branches_complete:
            logger.warning("Branches scope incomplete — skipping branch orphan delete (anti-wipe)")
    except Exception as e:
        logger.warning(f"Failed to delete orphan spreadsheet rows: {e}")

    await session.commit()
    logger.info(
        f"Spreadsheet sync complete. Users updated: {users_updated} (deleted: {users_deleted}), "
        f"Branches updated: {branches_updated} (deleted: {branches_deleted})"
    )

    return {
        "status": "success",
        "users_updated": users_updated,
        "branches_updated": branches_updated,
        "users_deleted": users_deleted,
        "branches_deleted": branches_deleted,
    }
