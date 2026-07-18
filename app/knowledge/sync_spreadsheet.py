import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert
from loguru import logger

from app.config.settings import get_settings
from app.database.models import UserKPIData, BranchData


async def sync_kpi_from_spreadsheet(session: AsyncSession) -> dict:
    """
    Fetches weekly KPI and Branch data from Google Apps Script Web App JSON endpoint
    and upserts it into PostgreSQL database.
    """
    settings = get_settings()
    url = settings.spreadsheet_sync_url
    token = settings.spreadsheet_sync_token

    if not url:
        logger.warning("SPREADSHEET_SYNC_URL is not configured. Skipping spreadsheet sync.")
        return {"status": "skipped", "message": "URL not configured"}

    logger.info(f"Starting spreadsheet sync from GAS Web App: {url}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params={"token": token}, follow_redirects=True, timeout=30.0)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.error(f"Failed to fetch spreadsheet data from GAS Web App: {exc}")
        return {"status": "failed", "message": str(exc)}

    users_updated = 0
    branches_updated = 0

    # The data can be a dict with {"users": [...], "branches": [...]} or a flat list
    users_list = []
    branches_list = []

    if isinstance(data, dict):
        users_list = data.get("users", [])
        branches_list = data.get("branches", [])
    elif isinstance(data, list):
        for item in data:
            # Check for user identifier (case-insensitive)
            has_user = any(k.lower().strip() in ("user_id", "username", "nik", "user_name") for k in item.keys() if item[k])
            if has_user:
                users_list.append(item)
            # Check for point or cabang key to store branch data (case-insensitive)
            has_branch = any(k.lower().strip() in ("point", "cabang") for k in item.keys() if item[k])
            if has_branch:
                branches_list.append(item)
    else:
        logger.error(f"Invalid spreadsheet data format returned. Expected list or dict, got: {type(data)}")
        return {"status": "failed", "message": "invalid data format"}

    # 1. Upsert User KPI Data
    for row in users_list:
        try:
            username = ""
            username_key = None
            for k, v in row.items():
                if k.lower().strip() in ("user_id", "username", "nik", "user_name"):
                    username = str(v).strip()
                    username_key = k
                    break

            if not username:
                continue

            exclude_keys = {"full_name"}
            if username_key:
                exclude_keys.add(username_key)
            data_payload = {k: v for k, v in row.items() if k not in exclude_keys}

            stmt = insert(UserKPIData).values(
                username=username,
                full_name=row.get("full_name"),
                data=data_payload,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["username"],
                set_={
                    "full_name": stmt.excluded.full_name,
                    "data": stmt.excluded.data,
                }
            )
            await session.execute(stmt)
            users_updated += 1
        except Exception as e:
            logger.warning(f"Error parsing user row {row}: {e}")

    # 2. Upsert Branch Data
    for row in branches_list:
        try:
            point = ""
            point_key = None
            for k, v in row.items():
                if k.lower().strip() in ("point", "cabang"):
                    point = str(v).strip()
                    point_key = k
                    break

            if not point:
                continue

            exclude_keys = {"nama_cabang", "full_name"}
            if point_key:
                exclude_keys.add(point_key)
            data_payload = {k: v for k, v in row.items() if k not in exclude_keys}

            stmt = insert(BranchData).values(
                point=point,
                nama_cabang=row.get("nama_cabang") or row.get("full_name") or point,
                data=data_payload,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["point"],
                set_={
                    "nama_cabang": stmt.excluded.nama_cabang,
                    "data": stmt.excluded.data,
                }
            )
            await session.execute(stmt)
            branches_updated += 1
        except Exception as e:
            logger.warning(f"Error parsing branch row {row}: {e}")


    await session.commit()
    logger.info(f"Spreadsheet sync complete. Users updated: {users_updated}, Branches updated: {branches_updated}")
    
    return {
        "status": "success",
        "users_updated": users_updated,
        "branches_updated": branches_updated,
    }
