# Plan: Spreadsheet Sync — Hapus Data yang Dihapus di Apps Script (Orphan Cleanup)

## Konteks & Keluhan User

> Hapus data di Google Apps Script / Spreadsheet, sync `POST /ingest/spreadsheet/sync` sukses, tapi data lama masih ada di Postgres (`user_kpi_data` / `branch_data`) sehingga AI masih menjawab pakai data spreadsheet yang sudah dihapus.

Endpoint: `POST /ingest/spreadsheet/sync` → `app/api/routes/ingest.py:131` → `app/knowledge/sync_spreadsheet.py:sync_kpi_from_spreadsheet()`

## Root Cause (Read-Only Audit)

**`app/knowledge/sync_spreadsheet.py:59-132` hanya UPSERT, tidak pernah DELETE.**

- Alur: `httpx GET SPREADSHEET_SYNC_URL` → parse `data` (`dict {users, branches}` atau `list` flat) → loop `insert(...).on_conflict_do_update(index_elements=[username/point])` → `await session.commit()` → return `users_updated/branches_updated`.
- Tidak ada: `DELETE WHERE username NOT IN (...)`, `DELETE WHERE point NOT IN (...)`, truncate, atau diff logic.
- Konsekuensi: baris yang sudah tidak ada di Spreadsheet tetap hidup selamanya di Postgres. Di `app/api/routes/chat.py:611,617` AI akan tetap load via `select(UserKPIData).where(username==...)` / `select(BranchData).where(point==...)` dan inject ke `user_context`.

**Yang tidak jadi masalah:**
- Upsert sendiri benar (idempotent, tidak duplikat).
- `data` payload di-derived dari row itu sendiri (exclude_keys), jadi tidak ada stale field di dalam JSON — masalahnya di **baris hilang**, bukan field hilang.

**File terkait:**
- Source: `app/knowledge/sync_spreadsheet.py` (139 baris)
- Model: `app/database/models.py:193-222` (`UserKPIData.username PK`, `BranchData.point PK`)
- Caller: `app/api/routes/ingest.py:131-160` (sync tanpa `job_id`, commit di dalam `sync_spreadsheet.py`)
- Konsumen: `app/api/routes/chat.py:602-628` (load per-request)
- Test: `tests/test_kpi_sync.py` (hanya cek `success`/`skipped`, tidak cek delete)

## Pertanyaan untuk User (Putuskan Sebelum Implement)

1. **Apakah delete di Spreadsheet = delete permanen di Postgres?** Atau butuh soft-delete / arsip? (Rekomendasi: hard delete, karena spreadsheet adalah source of truth mingguan. Jika butuh audit, tambahkan `deleted_at` tapi tidak disarankan untuk KPI mingguan.)
2. **Apakah spreadsheet bisa kosong sesaat (misal sheet belum keisi)?** Jika sync kosong lalu kita `DELETE WHERE NOT IN (empty)` akan wipe semua data. Perlu guard: jika `users_list` dan `branches_list` kosong → skip delete atau treat sebagai `skipped`.
3. **Apakah ada data manual di Postgres yang tidak boleh kehapus (seed manual)?** Jika ada, full-replace akan menghapusnya. Konfirmasi apakah semua data di 2 tabel ini 100% berasal dari spreadsheet.
4. **Mau mode `replace` selalu aktif atau via flag `?hard_delete=true`?** Rekomendasi: always-on dengan guard kosong + log `deleted_users/deleted_branches`.

Asumsi default jika user tidak jawab: hard delete selalu aktif, skip delete jika payload kosong, log jumlah deleted.

## Opsi Desain

### Opsi A — Full Replace (Direkomendasikan, simple)
Setelah upsert semua row yang ada di payload, delete orphan:
```python
incoming_usernames = {u for u in parsed usernames}
incoming_points = {p for p in parsed points}
if users_list:  # guard: jangan wipe kalau sheet kosong
    await session.execute(delete(UserKPIData).where(UserKPIData.username.notin_(incoming_usernames)))
if branches_list:
    await session.execute(delete(BranchData).where(BranchData.point.notin_(incoming_points)))
```
- Pro: 1 round-trip DELETE per tabel, idempotent, cocok untuk KPI mingguan yang memang full snapshot.
- Kontra: jika Apps Script bug kirim partial (misal 1 user karena filter), akan hapus sisanya. Mitigasi: guard `if len(incoming) == 0: skip delete` + log warning.

### Opsi B — Diff + Soft Delete
Tambah kolom `is_deleted` / `deleted_at` dan set flag alih-alih DELETE. Chat loader filter `where is_deleted == false`.
- Pro: reversible, audit trail.
- Kontra: perlu migration, loader harus diupdate, data stale tetap numpuk.

### Opsi C — Truncate + Re-insert
`DELETE FROM user_kpi_data; DELETE FROM branch_data;` lalu insert semua. Paling sederhana tapi window kosong sesaat (tidak transaksional jika tidak dalam 1 tx) dan rowcount log jadi tidak akurat.

**Rekomendasi: Opsi A** — full replace dengan `NOT IN` dalam transaksi yang sama (sebelum `commit()`), plus return `users_deleted/branches_deleted` di response.

## Rencana Implementasi (Saat Keluar Plan Mode)

1. **Patch `app/knowledge/sync_spreadsheet.py`:**
   - Kumpulkan `incoming_usernames: set[str]` dan `incoming_points: set[str]` selama loop upsert (sudah ada `username`/`point` yang di-parse).
   - Setelah loop, `from sqlalchemy import delete` lalu:
     ```python
     users_deleted = branches_deleted = 0
     if users_list and incoming_usernames:
         res = await session.execute(delete(UserKPIData).where(UserKPIData.username.notin_(incoming_usernames)))
         users_deleted = res.rowcount or 0
     if branches_list and incoming_points:
         res = await session.execute(delete(BranchData).where(BranchData.point.notin_(incoming_points)))
         branches_deleted = res.rowcount or 0
     ```
   - Guard tambahan: jika `data` adalah `dict` tapi `users`/`branches` key tidak ada → `users_list = []` → skip delete untuk tabel itu (jangan wipe karena sheet format berubah).
   - Update return dict: tambah `users_deleted`, `branches_deleted`.
   - Pindahkan `await session.commit()` ke setelah delete (sudah di situ, jaga agar DELETE dan UPSERT 1 transaksi).

2. **Patch `app/api/routes/ingest.py:120-160`:**
   - Extend `SpreadsheetSyncResponse` dengan `users_deleted: int`, `branches_deleted: int`.
   - Forward dari `result.get(...)`.
   - Log: `logger.info(f"Spreadsheet sync complete. Users: +{users_updated}/-{users_deleted}, Branches: +{branches_updated}/-{branches_deleted}")`

3. **Patch `tests/test_kpi_sync.py`:**
   - Tambah test `test_sync_deletes_orphans`: seed `FakeSession` dengan existing rows, mock payload hanya 1 user, assert `DELETE` statement tereksekusi dan `users_deleted == 1`.
   - Tambah test guard kosong: payload `{"users": [], "branches": []}` → `deleted == 0`, tidak ada DELETE.

4. **Opsional — Apps Script audit:** cek apakah GAS Web App memang return full snapshot atau incremental. Jika incremental, Opsi A tidak cocok — konfirmasi ke owner sheet.

## Verifikasi

- **Unit:** `pytest tests/test_kpi_sync.py -v` — 4 test (2 lama + 2 baru) harus pass.
- **Manual staging:**
  1. Isi sheet dengan 2 user (A,B) → sync → cek `SELECT username FROM user_kpi_data` → 2 rows.
  2. Hapus B di sheet → sync → `SELECT` harus 1 row (A), B hilang. Cek response `users_deleted: 1`.
  3. Hapus semua / kosongkan sheet → sync → tidak wipe semua (guard), `users_deleted: 0`.
  4. Tanya AI dengan user B → seharusnya tidak lagi dapat KPI B (chat.py tidak inject).
- **No regression:** `git diff --stat` hanya sentuh 3 file, tidak ubah `chat.py` loader.

## Risiko & Mitigasi

- **Partial payload wipe:** Guard `if not users_list: skip delete` + log warning jika `incoming_usernames` jauh lebih kecil dari `SELECT COUNT(*) FROM user_kpi_data` (misal <50% dari existing).
- **Case-sensitivity:** `username`/`point` di-parse via `str(v).strip()` tapi delete pakai exact match. Sudah konsisten dengan upsert (juga `str(v).strip()`). Normalized fallback di `chat.py:620` (`lower().replace(" ", "")`) hanya untuk read, tidak untuk delete — aman, tapi pertimbangkan normalisasi delete juga jika sheet kadang ada spasi.
- **Transaksi:** DELETE + UPSERT dalam 1 `commit()` → atomic. Jika DELETE gagal, UPSERT juga rollback — aman.

## Yang Tidak Perlu Diubah

- `app/database/models.py` — tidak perlu kolom baru untuk Opsi A.
- `app/api/routes/chat.py` — loader sudah benar, akan otomatis tidak menemukan orphan setelah di-delete.
- `.env` / `SPREADSHEET_SYNC_URL` — tidak ada perubahan domain seperti kasus Moodle sebelumnya.

