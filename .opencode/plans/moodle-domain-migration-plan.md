# Plan: Migrasi Domain Moodle Staging — ngrok → amarthapedia-staging.lifeatamartha.com

## Konteks

- Domain lama: `https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/` (ngrok free-tier, `https`)
- Domain baru: `http://amarthapedia-staging.lifeatamartha.com/` (staging, `http` — downgrade scheme)
- `.env:76` sudah diupdate ke domain baru → runtime lokal sudah pakai URL baru via `settings.moodle_api_url` (`app/config/settings.py:456` override via `.env`).
- Scope plan: rapikan sisa hardcode, CORS, komentar ngrok, verifikasi konektivitas, dan risiko `http` vs `https`.

---

## Temuan (read-only audit)

| Lokasi | Status | Catatan |
|---|---|---|
| `.env:76` `MOODLE_API_URL` | ✅ sudah baru | `http://amarthapedia-staging.../` |
| `.env:77` `MOODLE_API_TOKEN` | ✅ terbawa | `ec950af...` — validitas perlu cek live |
| `.env:106` `CORS_ALLOW_ORIGINS` | ❌ masih ngrok | Masih list `https://semiexpositive...ngrok-free.dev`, belum ada entry staging baru |
| `.env.example:92` | ❌ masih ngrok | Onboarding dev baru akan copy URL mati |
| `app/config/settings.py:39-48` `cors_allow_origins` default | ⚠️ tidak ada ngrok, tapi juga tidak ada staging | Default aman, override via env yang jadi masalah |
| `app/config/settings.py:450-456` `moodle_api_url` default + komentar | ❌ masih ngrok | Fallback jika `.env` tidak ter-mount akan balik ke ngrok |
| `app/main.py:235-258` komentar ngrok invariants | ⚠️ usang | Jelaskan kenapa tidak ada Host/IP allowlist — perlu rewrite untuk domain permanen |
| `app/static/script.js:98-99` `_shouldAddNgrokHeader()` + 6 call sites | ⚠️ dead code | `baseUrl.includes("ngrok")` selalu false sekarang, header `ngrok-skip-browser-warning` tidak terkirim lagi — aman tapi membingungkan |
| `app/api/routes/chat.py:1980,2245` komentar ngrok keepalive | ⚠️ usang | Referensi `ngrok/free proxies cut idle` |
| `plugin_moodle/` | ❓ perlu cek | `ai_trainer` & `knowledge_manager` tidak hardcode domain di repo, tapi setting plugin di DB Moodle staging mungkin masih simpan URL lama |

---

## Apa yang perlu diedit (3 file repo + 1 infra)

### 1. `app/config/settings.py:450-456` — ganti default + komentar
**Sebelum:**
```python
# Default points at the dev/free-tier ngrok tunnel ...
moodle_api_url: str = "https://semiexpositive-renaldo-unvindictively.ngrok-free.dev/"
```
**Sesudah (opsi A — direkomendasikan):**
```python
# Default for local dev — override MOODLE_API_URL in .env for staging/prod.
# Previously pointed at ngrok free-tier tunnel (removed 2026-08).
moodle_api_url: str = "http://amarthapedia-staging.lifeatamartha.com/"
```
**Opsi B (lebih strict):** jadikan required tanpa default: `moodle_api_url: str = Field(..., alias="MOODLE_API_URL")` — akan fail-fast kalau `.env` lupa di-set. Tanya user mau yang mana.

### 2. `.env.example:92` — sinkron dengan default baru
```
MOODLE_API_URL=http://amarthapedia-staging.lifeatamartha.com/
```

### 3. `.env:106` (dan env di server staging/prod) — CORS
Ganti entry ngrok dengan domain staging baru. Contoh:
```
CORS_ALLOW_ORIGINS='["http://localhost:8080", "https://academy.amartha.com", "http://amarthapedia-staging.lifeatamartha.com", "https://amarthapedia-staging.lifeatamartha.com", "http://localhost:3000", "http://localhost:8000", "https://ai-trainer.lifeatamartha.com", "https://ferdy-fadhil-lazuardi.my.id"]'
```
Catatan: ingest Moodle→API adalah server-to-server (tanpa `Origin`, tidak kena CORS — `app/main.py:251`), tapi browser fetch dari halaman Moodle ke API akan kena preflight jika tidak ada di allowlist.

### 4. `app/main.py:235-258` — rewrite komentar ngrok invariants
Hapus bullet ngrok, ganti dengan catatan domain permanen:
- Host-header validation sekarang **boleh** jika diperlukan
- IP allowlist / per-IP rate limit **boleh** dipertimbangkan (tidak lagi CGNAT ngrok)
- CSP/X-Frame tidak lagi perlu longgar untuk interstitial ngrok

### 5. `app/static/script.js:98-99` — bersihkan helper ngrok (opsional, low-risk)
Hapus `_shouldAddNgrokHeader` dan 6 call sites (`:140,355,394,798,870,1210`). Tidak fungsional, hanya mengurangi noise. Alternatif: biarkan — tidak merusak.

### 6. Komentar `app/api/routes/chat.py:1980,2245` — ganti `ngrok/free proxies` → `reverse proxy idle timeout`

### 7. Di luar repo — cek manual (tidak ada file yang diedit, tapi wajib verifikasi)
- **Moodle `config.php` `$CFG->wwwroot`** di server staging harus sudah `http://amarthapedia-staging.lifeatamartha.com` — kalau masih ngrok, `fileurl` di `moodle_markdown.py:111-115` akan balikin URL ngrok mati.
- **Knowledge Manager plugin** (`plugin_moodle/knowledge_manager/api.php`) — pastikan plugin sudah ter-install di site baru, kalau tidak ingest akan fallback ke `core_course_get_contents` (`moodle_markdown.py:86`).
- **Ava block/plugin setting di Moodle DB** — jika ada field `api_base_url` yang dulu isi `https://<ngrok>` atau `https://ai-trainer...`, update ke URL `cag-lms-agent` yang benar (arah Moodle→Agent, bukan Agent→Moodle).
- **DNS & firewall staging** — `amarthapedia-staging.lifeatamartha.com` harus resolvable dari container `cag-lms-api`/`worker`.

---

## Apa yang (tidak) akan rusak

| Area | Dampak |
|---|---|
| **Ingest / KB sync** | Aman jika token masih valid — sudah pakai `.env` baru. Risiko: `fileurl` masih ngrok jika `$CFG->wwwroot` belum ganti. |
| **Chat streaming** | Aman server-to-server. Browser chat via Moodle block perlu CORS entry baru. |
| **JWT auth** | Aman, tapi token lewat `http` plaintext di staging — jangan pakai di prod. |
| **CORS** | Tidak nge-block ingest, tapi bisa block dashboard/test-ui fetch dari domain staging. |
| **Ngrok header** | Tidak lagi dikirim — tidak perlu, tidak merusak. |

---

## Risiko `http` (bukan `https`)

1. **Mixed content block** — jika `cag-lms-agent` di-serve `https` lalu browser fetch `http` Moodle, browser modern block. Untuk staging internal tolerable, untuk prod harus `https`.
2. **Sniffing token** — `MOODLE_API_TOKEN` + `Authorization: Bearer <JWT>` lewat plaintext. Di VPC/staging tertutup masih ok, di internet publik tidak.
3. **Rekomendasi:** pasang TLS (Let's Encrypt / Cloudflare Tunnel / ALB) dan ganti semua referensi ke `https://amarthapedia-staging.lifeatamartha.com/` sebelum naik prod. Perubahan skema hanya ganti `http`→`https` di `.env` + `settings.py` default.

---

## Rencana Verifikasi (tanpa edit, bisa jalan sekarang)

```bash
# 1. Token & WS masih hidup di site baru
curl -s "http://amarthapedia-staging.lifeatamartha.com/webservice/rest/server.php?wstoken=$MOODLE_API_TOKEN&wsfunction=core_course_get_contents&moodlewsrestformat=json&courseid=3" | head -c 500

# 2. Knowledge Manager plugin kebawa migrasi?
curl -s "http://amarthapedia-staging.lifeatamartha.com/local/knowledge_manager/api.php?token=$MOODLE_API_TOKEN&include_hidden=0" | head -c 500

# 3. Dari dalam container (cek DNS/firewall)
docker compose exec api python -c "import urllib.request; print(urllib.request.urlopen('http://amarthapedia-staging.lifeatamartha.com/webservice/rest/server.php', timeout=5).status)"

# 4. Ingest dry-run
curl -s http://localhost:8011/api/v1/moodle/sections?course_id=3 | head

# 5. Cek browser console di Moodle staging saat buka Ava block — cari CORS error
```

Jika (1) balikin `invalidtoken` → token harus regenerate di `Site administration > Server > Web services > Manage tokens` di site baru.
Jika (2) 404 → reinstall `knowledge_manager` plugin di site baru (fallback masih jalan tapi lebih lambat).

---

## Urutan Eksekusi (saat keluar plan mode)

1. Tanya user: `settings.moodle_api_url` mau default baru atau jadi required (fail-fast)?
2. Patch 3 file repo: `settings.py`, `.env.example`, komentar `main.py` + `script.js`/`chat.py` (opsional).
3. Update `.env` staging/prod di server (bukan hanya lokal) — `CORS_ALLOW_ORIGINS`.
4. Verifikasi 5 cek di atas.
5. Buat follow-up ticket: pasang `https` di staging + ganti `http`→`https` di semua referensi.

---

## Pertanyaan untuk User

- Mau `moodle_api_url` jadi **required tanpa default** (lebih aman) atau tetap ada default baru `http://amarthapedia-staging...`?
- Domain staging final mau tetap `http` atau langsung ke `https`? Kalau langsung `https`, patch sekalian ke `https` biar tidak dua kali ganti.
- Ada env staging di server (bukan `.env` lokal) yang perlu diupdate manual? Jika ya, sebutkan lokasinya (docker swarm / ECS / env file di VPS).
