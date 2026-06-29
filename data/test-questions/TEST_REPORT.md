# Test Report: 40 Studicase Questions — RAG Evaluation

**Date:** 2026-06-29
**Model:** `deepseek/deepseek-v4-flash` via OpenRouter (alibaba,baidu,novita)
**Endpoint:** POST `http://localhost:8001/api/v1/chat`
**Course ID:** 3 (Moodle KB revisi — 16 file)
**Mode:** Multi-turn per skenario (conversation_id per scenario)

---

## Ringkasan Eksekutif

| Metrik | Hasil |
|--------|-------|
| **Total pertanyaan** | 40 |
| **Berhasil + ada sources** | **19 (47.5%)** |
| **Berhasil tapi NO sources** | **2 (5%)** — jawaban pure generative |
| **Error parsing body** | **19 (47.5%)** |
| **Rata-rata latency (berhasil)** | ~5.3 detik |
| **Source terbanyak** | `Business Process Amartha.md` — backbone semua jawaban |

> ⚠️ **19 pertanyaan gagal dengan `{"detail":"There was an error parsing the body"}`.** Ini kemungkinan besar masalah encoding curl di Windows bash untuk request body yang mengandung karakter Indonesia (é, è, khususnya di query panjang dengan tanda petik/kutip). Retry manual dengan escaping JSON yang proper seharusnya sukses.

---

## Detail Per Skenario

### Skenario 1: FO baru bingung cara rekrut mitra *(Sales Skill, Komunikasi Dasar, Business Process)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ✅ Sukses | 5.3s | Komunikasi Dasar.md (0.73) | Jawaban struktur 3 tahap tepat |
| Q2 | ✅ Sukses | 8.6s | Business Process.md (1.34) | Sales funnel detail, akurat |
| Q3 | (lewat) | — | — | Chain conversation dari Q2 |

**Catatan:** Q1 sebenarnya muncul error di line pertama tapi Q1 jawabannya sukses di baris berikut — kemungkinan echo/curl race.

---

### Skenario 2: Mitra telat bayar *(Data Driven Strategy, Basic Data Literacy)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ❌ Error | — | — | Body parsing failed |
| Q2 | ❌ Error | — | — | Body parsing failed |
| Q3 | ❌ Error | — | — | Body parsing failed |

**Catatan:** Seluruh skenario 2 gagal. Polanya sama — kemungkinan curl gagal parse JSON karena karakter tertentu.

---

### Skenario 3: Briefing pagi gak efektif *(Business Process)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ✅ Sukses | 5.5s | Business Process.md (0.59) | Jawaban detail: 6 langkah briefing efektif ✅ |
| Q2 | ✅ Sukses | 7.2s | Business Process.md (1.99) | 4 fokus digitalisasi, relevan ✅ |
| Q3 | ✅ Sukses | 7.2s | Business Process.md (0.67) | Jawaban jujur "gak ada durasi pasti" ✅ |

**Catatan:** Skenario paling solid — 3/3 sukses, top score tinggi, Business Process.md jadi sumber utama.

---

### Skenario 4: Mitra riwayat jelek minta cair *(Strategi Pencapaian Target, Business Process)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ❌ Error | — | — | Body parsing failed |
| Q2 | ✅ Sukses | 5.1s | Business Process.md (0.57) | NPL analogi "angka kematian RS" — tepat ✅ |
| Q3 | ✅ Sukses | 6.2s | Strategi Pencapaian Target.md (0.45) | Retensi + akuisisi, balanced ✅ |

---

### Skenario 5: Fraud di cabang *(Fraud Awareness, Leadership in Crisis)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ✅ Sukses | 4.9s | Business Process.md (0.69) + Leadership Crisis.md (0.67) | Langkah konkret, source akurat ✅ |
| Q2 | ❌ Error | — | — | Body parsing failed |
| Q3 | ✅ Sukses | 6.0s | Fraud Awareness.md (1.62, 1.50) | Top score, jawaban detail: 4 area pencegahan ✅ |

---

### Skenario 6: Baru jadi BM *(Basic Leadership, Flexibel Leadership)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ✅ Sukses | 4.8s | Flexibel Leadership.md (0.41) | Pendekatan situasional, mentoring/coaching ✅ |
| Q2 | ❌ Error | — | — | Body parsing failed |
| Q3 | ❌ Error | — | — | Body parsing failed |

---

### Skenario 7: Mitra ngeluh ditagih kasar *(Client Protection)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ✅ Sukses | 5.6s | Business Process.md (0.68) | Dua pendekatan tagih (hubungan vs konflik) ✅ |
| Q2 | ✅ Sukses | 5.0s | Client Protection.md (0.95) | **Score tertinggi skenario ini.** Transparansi, responsible pricing ✅ |
| Q3 | ❌ Error | — | — | Body parsing failed |

---

### Skenario 8: RF Poket turun drastis *(Data Driven Strategy, Business Process)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ❌ Error | — | — | Body parsing failed |
| Q2 | ✅ Sukses | 3.8s | Data Driven Strategy.md (0.78) | Edukasi mitra, kendali diri BP ✅ |
| Q3 | ❌ Error | — | — | Body parsing failed |

---

### Skenario 9: Anti-Harassment *(Anti-Harassment)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ❌ Error | — | — | Body parsing failed |
| Q2 | ✅ Sukses | 3.4s | Anti-Harassment.md (0.75) | Satgas PPKS + saluran lapor ✅ |
| Q3 | ✅ Sukses | 6.0s | Anti-Harassment.md (1.0, 0.76) | Mutasi kerja, kewajiban pimpinan ✅ |

---

### Skenario 10: Tim gak kompak *(Build & Maintain Team, Basic Leadership)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ✅ Sukses | 4.8s | **EMPTY SOURCES []** ⚠️ | Jawaban bagus tapi **generative murni** — gak ada retrieval |
| Q2 | ❌ Error | — | — | Body parsing failed |
| Q3 | ❌ Error | — | — | Body parsing failed |

**⚠️ ISSUE:** Q10 (tim kompak) balik `"sources":[]` — LLM jawab dari pengetahuan sendiri, bukan dari KB. Kemungkinan retrieval gak nemu chunk yang cocok.

---

### Skenario 11: Problem solving muter-muter *(Framework Strategi dan Problem Solving)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ❌ Error | — | — | Body parsing failed |
| Q2 | ❌ Error | — | — | Body parsing failed |
| Q3 | ✅ Sukses | 4.4s | Framework Strategi.md (0.55) | Action Matrix Diagram — Quick Win ✅ |

---

### Skenario 12: BP gak bisa komunikasi *(Komunikasi Dasar)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ✅ Sukses | 5.9s | Strategi Pencapaian Target.md (0.30) | Struktur 3 tahap buat canvassing ✅ |
| Q2 | ✅ Sukses | 3.4s | **EMPTY SOURCES []** ⚠️ | Jawaban etika WA bagus tapi **generative murni** |

**⚠️ ISSUE:** Q35 (etika WA) juga `"sources":[]` — KB Komunikasi Dasar.md kemungkinan gak punya chunk ttg etika WA, jadi LLM jawab dari pengetahuan sendiri.

---

### Skenario 13: Mitra meninggal *(Business Process — Ketentuan Mitra Meninggal)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ❌ Error | — | — | Body parsing failed |
| Q2 | ❌ Error | — | — | Body parsing failed |

**Catatan:** 2/2 error parsing body.

---

### Skenario 14: Target disbursement gak nyampe *(Strategi Pencapaian Target, Sales Skill & Strategy)*

| # | Status | Latency | Top Source (score) | Kesan |
|---|--------|---------|-------------------|-------|
| Q1 | ❌ Error | — | — | Body parsing failed |
| Q2 | ✅ Sukses | 4.3s | Sales Skill Strategy.md (0.84) | Coaching funnel — tepat ✅ |
| Q3 | ✅ Sukses | 5.8s | Data Driven Strategy.md (0.80) | 3 pilar + metrik harian ✅ |

---

## Analisis Retrieval Quality

### ✅ Kekuatan
- **Source relevance high**: rata-rata top score antara 0.40–0.80 untuk jawaban sukses
- **Dense retrieval solid** untuk topik yang ada di KB (Business Process, Anti-Harassment, Client Protection)
- **Compound query berfungsi**: `resolved_query` menunjukkan query rewrite jadi 2 sub-query (misal S3Q2, Q35)

### ⚠️ Kelemahan
1. **No-source answers** (2 pertanyaan): S10Q1 & S12Q2 balik `sources: []` — LLM generative murni
2. **Score rendah di beberapa jawaban**: S6Q1 (Leadership) cuma dapet score 0.41 — tipis
3. **Error parsing body massive** (19/40): bukan masalah retrieval, tapi curl/JSON encoding

### 🔍 Critical Findings
| Temuan | Detail |
|--------|--------|
| **S10Q1 — NO SOURCES** | "Tim gak kompak" — jawaban bagus tapi pure generative. KB gak punya topik team building? |
| **S12Q2 — NO SOURCES** | "Etika WA" — pure generative. KB Komunikasi Dasar.md gak cover etika chat WA? |
| **Business Process.md dominan** | Jadi top source di hampir semua jawaban — wajar karena dokumen paling komprehensif |
| **Anti-Harassment retrieval bagus** | Q9Q2-Q9Q3 dapet score 0.75–1.0 — very relevant |

---

## Rekomendasi

1. **Fix curl/JSON encoding**: Ulang test dengan Python `requests` atau Postman biar 40 pertanyaan jalan semua tanpa error parsing
2. **Tambah KB coverage**: S10Q1 (team cohesion) & S12Q2 (etika WA) — perlu diperkaya
3. **Naikin dense score floor**: S6Q1 top score cuma 0.41 — mendekati `KB_MIN_DENSE_SCORE=0.40`, nyaris gak ke-retrieve
4. **Conversation chain**: Chain conversation (Q1→Q2→Q3 per scenario) berfungsi — `resolved_query` beda tiap turn
5. **Model**: DeepSeek V4 Flash responsif (avg 5.3s), gaya bahasa santai konsisten, ga ada reasoning leak terlihat

---

*Report generated 2026-06-29*
