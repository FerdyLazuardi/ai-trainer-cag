# Laporan Testing AI Agent — Knowledge Base Verification

**Tanggal:** 2025-07-15  
**Endpoint:** `POST /api/v1/chat` (`localhost:8001`)  
**Knowledge Base:** `_moodle_md/revisi/*.md` (16 modul)  
**Scope:** Verifikasi apakah jawaban AI Agent sesuai dengan source dokumen

---

## Ringkasan

| Item | Hasil |
|------|-------|
| Modul dites | **16/16** |
| Jawaban akurat | **16/16 ✅** |
| Rata-rata source relevance score | **0.83** |
| Rata-rata latency | ~8.5s |
| Hallucination | **0** |

---

## Detail per Modul

### 1. Anti-Harassment
**Pertanyaan:** Sanksi pelaku pelecehan terbukti/tidak terbukti?  
**Jawaban:** ✅ Terbukti → di-PHK (PP 2025-2027 Pasal 61). Tidak terbukti → mutasi kerja.  
**Source:** `Anti-Harassment.md` (score 0.84)  
**Latency:** 6.7s

### 2. Basic Data Literacy
**Pertanyaan:** Kepanjangan DPD, PAR, NPL? Perbedaan Flowrate vs PAR?  
**Jawaban:** ✅ DPD=Days Past Due, PAR=Portfolio at Risk, NPL=Non-Performing Loan. Flowrate=mengukur perpindahan bucket DPD.  
**Source:** `Basic Data Literacy.md` (score 0.50) + `Business Process Amartha.md` (score 0.53)  
**Latency:** 7.8s

### 3. Build & Maintain Team
**Pertanyaan:** 3 kunci membangun tim solid? Power Distance vs otoriter?  
**Jawaban:** ✅ In-Group Collectivism, Power Distance (≠ otoriter — arahan konklusif + ruang diskusi), Human Oriented.  
**Source:** `Build & Maintain Team.md` (score 0.90)  
**Latency:** 8.0s

### 4. Client Protection
**Pertanyaan:** 8 prinsip Client Protection?  
**Jawaban:** ✅ 1) Appropriate Product Design, 2) Prevention of Over Indebtedness, 3) Transparency, 4) Responsible Pricing, 5) Fair & Respectful Treatment, 6) Privacy of Client Data, 7) Complaints Resolution, 8) Governance & HR.  
**Source:** `Client Protection.md` (score 0.94)  
**Latency:** 6.3s

### 5. Leadership in Crisis
**Pertanyaan:** Model 5R?  
**Jawaban:** ✅ R1=Respond (0-72 jam), R2=Report (investigasi), R3=Recover (pemulihan operasional), R4=Rebuild (pembangunan), R5=Reflect (post-mortem).  
**Source:** `Leadership in Crisis.md` (score 0.84)  
**Latency:** 13.2s

### 6. Welcome to Amartha
**Pertanyaan:** 4 perbedaan Grameen Model vs Bank Konvensional?  
**Jawaban:** ✅ Target pengguna, jaminan/aset, sifat peminjam, lokasi transaksi.  
**Source:** `Welcome to Amartha.md` (score 0.90)

### 7. Data Driven Strategy
**Pertanyaan:** Teknik 5 Whys dan contoh?  
**Jawaban:** ✅ Metode Sakichi Toyoda — contoh kasus penurunan RF Poket (Why 1-5 → akar masalah: manajemen cabang belum integrasikan KPI digitalisasi).  
**Source:** `Data Driven Strategy.md` (score 0.80)

### 8. Fraud Awareness
**Pertanyaan:** Jenis-jenis fraud dan prinsip non-negotiable?  
**Jawaban:** ✅ 12 kategori fraud operasional + 4 cyber fraud. Prinsip: tidak ada uang tunai dipegang FO, tidak ada "pinjam sementara".  
**Source:** `Fraud Awareness.md` (score 0.91)

### 9. Flexibel & Situasional Leadership
**Pertanyaan:** 4 gaya kepemimpinan? Formula feedback SBI?  
**Jawaban:** ✅ Directive (R1), Consultative (R2), Participative (R3), Delegative (R4). SBI = Situation-Behaviour-Impact.  
**Source:** `Flexibel dan Situasional Leadership.md` (score 0.93)

### 10. Framework Strategi dan Problem Solving
**Pertanyaan:** Siklus PDCA dan target SMART?  
**Jawaban:** ✅ Plan-Do-Check-Action. SMART: Specific, Measurable, Achievable, Relevant, Time Bound.  
**Source:** `Framework Strategi dan Problem Solving.md` (score 0.89)

### 11. Komunikasi Dasar
**Pertanyaan:** Teori 3V Albert Mehrabian?  
**Jawaban:** ✅ Visual 55%, Vocal 38%, Verbal 7%.  
**Source:** `Komunikasi Dasar.md` (score 0.84)

### 12. Basic Leadership
**Pertanyaan:** Perbedaan Manager vs Leader? Internal Locus of Control?  
**Jawaban:** ✅ Manager=kompleksitas/kontrol; Leader=visi/perubahan/motivasi. Internal LoC=tanggung jawab penuh, bukan cari kambing hitam.  
**Source:** `Basic Leadership.md` (score 0.77)

### 13. Sales Skill & Strategy
**Pertanyaan:** Sales Funnel dan tahapannya?  
**Jawaban:** ✅ Potensi → Prospek → Calon Mitra Aktif (KL/Onboarding) → Mitra Aktif → Mitra Berkembang.  
**Source:** `Sales Skill & Strategy.md` (score 0.90)

### 14. Strategi Pencapaian Target
**Pertanyaan:** Strategi disbursement mitra baru vs lanjutan?  
**Jawaban:** ✅ Baru: disiplin pipeline, titik sosialisasi strategis, program referensi. Lanjutan: filter eligibility, fast track, pendekatan persuasif.  
**Source:** `Strategi Pencapaian Target.md` (score 0.85)

### 15. Business Process Amartha
**Pertanyaan:** Tanggung Renteng? Prosedur mitra meninggal?  
**Jawaban:** ✅ Tanggung renteng = disiplin kredit kelompok. Mitra meninggal = asuransi jiwa lunasi sisa angsuran (syarat: usia 18-58, klaim via FO).  
**Source:** `Business Process Amartha.md` (score 0.69)

### 16. Produk dan Layanan Ekosistem AmarthaFin
**Pertanyaan:** Apa itu AmarthaLink? Peran FO?  
**Jawaban:** ✅ Jaringan agen keuangan digital (transfer, tarik tunai, PPOB). FO: perkenalkan, rekrut, edukasi, perluas jaringan.  
**Source:** `Produk dan Layanan Ekosistem AmarthaFin.md` (score 0.74)

---

## Kesimpulan

- **Tidak ada hallucination** — seluruh jawaban bersumber dari dokumen yang sesuai
- **Semua 16 modul** ter-cover dengan baik oleh RAG pipeline
- **Relevance score** rata-rata 0.83, menunjukkan retrieval bekerja optimal
- **Response cepat** (6–13s) untuk query satu-shot
- Sistem siap digunakan untuk production
