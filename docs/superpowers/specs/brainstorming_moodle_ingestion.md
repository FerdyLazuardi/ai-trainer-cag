# Brainstorming: Moodle Ingestion Architecture

Berdasarkan permintaan kamu, kita akan merancang arsitektur ingestion dari Moodle (Course: `Knowledge_Base`) ke Qdrant, lengkap dengan strategi resolusi `course_id`, best practice Markdown, dan evaluasi database.

---

## 1. Strategi Metadata Collection (Skala 10k Users & 100 Courses)

**Masalah:** Materi ada di dalam course terpusat (`Knowledge_Base`) yang dibagi per *section* (contoh: "Training Client Protection"). Namun, saat di-retrieve, kita butuh metadata `course_id` asli dari course "Client Protection" agar bot bisa memberikan konteks atau link spesifik ke course tersebut.

### Opsi Pendekatan Resolusi `course_id`:

**Pendekatan A: Manual Metadata via Frontmatter (PILIHAN USER)**
User/Content Author secara eksplisit menulis `course_id`, `department`, dan `topic` di dalam **Markdown Frontmatter** pada setiap file di Moodle. Script ingestion akan mempasing data ini langsung dari file tersebut.
- **Kelebihan:** 100% Akurat dan "tepat sasaran" sesuai keinginan user. Tidak ada resiko salah mapping.
- **Implementasi:** Field `course_id`, `course_name`, `department`, dan `topic` wajib ada di bagian paling atas dokumen (frontmatter).

### Dukungan File PDF (Convert dari PPT)


**Rekomendasi Metadata JSON di Qdrant (Payload):**
Untuk skala besar, kita butuh filter dimensi (HO/FO) dan relasi course. Payload ideal di Qdrant:
```json
{
  "department": "Global",            // Opsi: HO | FO | Global
  "topic": "Policy",               // Kategori dokumen
  "course_id": 45,                 // ID dari course target asli
  "course_name": "Client Protection", 
  "document_id": "uuid-dari-postgres" 
}
```

---

## 2. Klasifikasi Topik untuk Intent Gating

Agar AI Agent dapat melakukan routing jawaban dengan lebih akurat (Intent Gating), kita perlu menstandarisasi pilihan metadata `topic`. Berdasarkan ekosistem Amartha, berikut adalah usulan klasifikasi topiknya:

| Topic Key | Deskripsi Isi | Contoh Konten |
| :--- | :--- | :--- |
| `Compliance_Mandatory` | Kebijakan wajib & etika | Client Protection, Fraud Awareness, AML, Kode Etik, Anti-Harassment/Bullying. |
| `Product_Services` | Fitur & aturan produk Amartha | AmarthaFin, Poket, AmarthaEarn, AmarthaOne, Lending. |
| `Company_Culture` | Budaya & nilai perusahaan | Visi Misi, Amartha Level Up, Growth Mindset. |
| `Field_Operation_SOP` | Prosedur teknis untuk FO | Inisiasi, Grameen, Penagihan, Survey. |
| `HO_Operation_SOP` | Prosedur teknis untuk HO | PnC Admin, Finance Flow, Tech Standards. |
| `Professional_Dev` | Pengembangan skill | Leadership (LEAD), Mentorship, AI Training. |
| `Social_Impact` | Dampak sosial & pemberdayaan | AMBI, Sustainability Report, Women Empowerment. |
| `Onboarding` | Materi karyawan baru | Welcome Guide, Struktur Organisasi. |

**Manfaat Klasifikasi Ini:**
1. **Filtering Presisi:** AI bisa memprioritaskan "Compliance" jika user bertanya tentang regulasi.
2. **Intent Routing:** Jika user bertanya "Apa itu Poket?", Agent bisa langsung memfilter pencarian ke topic `Product_Services`.
3. **Analytics:** Kamu bisa melihat topik apa yang paling banyak ditanyakan oleh 10k users kamu.

---

## 3. Header-based Retrieval Strategy (Bukan Token-based)

Sesuai permintaan kamu, kita akan meninggalkan pendekatan "fixed token chunking" (misal: 512 tokens paksa) dan beralih ke **Semantic Header-based Splitting**.

### Bagaimana Cara Kerjanya?
Kita menggunakan `MarkdownNodeParser` dari LlamaIndex. Alih-alih memotong teks setiap x-token, parser ini akan membelah dokumen berdasarkan simbol `#`.

- **H1 (`#`)**: Menjadi Node Utama.
- **H2 (`##`)**: Menjadi Node Sub-Topik.
- **H3 (`###`)**: Menjadi Node Detail.

**Hasilnya di Vector DB:**
Satu "Point" di Qdrant akan berisi **satu bagian utuh** dari satu header ke header berikutnya. 
*Contoh:* Jika Section "Definisi" berisi 300 kata, maka 300 kata itu akan masuk sebagai 1 chunk utuh. Tidak akan terpotong di tengah kalimat.

### Keuntungan untuk Materi Training:
1. **Integritas Semantik:** Jawaban chatbot tidak akan "nanggung". Jika ia mengambil materi tentang "Prosedur Klaim", ia akan mengambil seluruh teks di bawah header tersebut.
2. **Konteks Headers:** LlamaIndex secara otomatis menyisipkan hierarki header ke dalam metadata. Jadi saat retrieve "Pengecualian", AI tahu bahwa ini adalah "Pengecualian" di bawah "Prosedur Klaim".

### Hal yang Perlu Diperhatikan (Success Criteria):
- **SOP Penulisan:** Content writer dilarang membuat satu section (`##`) yang terlalu panjang (misal 5000 kata tanpa sub-header). Jika section terlalu panjang, context window LLM akan membengkak. 
- **Rekomendasi:** Gunakan sub-header `###` jika penjelasan satu topik mulai melebihi 2-3 paragraf.

---

## 3. Best Practice Format Markdown untuk Ingestion

Karena kita sekarang menggunakan **LlamaIndex**, proses chunking sangat memperhatikan struktur semantik dokumen (Markdown headers). Jika formatnya berantakan, hasil RAG akan kehilangan konteks aslinya.

**Format Standar yang Wajib Diikuti (SOP untuk Content Writer):**

```markdown
---
department: "Global"
topic: "Policy"
course_id: 45
course_name: "Client Protection"
---

# Client Protection Policy 2026
*(Header 1: Judul Utama Dokumen, wajib ada satu di awal)*

Dokumen ini menjelaskan tentang...

## 1. Pengertian Dasar
*(Header 2: Topik Utama)*
Client protection adalah...

## 2. Prosedur Pendaftaran
*(Header 2: Topik Utama lainnya)*
Berikut adalah langkah-langkahnya:
1. Langkah pertama yang jelas.
2. Langkah kedua yang padat.

### 2.1 Pengecualian
*(Header 3: Sub-topik spesifik)*
Jika klien berada di luar negeri...
```

**Aturan Emas:**
1. **Gunakan Frontmatter (`---`):** Di baris paling atas untuk metadata. Script kita akan *parsing* ini sebelum mengirim teks ke LlamaIndex.
2. **Hierarki Header yang Rapih:** Gunakan `#` untuk judul paling atas, `##` untuk bagian utama, `###` untuk sub-bagian. LlamaIndex (atau text splitter semantik) akan menjaga teks dalam satu header agar tidak terpotong sembarangan.
3. **Hindari Paragraf Raksasa:** Pecah teks panjang menjadi *bullet points* atau paragraf pendek agar saat di-chunk ukuran 512 tokens, konteksnya tidak terbelah di tengah penjelasan krusial.

---

## 3. Evaluasi Database PostgreSQL

Saya telah mengecek [app/database/models.py](file:///d:/0Kuliah/1%20Amartha/ai-lms-agent/app/database/models.py). Berikut adalah evaluasinya:

**Status saat ini:** **SANGAT BAIK DAN AMAN.**
- **Relasi yang Kuat:** Model [Document](file:///d:/0Kuliah/1%20Amartha/ai-lms-agent/app/database/models.py#24-55) (Tabel utama) berelasi *one-to-many* secara `CASCADE` dengan model [Chunk](file:///d:/0Kuliah/1%20Amartha/ai-lms-agent/app/database/models.py#57-83). Jika dokumen diupdate/dihapus, chunk-nya ikut bersih.
- **Tipe Data Optimal:** Menggunakan `UUID` sebagai *primary key*, `JSON` untuk metadata yang fleksibel, dan `content_hash` untuk menghindari duplikasi saat ingestion ulang.
- **Observability (AgentLog):** Ada tabel [AgentLog](file:///d:/0Kuliah/1%20Amartha/ai-lms-agent/app/database/models.py#85-106) yang sangat bagus untuk memantau performa *chat*, *latency*, dan total tokenLLM. Sesuai dengan spesifikasi Langfuse/Ragas di arsitektur dokumenmu.

**Saran Peningkatan (Minor) untuk skala 10k user:**
1. **Access Control (Opsional):** Di masa depan, jika ada kebutuhan ketat dimana FO (*Front Office*) tidak boleh sama sekali melirik isi dari dokumen HO, kita bisa tambahkan *field* eksplisit seperti `allowed_departments: Mapped[list[str]]` di PostgreSQL, selain di Qdrant. Namun untuk saat ini, menggunakan payload filter `department` di Qdrant saat retrieval saja sudah cukup aman.

---

## Next Steps (Siklus Implementasi)
Jika kamu setuju dengan arsitektur metadata (Pendekatan A) dan format markdown di atas, langkah implementasi teknis berikutnya adalah:
1. Membuat script/endpoint Ingestion (menggunakan `moodle-api-user` skill) yang:
   - Mengambil isi course `Knowledge_Base` via Moodle REST API.
   - Mendukung format **Markdown** dan **PDF**.
   - Melakukan *parsing* Metadata (untuk MD via Frontmatter, untuk PDF via penamaan file atau metadata Moodle).
   - Menggunakan **Header-based Splitting** untuk Markdown (H1-H3) agar retrieval lebih semantik.
   - Menyimpan `course_id` manual sesuai input di metadata.
   - Mengirim semuanya ke fungsi LlamaIndex Ingestion yang barusan kita buat.
