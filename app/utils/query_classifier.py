"""Query Classification & Geo Helper Module for CAG LMS Agent.

Classifies field officer/branch queries into 11 consolidated domain categories
and provides geographic resolution helpers (e.g., Regional to Pulau).
"""
import re
from typing import Dict, Tuple

# Consolidated 11 Domain Categories Mapping
CATEGORY_MAPPING: Dict[str, str] = {
    # 1. KPI & Remunerasi
    "KPI & Remunerasi: Pengecekan Skor, Riwayat & Pencapaian KPI (Pribadi & Tim)": "KPI, Target & Remunerasi (Insentif / Bonus)",
    "KPI & Remunerasi: Kebijakan, Syarat & Jadwal Pencairan Insentif / Bonus": "KPI, Target & Remunerasi (Insentif / Bonus)",
    "KPI & Remunerasi: Parameter, Definisi, Bobot & Metrik Penilaian KPI (BP, BM, AM)": "KPI, Target & Remunerasi (Insentif / Bonus)",
    "KPI & Remunerasi: Strategi & Best Practices Pencapaian Target KPI Cabang": "KPI, Target & Remunerasi (Insentif / Bonus)",
    "KPI & Remunerasi: Monitoring Dashboard Performa, Ranking Area/Nasional & Superset": "KPI, Target & Remunerasi (Insentif / Bonus)",
    "KPI & Remunerasi: Pemahaman Definisi, Kriteria & Verifikasi Loan Paid": "KPI, Target & Remunerasi (Insentif / Bonus)",

    # 2. Portofolio & Collection
    "Portofolio & Collection: Strategi Manajemen Portofolio Cabang, Penurunan PAR/NPL & Action Plan": "Portofolio, Penagihan (Collection) & Recovery Cabang",
    "Portofolio & Collection: Teknik Penagihan Lapangan, Penanganan Mitra Mangkir / Kabur & Negosiasi": "Portofolio, Penagihan (Collection) & Recovery Cabang",
    "Portofolio & Collection: Terminologi & Indikator Portofolio Cabang (Flow Rate, DPD, WOB, RR, PnL)": "Portofolio, Penagihan (Collection) & Recovery Cabang",
    "Portofolio & Collection: Penegakan Sistem Tanggung Renteng & Mitigasi Konflik Majelis": "Portofolio, Penagihan (Collection) & Recovery Cabang",

    # 3. Produk & Proses Bisnis
    "Produk & Proses Bisnis: Siklus Pencairan, Limit Closed & Skema Mitra Lanjutan (ETB / Renewal)": "Produk Pembiayaan, Proses Bisnis & Akuisisi Lapangan",
    "Produk & Proses Bisnis: SOP Pelayanan Majelis, Sosialisasi Lapangan & Home Visit (HV)": "Produk Pembiayaan, Proses Bisnis & Akuisisi Lapangan",
    "Produk & Proses Bisnis: Akuisisi Mitra Baru (NTB), Kriteria Kelayakan & SOP Survey (UK / KL)": "Produk Pembiayaan, Proses Bisnis & Akuisisi Lapangan",
    "Produk & Proses Bisnis: Karakteristik & Perbandingan Produk: Modal Usaha vs Group Loan (GL)": "Produk Pembiayaan, Proses Bisnis & Akuisisi Lapangan",
    "Produk & Proses Bisnis: Ekosistem Digital Mitra: AmarthaFin, Poket & Layanan Celengan": "Produk Pembiayaan, Proses Bisnis & Akuisisi Lapangan",
    "Produk & Proses Bisnis: Prosedur & Perhitungan Pelunasan Dini Pembiayaan (Peldin)": "Produk Pembiayaan, Proses Bisnis & Akuisisi Lapangan",

    # 4. Kepemimpinan & Cabang
    "Kepemimpinan & Cabang: Ritme Kerja Cabang: Briefing Pagi, Perencanaan Harian & Evaluasi Operasional": "Kepemimpinan Cabang & Manajemen Tim (Leadership BM & AM)",
    "Kepemimpinan & Cabang: Model Kepemimpinan Situasional, Pemetaan Kompetensi R1-R4 & Coaching 1-on-1": "Kepemimpinan Cabang & Manajemen Tim (Leadership BM & AM)",
    "Kepemimpinan & Cabang: Manajemen & Pembinaan Tim Lapangan (Penanganan BP Mangkir & Penegakan Disiplin)": "Kepemimpinan Cabang & Manajemen Tim (Leadership BM & AM)",
    "Kepemimpinan & Cabang: Penataan Wilayah Operasional Cabang, Split Majelis & Revamping Teritori": "Kepemimpinan Cabang & Manajemen Tim (Leadership BM & AM)",

    # 5. Sistem & Dukungan Teknis
    "Sistem & Dukungan Teknis: Alur Eskalasi Insiden & Pelaporan Tiket Kendala (PST & Amartha Care)": "Sistem, Aplikasi Lapangan & Helpdesk (PST / Amartha Care)",
    "Sistem & Dukungan Teknis: Troubleshooting Aplikasi Lapangan BP & Kendala Input Transaksi/Setoran VA": "Sistem, Aplikasi Lapangan & Helpdesk (PST / Amartha Care)",
    "Sistem & Dukungan Teknis: Administrasi Berkas Onboarding Lapangan Cabang & Kas Kecil": "Sistem, Aplikasi Lapangan & Helpdesk (PST / Amartha Care)",

    # 6. LMS & Pelatihan
    "LMS & Pelatihan: Pengerjaan Kuis, Ujian Post-Test, Unduh Modul & Sertifikasi LMS": "LMS, Pelatihan & Materi Pembelajaran",
    "LMS & Pelatihan: Presensi, Absensi & Konfirmasi Kehadiran Sesi Pelatihan / Sosialisasi": "LMS, Pelatihan & Materi Pembelajaran",
    "LMS & Pelatihan: Kendala Teknis Pemutar Video / Multimedia Modul LMS": "LMS, Pelatihan & Materi Pembelajaran",

    # 7. HR & Kepegawaian
    "HR & Kepegawaian: Kompensasi, Tunjangan Operasional (Reimburse Bensin/Tol) & Benefit Karyawan": "HR, Karir, Benefit & Regulasi Kepegawaian",
    "HR & Kepegawaian: Regulasi Hubungan Industrial: Jam Kerja, Lembur Hari Sabtu, Cuti & Status Kemitraan": "HR, Karir, Benefit & Regulasi Kepegawaian",
    "HR & Kepegawaian: Jalur Karir Karyawan, Persyaratan Asesmen Promosi BP ke BM / BM ke AM": "HR, Karir, Benefit & Regulasi Kepegawaian",

    # 8. Kepatuhan & Etika
    "Kepatuhan & Etika: Mitigasi Fraud Lapangan, Perlindungan Nasabah (Client Protection) & Kode Etik": "Kepatuhan, Mitigasi Fraud & Perlindungan Nasabah (Client Protection)",

    # 9. Simulasi Lapangan
    "Simulasi Lapangan: Roleplay Interaktif Sales Pitching & Simulasi Penanganan Negosiasi Mitra": "Simulasi Interaktif & Roleplay Lapangan (Sales & Penagihan)",

    # 10. Budaya & Profil
    "Budaya Perusahaan: Internalisasi Nilai Budaya Perusahaan & Yel-Yel Semangat Kerja Amartha": "Budaya Perusahaan, Profil Organisasi & Personel Lapangan",
    "Profil Organisasi & Data: Pertanyaan Profil Amartha, Struktur Organisasi & Personel Lapangan": "Budaya Perusahaan, Profil Organisasi & Personel Lapangan",

    # 11. Interaksi & Percakapan
    "Interaksi & Percakapan: Konfirmasi, Umpan Balik & Tanggapan Singkat Pengguna": "Interaksi Percakapan, Sapaan & Obrolan Santai (Off-Scope)",
    "Interaksi & Percakapan: Sapaan Pembuka Interaksi (Greeting)": "Interaksi Percakapan, Sapaan & Obrolan Santai (Off-Scope)",
    "Interaksi & Percakapan: Obrolan Santai, Humor & Pertanyaan di Luar Konteks Kerja (Off-Scope)": "Interaksi Percakapan, Sapaan & Obrolan Santai (Off-Scope)",

    # Fallback
    "Operasional & Lapangan: Prosedur Kerja & Operasional Lapangan Umum": "Produk Pembiayaan, Proses Bisnis & Akuisisi Lapangan",
}


def classify_raw(q: str) -> Tuple[str, str, str]:
    """Classify a single raw query string into (category, subtopic, role)."""
    ql = (q or "").lower().strip()
    if not ql:
        return (
            "Interaksi & Percakapan",
            "Konfirmasi, Umpan Balik & Tanggapan Singkat Pengguna",
            "Semua Pengguna",
        )

    # 1. KPI & REMUNERASI
    # 1.1 Cek Skor, Riwayat & Status Pencapaian KPI
    if any(k in ql for k in [
        "cek kpi", "kpi aku", "kpi gw", "kpi saya", "kpi satya", "kpi kintamani", "mau ngecek kpi",
        "mau cek kpi", "chek kpi", "progrs kpi", "skor tim aku", "riwayat pencapaian",
        "berapa kpi saya", "kpi gw apa", "kpi gw brpa", "kpi gw pa ya", "kpi saya apa",
        "berapa hasil kpi saya", "tolong di jelaskan progrs kpi saya", "chek performa tim saya",
        "lihat nilai saya", "bagaimana cara saya bisa melihat nilai saya", "cara cek skor kpi",
        "cek kpi bulan juli", "ya coba chek juli aku brpa", "kalau dari 3 bulan terkahir?", "dpt brpa",
        "kalau saya totalnya berapa", "juni gimana", "malsudnya total isentif nya dari juli",
        "dari capaian kpi saya", "apa yang kurang dri kpi saya", "dengan melihat data pencapaian saya",
        "kpi bm kintamani", "kpi bp terbaru september", "kpi bp", "bm kintamani", "kalau bm karangasem",
        "klo bp gw gimana", "rata ratanya berapa", "emng targetnya berapa", "untuk kpi bulan ini oktober",
        "bagaimana dengan kpi satya", "nilainya gimana"
    ]):
        return (
            "KPI & Remunerasi",
            "Pengecekan Skor, Riwayat & Pencapaian KPI (Pribadi & Tim)",
            "BP & BM (Field Officers & Leaders)",
        )

    # 1.2 Kebijakan, Syarat & Jadwal Pencairan Insentif / Bonus
    if any(k in ql for k in [
        "insentif", "isentif", "bonus", "uangnya masuk", "kapan saya terima bonus",
        "estimasi bonus", "dapat sp 3, bonus", "sp 3, bonus atau insentif", "kapan uangnya masuk",
        "kapan kpi isentif cair", "kalo aku capai semua target bonus ku berapa",
        "berarti insentif ku tidak akan cair", "tapi untuk bp saya di point ada keterangan insentif",
        "pembagian insentifnya bagaimana", "mau lihat d area bontang mana saja yg dapat insentif",
        "hallo ai trainer saya ingin bertanya terkait insentif ketika capai target kpi",
        "terus untuk masuk ke gaji dalam isentif nya kapan", "kapan saya dapat isentif nya",
        "cairnya kapan", "itu cairnya kapan"
    ]):
        return (
            "KPI & Remunerasi",
            "Kebijakan, Syarat & Jadwal Pencairan Insentif / Bonus",
            "BP & BM (Field Officers & Leaders)",
        )

    # 1.3 Parameter, Bobot, Skema & Metrik Penilaian KPI
    if any(k in ql for k in [
        "parameter branch manager", "parameter business partner", "parameter area manager",
        "parameter kpi", "kpi itu apa", "kpi adalah", "kpi boom", "kpi q3", "range score kpi",
        "metode perhitungan score", "tahapan kpi", "alasan kita harus memahami kpi",
        "fungsi dasboard kpi", "acv expectation", "exceed expectation", "apa itu cohort",
        "apa saja parameter kpi bp", "apakah target kpi fo itu berdasarkan bcg mikronova",
        "pertanyaan 1 lagi apakah dibulan ini parameter kpi masih akan sama",
        "mari kita bahas soal kpi", "emang kpi sekarang berapa coba", "itu kpi atau rr",
        "kpi monitoring & support", "sosialisasi kpi fo 2026", "materi kpi", "soskpi fo",
        "ok klau kpi yang terupdate gmn?yang kpi am aja ya"
    ]) or ql in ["kpi", "kpi adalah", "terkait kpi"]:
        return (
            "KPI & Remunerasi",
            "Parameter, Definisi, Bobot & Metrik Penilaian KPI (BP, BM, AM)",
            "BP, BM & AM (Semua Jenjang)",
        )

    # 1.4 Strategi Pencapaian Target & Peningkatan Kinerja KPI
    if any(k in ql for k in [
        "cara mencapai kpi", "capai target", "kebiasaan apa yang harus diterapkan agar mencapai kpi",
        "bisa capai skor 100%", "tips dan strategi untuk capai kpi", "gimana cara achievenya",
        "bagaimana cara mengatur team agar bisa capai target", "cara biar area karangasem menyala",
        "bagaimana saya bisa mencapainya", "naikin angka disbursement", "target disburse point",
        "target bulanan", "dalam 1 hari sy harus acheve", "gimana cara achieve salesnya",
        "cara menghitung target bulanan", "bagaimana cara capai target yang baik",
        "coba kasih saya usulan dong, saya harus gimana biar bisa achieve", "mungkin ga ya sya achieve itu",
        "apa yang harus saya benahi", "tutor menjadi top 1 national kpi", "apa km bisa buatin schedule utk saya"
    ]):
        return (
            "KPI & Remunerasi",
            "Strategi & Best Practices Pencapaian Target KPI Cabang",
            "BM & BP",
        )

    # 1.5 Monitoring Dashboard KPI, Ranking & Superset
    if any(k in ql for k in [
        "rangking", "ranking", "top 1", "top 10", "dashboard kpi", "superset", "link kpi",
        "am dashborad", "dashboars untuk area manager", "dashboard intik am", "olimpiade fo",
        "ranking berapa saya di academy", "untuk rangking 1-5 siapa aja nasional",
        "rangking nasional bm september", "cara membaca dashbiad am",
        "kalo gabisa akses superset", "untuk detail mitra untuk rr apakah mitranya sama dengan superset",
        "padahal no 1 aku sekalimantan", "kalau sekalimantan", "se indonesia kah", "rank brpa saya",
        "boleh minta link kpi", "link perform kpi yang mana"
    ]):
        return (
            "KPI & Remunerasi",
            "Monitoring Dashboard Performa, Ranking Area/Nasional & Superset",
            "BM & AM",
        )

    # 1.6 Metrik Loan Paid
    if any(k in ql for k in [
        "loan paid", "loan pai"
    ]) or (("paid" in ql) and ("mitra" in ql or "dpd" in ql or "jumlah" in ql)):
        return (
            "KPI & Remunerasi",
            "Pemahaman Definisi, Kriteria & Verifikasi Loan Paid",
            "BP & BM",
        )

    # 2. PORTOFOLIO, COLLECTION & RECOVERY MANAGEMENT
    # 2.1 Indikator & Metrik Portofolio: Flow Rate, DPD, WOB, RR & PnL
    if any(k in ql for k in [
        "flow rate", "new flow", "flow adalah", "monitoring %flow rate", "menghitung flow rate",
        "menurunkan flow rate", "wob adalah", "wob 6", "apa itu wob", "dpd adalah", "dpd 1-30",
        "dpd 31-90", "dpd 90+", "amount collect", "rr apa", "rr point", "repayment rate",
        "kita coba bedah rr dulu", "wajib 98% itu mimal", "kalau aku punya mitra paid 267, agar rr 98%",
        "drawdown itu apa", "status drawdown", "istilah drawdown", "bcg itu apa", "jelaskan saja profit itu berapa",
        "profit dan lose poin", "poin profit", "profit itu berapa"
    ]):
        return (
            "Portofolio & Collection",
            "Terminologi & Indikator Portofolio Cabang (Flow Rate, DPD, WOB, RR, PnL)",
            "BM & AM",
        )

    # 2.2 Strategi Penurunan PAR / NPL & Manajemen Portofolio Cabang
    if any(k in ql for k in [
        "point batu mandi", "point tanggul", "portofolio point", "studi kasus lapangan",
        "perbaikan portofolio", "pembuatan strategi perbaikan portofolio", "kondisi %npl",
        "penyeimbangan gas dan rem", "menjaga kualitas portofolio", "tipsmenurunkan par",
        "strategi penurunan npl", "recovery portofolio & collection process", "recovery_strategy",
        "basic_portofolio_management", "langkah preventif wajib", "implementasi collecion",
        "menurut prinsip strategi pengelolaan portofolio", "dampak tidak menjaga kualitas portofolio",
        "evaluasi pasca penagihan", "strategi perbaikan bulan berikutnya", "point mojosongo",
        "tulis progress atau realisasi pencapaian target bulan pertama",
        "rencana aksi dan strategi collection dengan akar masalah",
        "smart goal", "apakah kamu bisa buatkan rencana aksi dan strategi collection",
        "eksekusi langkah awal stretegi", "sekarang gimana memulai eksekusi strategi itu",
        "evaluasi pertama terhadap langkap/strategi awal yang di ambil",
        "berarti lebih baik gak target drpd ngepar", "berarti lebih baik point diisi sedikit mitra, tp rr bagus",
        "menurutmu apa yg lebih penting,, kualitas atau kuantitas",
        "mitra aktif ato loan aktif ada 634 dpd 1-30 ada 32", "yg disb aku harus ngejar 50%",
        "ralat deh ppob uda 9,01 bro celengan 87,20", "ralat deh uda 61,66",
        "loan paid mitra dpd 0 minggu ini 96%", "untuk eligible ku kemungkinan yang betul-betul layak dibulan ini dibawah 100t",
        "led recovery", "gmna strategi untuk perbaikan portoflio", "halo saya mau sharing mengenai collection",
        "tulis strategi anda sebagai bm untuk perbaikan bulan depan"
    ]):
        return (
            "Portofolio & Collection",
            "Strategi Manajemen Portofolio Cabang, Penurunan PAR/NPL & Action Plan",
            "BM & AM",
        )

    # 2.3 Teknik Penagihan Lapangan, Negosiasi & Penanganan Mitra Menunggak
    if any(k in ql for k in [
        "aktivitas penagihan mitra par", "cara nagih", "cara menagih", "mitra kabur",
        "mitra mangkir", "gimana menghadapi nasabah nakal", "menghadapi masabah yg nakal",
        "mitra yang di ajak bicara dia malah kabur", "mitra yang ikut ikutan", "bad character",
        "jb adalah", "ptp", "btc", "persiapan penagihan", "teknis negosiasi",
        "daftar mitra tunggakan", "mitra tunggakan baru", "solusi banyak tugas rf",
        "jika memang mitra sudah kabur ke luar daerah", "jika mitra kabur ketika ditagih",
        "cara menarik data list mitra tunggakan baru", "bagaimana cara menagih mitra yang ikut ikutan",
        "ketika mitra jika terlambat membayar apakah pembayaran akan dilimpahkan ke minggu berikutnya",
        "saya ada kendala ketika melakukan penagihan ke ibu mitra dpd 7", "cara nagih par",
        "semisal ibu mitra menunggak lalu kita menyita barang", "apakah sita barang berlaku",
        "bagaiamna solusi menghadapi mitra npl yg kondisinya tdk baik, mangkir",
        "lalu mitra menungaknya bagaimana nanti", "kalau lembaga masyarakat cuek",
        "tiap pagi sebelum briefing pasti gue shorting mitra-mitra yang belum bayar",
        "gue tetap mengingatkan sesuai dengan apa sudah mereka janjikan",
        "gue sendiri mengingatkan kepada tim supaya jangan terlalu banyak memberikan jb atau ptp",
        "mereka wajib btc si", "follup rutin minimal h-1", "mitra wajib dikunjungi",
        "kadang-kadang ada kejadian-kejadian tak terduga yang mungkin dialami mitra"
    ]):
        return (
            "Portofolio & Collection",
            "Teknik Penagihan Lapangan, Penanganan Mitra Mangkir / Kabur & Negosiasi",
            "BP & BM (Dominan BP)",
        )

    # 2.4 Tanggung Renteng & Dinamika Konflik Majelis
    if any(k in ql for k in [
        "tanggung renteng", "menolak tanggung renteng", "hasut", "menghasut",
        "disiplin majelis", "bagaimana jika sistem tanggung renteng sulit dijalankan",
        "ibu-ibu mitra kelolaan saya di salah satu majelis menolak tanggung renteng",
        "gini, jadi majelis yang saya kelola ada satu ibu mitra yang mulai menghasut",
        "saya udah mastiin dan melihat buktinya dari isi chat", "saya mau melakukan home visit langsung ke ibu mitra bersama bp",
        "saya mau tahu kenapa ibu tersebut melakukan penghasutan", "jika kendalanya adalah terkait usaha",
        "jika memang ada niat sendiri ingin merusak disiplin majelis", "dampaknya terhadap majelis tentu akan membuat setiap anggota"
    ]):
        return (
            "Portofolio & Collection",
            "Penegakan Sistem Tanggung Renteng & Mitigasi Konflik Majelis",
            "BP & BM",
        )

    # 3. PRODUK PEMBIAYAAN & PROSES BISNIS LAPANGAN
    # 3.1 Produk Modal Usaha vs Group Loan (GL)
    if any(k in ql for k in [
        "gl adalah", "apa itu modal", "aapa itu modal", "perbedaan gl dan modal",
        "bedanya gl dan modal", "produk pembiayaan modal", "grup loan", "double loan",
        "bll gl", "bll", "mitra gl", "apa bedanya pembiayaan grup loan dengan modal",
        "double loan modal", "yang membedakan apa", "apa saja produk amartha",
        "berapa bunga pinjaman amartha", "berapa plafond mitra cycle", "berapa angsuran pinjaman 9jt"
    ]):
        return (
            "Produk & Proses Bisnis",
            "Karakteristik & Perbandingan Produk: Modal Usaha vs Group Loan (GL)",
            "BP & BM",
        )

    # 3.2 Prosedur Survey / Uji Kelayakan (UK), Komite Lapang (KL) & Kriteria Mitra Baru (NTB)
    if any(k in ql for k in [
        "syarat uk new", "cara uk", "dokumen saat uk", "penggantian penanggung jawab",
        "cara mengganti penanggung jawab", "batas maksimal penanggung jawab", "umur",
        "syarat usia", "dsr", "debt service ratio", "plafond", "kl / onboarding",
        "komite lapang", "komite lapangan", "apa syarat menjadi mitra", "validasi uk",
        "urutan dalam uk", "credit score", "xcore", "terflaging high quality",
        "apa itu ntb", "ciri ciri mitra new to book", "apa itu komite lapangan",
        "bagaimana sop saat uk", "berapa syarat usia menjadi mitra", "berapa persen dsr layak",
        "rumus debt service ratio", "apa rumus dsr", "dokumen yg di butuhkan mitra terflaging high quality",
        "jika validasi uk dilakukan oleh bm apakah ada", "apakah ada proses validasi uk di proses modal",
        "tahap 3: komite lapang (kl) / onboarding", "bagaimana kalau mitra pengajuan tetapi di dalam kelompoknya ada mitra macet dan par",
        "ntb adalah", "klo ntb beda lag?", "oiya aturan penentuan target ntb per bp dari mana"
    ]) or ql == "ntb":
        return (
            "Produk & Proses Bisnis",
            "Akuisisi Mitra Baru (NTB), Kriteria Kelayakan & SOP Survey (UK / KL)",
            "BP & BM (Dominan BP)",
        )

    # 3.3 Pencairan, Limit Closed & Penanganan Mitra Lanjutan (ETB / Renewal)
    if any(k in ql for k in [
        "mitra lanjutan", "limit", "closed", "cycle 0 rejec", "mitra baru cycle 0 tertolak",
        "pengajuan mitra baru tertolak", "renewal", "ranwel", "rineweel", "winback do",
        "reaktivasi", "disburs", "disbursement", "akselerasi pencairan", "pencairan modal",
        "cara pencairan", "alur pencairan", "proses pencairan pinjaman", "program ranwel",
        "notifikasi (maaf belum bisa pencairan lagi)", "notifikasi maaf", "etb", "poi",
        "etb apps script", "kategori tanggapan pada etb", "progress dari data leads etb",
        "leads etb", "panduan sosialisasi per program etb", "2 pilar akselerasi pencairan",
        "bagaimana alur pencairan modal digital loan", "maksud saya proses cair agar semua mitra dapat lender",
        "gimana supaya proses diamartha bisa cepet", "product modal kan gak pakai uk., selalu closed",
        "saya selalu pusing dengan pengajuan modal yang tiba tiba closed",
        "untuk di point kapuas murung sekarang itu dishburse juga susah",
        "parameter apa aja yang menentukan mitra cycle 0 dapat menerima pencairan",
        "tahapan apa saja proses cycle 0 sampai pencairan", "hal apa aja yang menyebabkan pengajuan",
        "di pengajuan mitra lanjutan selalu ada notifikasi maaf",
        "mitra lanjutan dapat limit bisa di kil cair, tapi tiba2 closed",
        "aku masih selalu bingung ketika cycle 0 rejec", "baru pencairan",
        "kamu tahu ada berapa mitra eligible ku bulan ini?", "ku mulai dari etb?",
        "jadi bisa di tawarkan di mnggu ke brapa sebelm jatuh tempo", "pengajuan modal gimna caranya",
        "berapa minggu sebelum jatuh tempo mitra dengan pinjaman aktif mulai dibtawarkan program ranwel"
    ]):
        return (
            "Produk & Proses Bisnis",
            "Siklus Pencairan, Limit Closed & Skema Mitra Lanjutan (ETB / Renewal)",
            "BP & BM",
        )

    # 3.4 Pelunasan Dini Pembiayaan (Peldin)
    if any(k in ql for k in [
        "peldin", "pelunasan dini", "pemotongan peldin", "rumus pmotongan peldin"
    ]):
        return (
            "Produk & Proses Bisnis",
            "Prosedur & Perhitungan Pelunasan Dini Pembiayaan (Peldin)",
            "BP & BM",
        )

    # 3.5 Ekosistem Digital AmarthaFin, Poket & Celengan
    if any(k in ql for k in [
        "amarthafin", "afin", "poket", "celengan", "ppob", "cakap digital", "mitra pnc modal",
        "poket anda dibekukkan", "produk dan layanan ekosistem amarthafin", "bukan fitur amartha fin",
        "berapa limit amarthafin premium", "registrasi untuk mitra lama", "mitra lama tidak mau instal poket",
        "contoh penerapan 5 whys pada kasus penurunan rf poket", "halo ka bisa minta piket sejuta",
        "maksudnya poket sejuta", "celengan itu apa",
        "mitra modal mau pelunassan dan akan diajukan pencairan kembali akan tetapi akun mitra terdapak tulisan \"poket anda dibekukkan\"",
        "karakteristik calon agen"
    ]):
        return (
            "Produk & Proses Bisnis",
            "Ekosistem Digital Mitra: AmarthaFin, Poket & Layanan Celengan",
            "BP & BM",
        )

    # 3.6 Standar Operasional Prosedur (SOP) Pelayanan & Kumpulan Majelis
    if any(k in ql for k in [
        "business process amartha", "proses bisnis amartha", "indikator keberhasilan proses bisnis",
        "siklus akuisisi bp", "pasti adalah", "langkah yang benar menurut pasti",
        "pelayanan dan kumpulan majelis", "cara pelayanan", "bagaimana caranya pelayanan",
        "hv adalah", "monitoring hv", "home visit", "topsheet", "ikrar petugas", "ikrar bp",
        "ikrar business phartner", "dimana sop", "sop", "cara sos", "sosialisasi yang benar",
        "tahap 2: sosialisasi produk", "yang ku maksud masih layak kah kami melakukan sos",
        "gimana ya caranya ningkatin capaian ntb di point saya ya", "berapa target home visti",
        "berapa target leads", "form calont mitra baru mana emngya", "akhir-akhir saya ngerasa kesulitan untuk ngelakuin sosialisasi",
        "aku punya bm tapi minim sos poi kenapa ya", "aku sudah sos poi tapi belum minat itu kenapa ya",
        "kalo dilihat dari laporan bp sih kebanyakan di tahap sosialiasi", "kalau dari cerita bp memang di beberapa wilayah",
        "kalo saya lihat sih beberapa bp masih ada yg langsung menjual produk amartha",
        "berarti beberapa bp sepertinya kurang untuk melakukan tahapan eksplor dan atasi ya",
        "bagaimana cara meningkatkan jumlah mitra baru sedangkan wilayah sudah full adanya amartha",
        "adalah menjadi tanggung jawab kami dalam mengemban amanah dari amartha",
        "berapa jumlah mitra dalam 1 majelis", "cara bisa dptin ketua majelis", "sosialisasi",
        "proses pencarian pinjaman kepada mitra adalah definisi dari", "sales dan disbusment",
        "kalo terkait capaian leadsnya gimana", "halo, coba jelasin tentang sales"
    ]):
        return (
            "Produk & Proses Bisnis",
            "SOP Pelayanan Majelis, Sosialisasi Lapangan & Home Visit (HV)",
            "BP & BM (Dominan BP)",
        )

    # 4. OPERASIONAL LAPANGAN & DUKUNGAN TEKNIS SISTEM (PST & APLIKASI)
    # 4.1 Eskalasi Kendala Sistem via PST & Amartha Care
    if any(k in ql for k in [
        "pst", "amartha care", "gform", "salah input", "ubah no rekening",
        "bantu ganti nomer afin", "perbandingan amartha care vs pst", "cara masuke gambar",
        "gabisa tanya disertain gambar", "halo aku mau tanya dong kalo ada kendala terkait modal cycle 0 tanya ke siapa",
        "adakah tim pst yang bisa aku hubungi", "gimana cara aku tau progressnya",
        "mana? di bawah form pelaporan", "bisa ga kamu langsung gabungin gform", "di pst ada bisa lapor apa lagi",
        "berapa lama sih laporan aku dikerjain", "wah kacau juga ya masa ga ada sla nya",
        "kamu tau gak pic masing-masing regional", "kaya mitra yang aku pegang itu pas uk salah masukin nik",
        "nah kalo ini mitranya yang salah input", "mitra ku juga ada yang pengen ubah no rekening",
        "karena kamu mention ktp", "kalo aku mau cuti isi gform pst juga", "pst nya lama",
        "baik coba saya laporkan ke pst"
    ]):
        return (
            "Sistem & Dukungan Teknis",
            "Alur Eskalasi Insiden & Pelaporan Tiket Kendala (PST & Amartha Care)",
            "BP & BM",
        )

    # 4.2 Troubleshooting Aplikasi BP, Transaksi & Setoran VA
    if any(k in ql for k in [
        "save va", "batas maksimal save va", "tugas pelayanan ga ada", "tugas hv ga muncul",
        "repayment tidak bisa diinput", "50/50", "aplikasi bp", "sms already exist",
        "gabisa setoran", "kendala repayment", "caranya tau kalo kendala yang saya alami iu incident",
        "pas survey ada notif sms already exist", "kalau bm gabisa uk val",
        "kalau tugas di aplikasi bp ga muncul itu kemana ya", "jadi mitranya belum lunas,tapi di bp apps mitranya sudah 50/50",
        "kamu bisa bantu kendala repayment enggak", "mitra gl gd slik knp masih muncul di tugas bp",
        "mengapa mitra gl yang sudah di undang ke majelis digital tidak bisa di proses lagi ke produk gl"
    ]):
        return (
            "Sistem & Dukungan Teknis",
            "Troubleshooting Aplikasi Lapangan BP & Kendala Input Transaksi/Setoran VA",
            "BP (Field Officer)",
        )

    # 4.3 Administrasi Berkas Onboarding Cabang & Kas Kecil
    if any(k in ql for k in [
        "upload unbroading", "upload file di onboardingnya", "checklist onboarding",
        "bagaimana cara mengupload checklist onboarding", "kenapa tidak bisa upload unbroading",
        "kok aku gbisa upload file di onboardingnya ya", "kas kecil"
    ]):
        return (
            "Sistem & Dukungan Teknis",
            "Administrasi Berkas Onboarding Lapangan Cabang & Kas Kecil",
            "BM & BP",
        )

    # 5. KEPEMIMPINAN & MANAJEMEN CABANG (BM & AM)
    # 5.1 Ritme Kerja Cabang: Briefing Pagi, Perencanaan & Evaluasi
    if any(k in ql for k in [
        "briefing pagi", "morning brifing", "panduan briefing pagi", "dimana untuk briefing",
        "evaluasi pasca penagihan", "ritme", "tugas harian bp - bm", "breakdown tugas",
        "step step bp", "sesi diskusi solusi", "breakdown tugas yang akan di lakukan oleh bm",
        "breakdown tugas yang akan di delegasi kan ke bp", "tujuan utama : evaluasi pertama",
        "tujukan kepada point batu mandi", "bagaimana cara membuat bm aware dengan tugas hariannya",
        "memastikan bm punya ritme atau plan pembahasan di setiap briefing dan evaluasi",
        "karena pada saat briefing dan evaluasi, pembahasan yang lebih ditekankan",
        "2 kali, saat briefing dan saat evaluasi", "menentukan skala prioritas", "prioritas",
        "bm tidak konsisten melakukan monitoring", "monitoring", "spotcheck",
        "biar tim lebih efektif tanpa di ingatkan setiap hari", "materi briefing pagi",
        "evaluasi", "feedback dari tim (bp)"
    ]):
        return (
            "Kepemimpinan & Cabang",
            "Ritme Kerja Cabang: Briefing Pagi, Perencanaan Harian & Evaluasi Operasional",
            "BM (Branch Manager)",
        )

    # 5.2 Pembinaan Tim, Penanganan BP Mangkir, Karakter & Senioritas
    if any(k in ql for k in [
        "bp senior", "mangkir", "boom", "bad karakter", "beban", "slow respon",
        "punishment", "tim lebih efektif", "pimpin bp", "mengatur team", "bp mangkir",
        "bp yang jadi beban", "bp minim pembayran paid", "bp gw mangkir semua",
        "mengatasi bp senior yang kerjanya tidak mau di atur", "bagaimana cara mengeksekusi bp yang tidak produktif",
        "apa yang harus dilakukan untuk tim yang slow respon group wa", "bagaimana cara pimpin bp yang bad karakter",
        "gimana cara ngasih punishment kepada bm yg gak target", "bp yang jadi beban harusnya di apakan",
        "kalau bp gw mangkir semua gimana", "mereka berdua udah hampir setahun mungkin disini",
        "saya sudah ber email kepada rpp untuk bp mangkir tersebut",
        "saya memiliki bp yg di mangkirkan karena berperilaku tidak baik",
        "semuanya sekarang selalu bawa np dan satu lagi mereka gatau soal status bm ku",
        "ralat bm lama sejak tayu berdiri", "masalahnya bp susah buat report tepat waktu",
        "mereka ga paham sama kompetensinya teamnya masing2", "gimana caranya kita sebagai bm agar omongan kita itu selalu berdampak",
        "saya udah coba untuk bikin beberapa kegiatan bersama", "gimana ya caranya untuk ngebuat tim jadi lebih kompak",
        "buatkan 5 pertanyaan untuk bp seputar amartha"
    ]):
        return (
            "Kepemimpinan & Cabang",
            "Manajemen & Pembinaan Tim Lapangan (Penanganan BP Mangkir & Penegakan Disiplin)",
            "BM & AM",
        )

    # 5.3 Kepemimpinan Situasional, Pemetaan R1-R4 & Supervisi
    if any(k in ql for k in [
        "1-on-1", "r1-r4", "pemetaan", "flexible leadership", "situasional leadership",
        "basic leadership", "metode clear", "kompetensi", "tahapan plan", "tahap check",
        "tahap check: monitoring vs controlling", "praktik lapangan 3v", "trust equation",
        "5 whys", "eisenhower matrix", "peran kepemimpinan leaders", "susun format observasi",
        "bantu susun poin-poin untuk sesi 1-on-1", "bantu susum pemetaan r1-r4",
        "jelaskan tentang basic leadership", "apa itu metode clear", "jelaskan tentang tahapan plan",
        "jelaskan tentang 4 elemen kunci kepercayaan", "jelaskan tentang ringkasan metode pengembangan",
        "jelaskan tentang tahapan check", "jelaskan tentang eisenhower matrix",
        "jelaskan tentang peran kepemimpinan leaders & regulasi kepegawaian", "4 tahap ini sma kyk pdca",
        "gimana cara bangun kesadaran di level am", "menghadapi hubungan dalam satu kantor bagaimana",
        "oh berarti saya harus buat kegiatan yang lebih personal dan meningkatkan kedekatan emosional"
    ]):
        return (
            "Kepemimpinan & Cabang",
            "Model Kepemimpinan Situasional, Pemetaan Kompetensi R1-R4 & Coaching 1-on-1",
            "BM & AM",
        )

    # 5.4 Penataan Wilayah, Majelis & Ekspansi (Revamping, Remapping & Split)
    if any(k in ql for k in [
        "split majelis", "revamping majelis", "remapping", "jarak rumah", "jumlah kecamatan",
        "maksimal mitra di 1 kecamatan", "1 kecamatan berapa mitra", "memetakan daerah",
        "analisa potensi wilayah", "jumlah anggota dalam 1 kelompok", "tahap 1: analisa potensi wilayah",
        "berapa max jumlah anggota dalam 1 kelompok", "apa saja yang perlu diperhatikan saat revamping majelis",
        "mekanismenya", "jumlah kecamatan yang bisa dibuka pada tahun pertama", "jumlah kecamatan yang bisa dibuka",
        "hm kalo split gitu tuh harus modal sama modal", "oo gitu, kalo mau split majelis gimana caranya",
        "hmm kamu tau ga split majelis tuh apa", "bantu saya memetakan daerah yang bisa untuk dibuka majelis baru",
        "berapa jarak rumah mitra dengan tempat kumpulan"
    ]):
        return (
            "Kepemimpinan & Cabang",
            "Penataan Wilayah Operasional Cabang, Split Majelis & Revamping Teritori",
            "BM & AM",
        )

    # 6. KEPATUHAN, FRAUD AWARENESS & CLIENT PROTECTION
    if any(k in ql for k in [
        "fraud", "tanda awal fraud", "faktor penyebab fraud", "fraud triangle",
        "dokumen wajib laporan fraud", "cyber fraud", "mitra terdampak froud",
        "mitra atas nama", "fraud joki", "bp fraud", "client protection",
        "collection_ethics", "dampak pelecehan", "relasi kuasa & consent",
        "kode etik 23", "dimana kita dapat melaporkan tentang fraude",
        "jelaskan tentang tanda awal fraud lapangan", "jelaskan tentang 3 faktor penyebab fraud",
        "strategi negosiasi mitra terdampak froud", "jelaskan tentang jenis serangan cyber fraud",
        "bagaimana supaya bp fraud mau membayar kerugian", "bagaimana cara mengatasi mitra atas nama",
        "jelaskan secara mendalam semua prosedur penanganan kredit bermasalah", "jelaskan secara singkat apa itu client protection",
        "sedangkan mitra banyak yang sudah tidak percaya lagi sama amartha, dikarenakan banyaknya mitra yang menjadi korban fraud"
    ]):
        return (
            "Kepatuhan & Etika",
            "Mitigasi Fraud Lapangan, Perlindungan Nasabah (Client Protection) & Kode Etik",
            "BP, BM & AM (Semua Jenjang)",
        )

    # 7. HR, KARIR & REGULASI KEPEGAWAIAN
    # 7.1 Jalur Karir & Promosi Internal (Assesment BM / AM)
    if any(k in ql for k in [
        "cara agar aku menjadi bm", "assesment to bm", "bmdp", "ojt bm", "promosi ke am",
        "pjs am", "ingin belajar memahami pekerjaan menjadi seorang bm", "jobdesk bm",
        "apa saja syarat agar bp bisa diajukan assesment to bm", "apa itu bmdp",
        "apa jobdesk bm secara singkat", "iya karna kebetulan aku mau ojt bm jadi butuh pembekalan",
        "berpa lama masa pjs am di amartha, dan kapan seorang bm yg promosi", "kapan bm promosi jadi am akan menerima gaji am"
    ]):
        return (
            "HR & Kepegawaian",
            "Jalur Karir Karyawan, Persyaratan Asesmen Promosi BP ke BM / BM ke AM",
            "BP & BM",
        )

    # 7.2 Kompensasi, Tunjangan & Fasilitas Benefit Karyawan
    if any(k in ql for k in [
        "gaji", "payroll", "benefit", "fleksibel benefit", "reimburse", "perjalanan dinas",
        "travel alowence", "melahirkan", "kesehatan gigi", "insentif makan malam",
        "gimana caranya gaji meningkat", "aku mau tau link semua benifit di amartha",
        "aku mau tau benefit untuk bm yang melahirkan", "hi ai aku mau tanya apakah di amartha ini banyak benifit nya",
        "pengajuan perjalanan dinas", "pengajuan reimburse", "pengajuan travel alowence",
        "cara pengajuan reimburse bensin dan tol", "berapa jumlah payroll/gaji bm",
        "tunjangan kesehatan gigi mana sk nya", "carikan tentang benedit dong",
        "bagaimana cara melakukan pengajuan fleksibel benefit", "oke, kalo aku mau tau no bpsjtk aku berapa sih"
    ]):
        return (
            "HR & Kepegawaian",
            "Kompensasi, Tunjangan Operasional (Reimburse Bensin/Tol) & Benefit Karyawan",
            "BP & BM (Semua Karyawan)",
        )

    # 7.3 Kebijakan Jam Kerja, Lembur, Hak Cuti & Peraturan Perusahaan
    if any(k in ql for k in [
        "lembur", "sabtu", "kontrak", "peraturan perusahaan", "sp 3", "cuti", "jadwal kerja",
        "kemitraan", "kenapa bp masih menjadi kemitraan", "apakah bp boleh cuti? berapa lama",
        "perarturan perusahaan pasal 48 ayat 3 huruf f", "mengenai surat edaran sama sekali tidak ada",
        "saya sama persis mengalami hal tersebut. diminta untuk lembur sampai sabtu",
        "untuk jam kerja resmi senin - jumat", "saya mau tanya, untuk jadwal kerja itu mulai dari hari senin",
        "kenapa sekarang setiap sabtu karyawan amartha masuk tidak sesuai dengan kontrak",
        "aku sebenar nya udh mau nyerah sih kerja di amartha", "semangatin dong aku mau mangkir",
        "peraturan", "life at amartha"
    ]):
        return (
            "HR & Kepegawaian",
            "Regulasi Hubungan Industrial: Jam Kerja, Lembur Hari Sabtu, Cuti & Status Kemitraan",
            "BP & BM (Semua Karyawan)",
        )

    # 8. LMS, TRAINING & MATERI PEMBELAJARAN
    # 8.1 Presensi & Absensi Sesi Pelatihan
    if any(k in ql for k in [
        "absensi pagi", "absen", "hadir", "presensi kehadiran", "persentasi kehadiran",
        "mau absen di sosilasi", "sosialisasi pelaksanaan tugas harian bp - bm", "sosialisasi rm",
        "kenapa ga bisa presensi kehadiran di refreshment fraud awareness",
        "kenapa gabisa persentasi kehadiran"
    ]):
        return (
            "LMS & Pelatihan",
            "Presensi, Absensi & Konfirmasi Kehadiran Sesi Pelatihan / Sosialisasi",
            "BP & BM (Peserta Training)",
        )

    # 8.2 Aktivitas Kelas LMS: Ujian, Kuis, Sertifikasi & Akses Modul
    if any(k in ql for k in [
        "post test", "pre-test", "kuis", "soal pre-test", "sertifikat", "tidak bisa masuk untuk melihat sertifikat",
        "ga bisa aku ikut kuis", "amarthapedia", "download sop", "materi tadi", "materi sales",
        "materi training", "kelas hari ini", "training of trainer", "training recovery",
        "new onboarding leaders", "contoh soal", "risk & portofolio", "people & leadership",
        "operation & compliance", "sales & growth", "empower growth", "amartha pedia",
        "bagaiman cara akses amarthapedia", "post test nya yang mana ini kak", "materi tadi mana ya",
        "materi training off trainer", "kamu tidak bisa ikuti kelas ini katanya", "soal pre-testnya dimana ya",
        "training of traine recovery", "bisa kirim file nya", "download sop lengkap terbaru dimna",
        "materi new onboarding sesi 2", "kmi lagi treaning", "kok tidak ada di kelas saya",
        "nama materi nya apa", "link nya", "cara belajar dan merangakup kerjaan agar mudah di pahamii",
        "materi amartha pedia", "bantu aku mencari materi sales & growth", "apa yg bisa sy pelajari hari ini",
        "tes psikotes talentica", "sales skill  strategy", "sales skil strategy",
        "jelaskan tentang esensi komunikasi field officer (fo)", "jelaskan tentang glosarium",
        "jelaskan tentang langkah 1: teknik membaca data & mendapatkan insight",
        "jelaskan tentang langkah penanganan kasus", "training of totainer recovery"
    ]):
        return (
            "LMS & Pelatihan",
            "Pengerjaan Kuis, Ujian Post-Test, Unduh Modul & Sertifikasi LMS",
            "BP & BM (Peserta Training)",
        )

    # 8.3 Kendala Pemutaran Video & Media Pembelajaran LMS
    if any(k in ql for k in [
        "video", "videonya", "vd nya", "nonton", "kurang paham kalo nonton", "kepotong",
        "kenapa vd nya tidak bisa di putar", "kendala kalau di buka di hp videonya kepotong",
        "saya kurang paham   kalo nonton", "saya lebih masuk belajar nya dari membaca daripada melihat video",
        "kok kepotong"
    ]):
        return (
            "LMS & Pelatihan",
            "Kendala Teknis Pemutar Video / Multimedia Modul LMS",
            "BP & BM (Peserta Training)",
        )

    # 9. BUDAYA PERUSAHAAN & YEL-YEL AMARTHA
    if any(k in ql for k in [
        "yel yel", "berdaya untuk tumbuh", "berjaya untuk tumbuh", "berjaya untuk maju", "bukannya skrg berjaya untuk maju",
        "culture", "budaya", "roket melambangkan culture apa", "apa budaya kerja baru amartha",
        "amartha culture", "amartha new culture mana kelas nya", "yel yel martha yang grow ada video nya",
        "yel yel yg dilakukan sebelum brifing pagi"
    ]):
        return (
            "Budaya Perusahaan",
            "Internalisasi Nilai Budaya Perusahaan & Yel-Yel Semangat Kerja Amartha",
            "BP & BM (Semua Karyawan)",
        )

    # 10. SIMULASI LAPANGAN & ROLEPLAY INTERAKTIF
    if any(k in ql for k in [
        "latihan sales", "berperan sebagai", "calon ibu mitra", "latihan penagihan",
        "ai trainer bantu aku untuk latihan", "bu siti", "faisal dari amartha",
        "warung", "sawit", "butuh modal rp10.000.000", "minjem ke rentenir",
        "bisa pinjam sendiri", "harus ada kelompok ya", "tolong berperan sebagai calon ibu mitra",
        "saya mau latihan sales", "kebetulan nih saya butuh modal untuk nambah stok usaha",
        "saya butuh modal rp10.000.000 nih", "waduh,, tetangga saya disini sudah punya pinjaman",
        "sudah saya bilang tadi,, riwayat pembayaran mereka buruk", "saya tidak ada tunggakan dipinjaman lain",
        "sudah lah, ribet itu. mending saya cari aja pinjaman lain", "halo ibu sri, apakah ibu benari mitra amartha",
        "kalo ditolak gimana", "gimana caranya soalnya mitra suka langsung nolak", "kalo saya ikutin ini apa pasti berhasil",
        "bisa sih, nanti saya coba infokan dulu. tapi kalau dilihat rata2 mereka bukan penduduk asli",
        "mereka usaha memang disekitar lokasi transaksi dan kebun. tapi mereka tinggal di daerah lain",
        "untuk pinjaman usaha bisa berapa mas ?", "iya bu, sebelumnya saya boleh tau nama ibu?",
        "ooo... dari amartha ya", "waalaikumsalam..", "saya udah coba tawarkan, namun dia tetap menolak",
        "alasannya ya itu tadi, dia merasa ada kebutuhan yang lebih penting dari membayar angsuran.",
        "dia merasa usahanya lagi sepi dan ada kebutuhan lain yang lebih penting",
        "kalau kata ibunya sih penghasilan usahanya saat ini hanya cukup untuk kebutuhan sehari-hari",
        "karena uang untuk bayar angsurannya tidak disisihkan sedikit demi sedikit ?",
        "saya harus tetap menganjurkan si ibu untuk bayar sebagian tapi dengan jumlah yang kecil"
    ]):
        return (
            "Simulasi Lapangan",
            "Roleplay Interaktif Sales Pitching & Simulasi Penanganan Negosiasi Mitra",
            "BP (Field Officer)",
        )

    # 11. PROFIL ORGANISASI, DATA CABANG & UMUM AMARTHA
    if any(k in ql for k in [
        "kapan amartha berdiri", "pendiri amartha", "ceo amartha", "ada berapa area di kalimantan barat",
        "ada berapa jumlah karyawan amartha", "siapa jokowi", "siapa prabowo", "5+5", "5+5 berapa",
        "siapakah nama area manager lampung barat", "dominggus ly bm point karangasem",
        "bp gw siapa ja", "bp gw siapa aja", "bp aku siapa aja", "siapa aja bp gw", "gw di point mana",
        "menurut lu si paulus gimana", "menurut anda bp paulus gimana", "point kintamani", "ahmad", "wahyu hidayat",
        "mala ainun rohmah", "soal kerjaan di amartha", "bisakah kamu menjelaskan rangkuman tentang amartha"
    ]):
        return (
            "Profil Organisasi & Data",
            "Pertanyaan Profil Amartha, Struktur Organisasi & Personel Lapangan",
            "BP & BM",
        )

    # 12. SAPAAN & INTERAKSI PERCAKAPAN UMUM
    # 12.1 Sapaan Formal & Kasual
    if ql in [
        "halo", "hallo", "pagi", "selamat malam", "hi", "sore", "malam", "terima kasih",
        "thanks", "makasih sayang", "iya selamat pagi", "halo bro", "hallow", "hai",
        "haii", "p", "halo pak", "hlo", "halo team", "haloo beb", "iya selamat pagi juga",
        "selamat malam", "hallo,."
    ]:
        return (
            "Interaksi & Percakapan",
            "Sapaan Pembuka Interaksi (Greeting)",
            "BP & BM (Semua Pengguna)",
        )

    # 12.2 Respon Singkat & Konfirmasi Dialog
    if ql in [
        "ok", "oke", "oke baik", "siap, makasih atas informasinya", "baik lah", "baik lah,",
        "tidak perlu lah saya sudah ikhlas memang bajingan orang tersebut mengakibatkan kami seperti ini.",
        "boleh", "sudah selesai kak", "iya", "tidak", "ya", "oke ada yang lain yang kurang ga",
        "mau", "mau dong", "sudah di coba dan masih sama", "thanks bro",
        "ko kamu malah jadi nanya aku sih", "kok balik nanya sih", "lo kok nanya saya",
        "wkwkwk", "salah kamu", "pembohong, jelas 130", "tidak ada", "aku bingung",
        "gamua,kamu aja", "👍", "iya, bye", "iya, hati-hati dijalan ya 😊", "aamiin..",
        "apa?", "boleh aku berkeluh kesah?", "aku", "aku gak ngerti apa aja",
        "baik terimakasih", "ada lagi", "gatau kemana", "6 bulan bilangnya",
        "saya mah buka sosialisasi", "data kamu diperoleh dari mana?", "bp", "lead",
        "owh oke lah", "kok kamu tau saya mt?", "saya",
        "appscript", "materi tentang appscript", "2 2 nya boleh?", "buatkan secara rinci saja",
        "bikin kan sesingkat mungkin", "caranya?", "kok yang dijelaskan atasan beda sih dengan kamu",
        "ok tolong dipelajari ya", "oke boleh", "tapi gak sesuai", "tapi mitranya group loan bukan modal",
        "ya bantu", "role aku apa", "\"ok pahammmm.,"
    ] or ql.startswith("tidak perlu lah saya sudah ikhlas") or ql.startswith("\"ok pahammmm"):
        return (
            "Interaksi & Percakapan",
            "Konfirmasi, Umpan Balik & Tanggapan Singkat Pengguna",
            "BP & BM (Semua Pengguna)",
        )

    # 12.3 Percakapan Santai / Di Luar Konteks (Off-Scope)
    if any(k in ql for k in [
        "ganteng", "pacar", "lapar", "laper", "ngantuk", "diah devita wijaya",
        "kamu kenal sya", "kamu namanya siapa", "kamu aku panggil wiwin", "kamu sudah makan belum",
        "kenapa ga makan", "aku laper", "sudah lapar", "aku lagi jalan", "win aku capek",
        "aku mau gadun kaya raya", "kecewa hiks", "basis otakmu apa", "aku bisa bertanya soal apa disini",
        "ai trainer bisa disambungkan ke wa gak ya", "apakah bisa konsultasi", "ko kamu gak tau sih",
        "lha mana kelanjutannya kok malah belajar ngerjain tugas",
        "aku ngantuk tapi tidak bisa minum kopi kenapa ya", "aku udah makan tadi", "cara cari pasangan"
    ]):
        return (
            "Interaksi & Percakapan",
            "Obrolan Santai, Humor & Pertanyaan di Luar Konteks Kerja (Off-Scope)",
            "BP & BM (Semua Pengguna)",
        )

    # Fallback default
    return (
        "Operasional & Lapangan",
        "Prosedur Kerja & Operasional Lapangan Umum",
        "BP & BM",
    )


def classify_query(query: str) -> dict:
    """Classify user query into 11 consolidated domain categories."""
    raw_cat, subtopic, role = classify_raw(query)
    full_key = f"{raw_cat}: {subtopic}"
    consolidated_category = CATEGORY_MAPPING.get(
        full_key,
        "Produk Pembiayaan, Proses Bisnis & Akuisisi Lapangan"
    )
    return {
        "category": consolidated_category,
        "subtopic": subtopic,
        "original_category": raw_cat,
        "role": role,
    }


def infer_pulau_from_regional(regional: str) -> str:
    """Infer island/pulau from regional or area name if island is missing."""
    r = (regional or "").lower().strip()
    if not r:
        return ""
    if "jawa" in r or "java" in r:
        return "Jawa"
    if any(k in r for k in ["sumatera", "sumatra", "sumbagsel", "sumbagut", "sumbagteng", "aceh", "lampung", "riau", "jambi", "bengkulu"]):
        return "Sumatera"
    if any(k in r for k in ["sulawesi", "sulsel", "sulteng", "sulut", "sultra", "gorontalo"]):
        return "Sulawesi"
    if any(k in r for k in ["kalimantan", "borneo", "kalbar", "kalteng", "kalsel", "kaltim", "kalut"]):
        return "Kalimantan"
    if any(k in r for k in ["bali", "nusa", "nusra", "ntb", "ntt", "lombok"]):
        return "Bali & Nusa Tenggara"
    if any(k in r for k in ["maluku", "papua"]):
        return "Maluku & Papua"
    return ""
