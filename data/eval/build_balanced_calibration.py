r"""Build a BALANCED, discrimination-valid D3 calibration set.

WHY THIS EXISTS (oracle D3 methodology)
A faithfulness judge is a DISCRIMINATOR. The natural 50-query seed produced an
effectively single-class sample (46/50 scored 1.0, 26/50 clean refusals) — κ is
mathematically unreachable on one class. To validate the judge we must inject a
real NEGATIVE class.

This script assembles a calibration set with three gold strata:
  - FAITHFUL  (gold=1.0): REAL generator answers, each verified fully grounded
                          in its retrieved context (provenance preserved).
  - HALLUCINATED (gold=0.0): REALISTIC hard negatives — confident false
                          assertions (wrong numbers that look like KB figures,
                          swapped names, fabricated products/regs, direct
                          contradictions). NEVER refusal-shaped. Hand-authored
                          against each base item's ACTUAL context.
  - PARTIAL   (gold=0.5): mostly grounded + exactly ONE unsupported claim, or a
                          world-knowledge fact presented as if from the KB.

Every synthetic item REUSES the exact `retrieved_context` the live graph
retrieved for its base query (pulled from the prior run JSON by id), so the
judge sees a real, faithful context — only the ANSWER varies. Emitted in the
harness PROVIDED-ANSWER schema ({query, answer, retrieved_context,
gold_faithful, intent:"PROVIDED"}), so calibrate_judge.py judges the supplied
answer directly (no generator), giving us both classes.

Run:
    .\.venv\Scripts\python.exe data/eval/build_balanced_calibration.py
Then calibrate (live judge on provided answers):
    .\.venv\Scripts\python.exe -m app.eval.calibrate_judge \
        data/eval/judge_calibration_balanced.json --concurrency 2
"""
from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).parent
_RUN = _HERE / "judge_calibration_seed_judge_run.json"
_OUT = _HERE / "judge_calibration_balanced.json"

# ── FAITHFUL (gold=1.0): reuse the REAL generated answer, verified grounded ───
# Each base_id's stored answer was read against its context and confirmed to
# make no unsupported claim. cal-003/cal-008 carry judge<1.0 (0.7/0.6) despite
# being faithful — kept deliberately as judge-vs-gold disagreement signal.
FAITHFUL_REUSE = [
    "cal-001", "cal-002", "cal-003", "cal-004", "cal-005", "cal-008",
    "cal-009", "cal-013", "cal-015", "cal-017", "cal-021", "cal-029",
    "cal-030", "cal-032", "cal-033",
]

# ── HALLUCINATED (gold=0.0): realistic hard negatives ─────────────────────────
# answer is authored; retrieved_context is pulled from base_id's real run.
HALLUCINATED = [
    {"id": "halu-001", "base": "cal-001", "note": "fabricated loan amounts/rate/tenor not in context",
     "answer": "Produk Modal menawarkan pinjaman mulai Rp 2.000.000 hingga Rp 50.000.000 dengan bunga tetap 1,5% per bulan dan tenor maksimal 24 bulan, dicairkan langsung ke rekening bank Mitra dalam 1x24 jam."},
    {"id": "halu-002", "base": "cal-015", "note": "corrupted years/borrowers/group size/amount",
     "answer": "Amartha memulai model group lending pada 2008–2014 dengan 5.000 peminjam pertama, melayani kelompok 10–15 Ibu Mitra dengan pinjaman awal Rp 1.000.000 per orang."},
    {"id": "halu-003", "base": "cal-002", "note": "swapped/invented principle name + fabricated guarantee",
     "answer": "Salah satu dari 8 Prinsip Client Protection Amartha adalah Prinsip ke-4: Competitive Pricing Guarantee, yaitu Amartha menjamin memberikan bunga termurah di seluruh industri dan akan memberi kompensasi tunai jika Mitra menemukan penawaran yang lebih murah di tempat lain."},
    {"id": "halu-004", "base": "cal-009", "note": "wrong ISO standard + fabricated OJK server claim",
     "answer": "Amartha menjaga kerahasiaan data nasabah sesuai standar ISO 9001 dan menyimpan seluruh data pribadi pelanggan di server pusat milik OJK yang tidak dapat diakses oleh karyawan Amartha."},
    {"id": "halu-005", "base": "cal-005", "note": "fabricated commission/bonus scheme",
     "answer": "Agen AmarthaLink memperoleh komisi tetap 10% dari setiap transaksi tarik tunai dan bonus bulanan sebesar Rp 500.000 bagi agen yang berhasil mencapai 100 transaksi dalam sebulan."},
    {"id": "halu-006", "base": "cal-008", "note": "DNA replaced with Amazon-style fabricated values",
     "answer": "8 DNA budaya Amartha adalah: Integrity First, Speed, Customer Obsession, Ownership, Frugality, Bias for Action, Think Big, dan Hire and Develop the Best."},
    {"id": "halu-007", "base": "cal-004", "note": "fabricated call-center number/email/SLA",
     "answer": "Nasabah dapat menyampaikan keluhan melalui call center 24 jam di nomor 1500-789 atau mengirim email resmi ke complaint@amartha.com, dengan jaminan SLA penyelesaian maksimal 3x24 jam kerja."},
    {"id": "halu-008", "base": "cal-029", "note": "visi and misi swapped (subtle contradiction)",
     "answer": "Visi Amartha adalah \"Mewujudkan kesejahteraan merata untuk semua\", sedangkan misinya adalah \"Kemakmuran Bersama\" yang berfokus pada pembangunan infrastruktur keuangan digital."},
    {"id": "halu-009", "base": "cal-003", "note": "fabricated product names",
     "answer": "Amartha menawarkan lima produk pembiayaan utama: Modal, Dana Kilat, KreditPlus, AmarthaPay, dan Celengan Emas, yang semuanya dapat diakses tanpa agunan."},
    {"id": "halu-010", "base": "cal-032", "note": "direct contradiction — context states NO late fees",
     "answer": "Untuk mendisiplinkan Mitra, Amartha menerapkan denda keterlambatan sebesar 2% per hari dari sisa pinjaman, sesuai Prinsip Responsible Pricing."},
    {"id": "halu-011", "base": "cal-017", "note": "fabricated 3 Value",
     "answer": "3 Value inti Amartha adalah Integrity, Innovation, dan Inclusion, yang menjadi fondasi seluruh budaya perusahaan."},
    {"id": "halu-012", "base": "cal-013", "note": "contradicts ethical collection code (asset seizure)",
     "answer": "Field Officer berwenang menyita aset jaminan milik Mitra yang menunggak lebih dari 30 hari dan menetapkan denda administratif tambahan sebagai bagian dari prosedur penagihan resmi Amartha."},
    {"id": "halu-013", "base": "cal-021", "note": "fabricated impact stats + scope",
     "answer": "Program pemberdayaan Amartha terbukti meningkatkan pendapatan Mitra perempuan rata-rata 300% hanya dalam 6 bulan dan telah menjangkau lebih dari 45 juta perempuan di seluruh Asia Tenggara."},
    {"id": "halu-014", "base": "cal-005", "note": "fabricated free insurance + Visa debit card",
     "answer": "Setiap Mitra aktif AmarthaLink otomatis mendapatkan asuransi jiwa gratis senilai Rp 100 juta serta kartu debit fisik berlogo Visa yang dapat digunakan di seluruh ATM."},
    {"id": "halu-015", "base": "cal-015", "note": "fabricated IPO + AUM",
     "answer": "Sejak 2022 Amartha resmi melantai di Bursa Efek Indonesia melalui IPO dan kini mengelola dana kelolaan lebih dari Rp 10 triliun dari investor institusi global."},
    {"id": "halu-016", "base": "cal-002", "note": "wrong principle count (8 not 10) + fabricated certification",
     "answer": "Client Protection Amartha terdiri dari 10 prinsip yang disertifikasi ulang secara langsung oleh lembaga Smart Campaign internasional setiap tahun."},
]

# ── PARTIAL (gold=0.5): mostly grounded + ONE unsupported claim ───────────────
PARTIAL = [
    {"id": "part-001", "base": "cal-001", "note": "grounded Modal desc + fabricated limit",
     "answer": "Produk Modal di Amartha adalah layanan pembiayaan mikro untuk pelaku usaha kecil, dengan pencairan fleksibel, pembayaran cashless, dan akses mudah lewat aplikasi AmarthaFin. Limit pinjaman maksimal mencapai Rp 25 juta per siklus."},
    {"id": "part-002", "base": "cal-002", "note": "correct 8 principles + unsupported quarterly OJK audit",
     "answer": "Ada 8 Prinsip Client Protection Amartha, mulai dari Appropriate Product Design hingga Governance and HR, yang mengatur perlindungan pelanggan secara menyeluruh. Seluruh prinsip ini diaudit langsung oleh OJK setiap kuartal."},
    {"id": "part-003", "base": "cal-013", "note": "correct FO roles + unsupported daily-visit mandate",
     "answer": "Field Officer menerapkan kode etik penagihan (dilarang menagih di luar jam kerja, masuk rumah tanpa izin, atau intimidasi) dan memasarkan produk tanpa paksaan. FO juga diwajibkan melakukan kunjungan harian ke rumah setiap Mitra binaannya."},
    {"id": "part-004", "base": "cal-005", "note": "correct services + commission-per-tx not stated (context says 'peluang pendapatan')",
     "answer": "AmarthaLink adalah jaringan agen layanan keuangan digital yang menyediakan setor/tarik tunai, jual pulsa & PPOB, dan kirim uang melalui AmarthaFin. Agen mendapat komisi dari setiap transaksi yang mereka proses."},
    {"id": "part-005", "base": "cal-015", "note": "correct Grameen def + world-knowledge (Yunus/Bangladesh) not in context",
     "answer": "Group lending model adalah model keuangan mikro akar rumput yang dipegang Amartha untuk melayani kelompok berpenghasilan rendah tanpa akses keuangan konvensional. Model ini terinspirasi langsung dari Grameen Bank Bangladesh yang didirikan oleh Muhammad Yunus."},
    {"id": "part-006", "base": "cal-017", "note": "correct values + unsupported annual survey-update",
     "answer": "Nilai inti Amartha terdiri dari 3 Value (Get Things Done & Never Settle, Collaborate & Communicate Effectively, Be Customer-Driven & Make an Impact) dan 8 DNA Amartha. Nilai-nilai ini diperbarui setiap tahun melalui survei karyawan."},
    {"id": "part-007", "base": "cal-009", "note": "correct CP definition + fabricated specific POJK number",
     "answer": "Client Protection (CP) adalah panduan agar produk dan layanan Amartha aman, adil, dan tidak merugikan pelanggan, terdiri dari 8 prinsip. Penerapan CP ini diwajibkan sesuai POJK Nomor 6 Tahun 2022."},
    {"id": "part-008", "base": "cal-030", "note": "correct misi/visi + unsupported 2030 target",
     "answer": "Amartha menjaga keberlanjutan usaha mitra lewat misi \"Mewujudkan kesejahteraan merata untuk semua\" dan visi \"Kemakmuran Bersama\". Perusahaan menargetkan menjangkau 100 juta penerima manfaat pada tahun 2030."},
    {"id": "part-009", "base": "cal-021", "note": "correct benefits + unsupported health-insurance perk",
     "answer": "Menjadi mitra Amartha meningkatkan pendapatan dan kesejahteraan, literasi keuangan, akses keuangan setara, dan memberdayakan perempuan. Mitra juga otomatis mendapatkan perlindungan asuransi kesehatan keluarga."},
    {"id": "part-010", "base": "cal-033", "note": "correct tech role + buzzword fabrication (AI/blockchain)",
     "answer": "Teknologi memperluas jangkauan layanan Amartha lewat AmarthaLink dan mengintegrasikan ekosistem digital AmarthaFin. Amartha juga menggunakan AI dan blockchain untuk mencatat seluruh transaksi Mitra secara terdesentralisasi."},
]


def main() -> int:
    if not _RUN.exists():
        raise SystemExit(f"Run file not found: {_RUN}. Run the live harness first.")
    run = json.loads(_RUN.read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in run.get("results", [])}

    def ctx_for(base_id: str) -> tuple[str, list]:
        src = by_id.get(base_id)
        if src is None:
            raise SystemExit(f"base id {base_id} not found in run results")
        return src["query"], (src.get("retrieved_context") or [])

    items: list[dict] = []

    # Faithful: reuse the REAL answer + REAL context.
    for bid in FAITHFUL_REUSE:
        src = by_id.get(bid)
        if src is None:
            raise SystemExit(f"faithful base id {bid} missing from run")
        items.append({
            "id": f"pos-{bid}",
            "query": src["query"],
            "answer": src["answer"],
            "retrieved_context": src.get("retrieved_context") or [],
            "gold_faithful": 1.0,
            "intent": "PROVIDED",
            "note": f"real grounded answer reused from {bid} (judge orig={src.get('judge_score')})",
        })

    # Hallucinated (gold=0.0) + Partial (gold=0.5): authored answer + base context.
    for spec, gold in [(h, 0.0) for h in HALLUCINATED] + [(p, 0.5) for p in PARTIAL]:
        query, ctx = ctx_for(spec["base"])
        items.append({
            "id": spec["id"],
            "query": query,
            "answer": spec["answer"],
            "retrieved_context": ctx,
            "gold_faithful": gold,
            "intent": "PROVIDED",
            "note": spec["note"],
        })

    _OUT.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    n_pos = sum(1 for i in items if i["gold_faithful"] == 1.0)
    n_halu = sum(1 for i in items if i["gold_faithful"] == 0.0)
    n_part = sum(1 for i in items if i["gold_faithful"] == 0.5)
    print(f"Wrote {len(items)} items to {_OUT.name}")
    print(f"  faithful (1.0): {n_pos}")
    print(f"  partial  (0.5): {n_part}")
    print(f"  halluc   (0.0): {n_halu}")
    print(f"  binarized@0.70 -> faithful={n_pos}, unfaithful={n_halu + n_part}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
