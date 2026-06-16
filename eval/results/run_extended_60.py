"""Extended 60-Q eval — runs alongside halu_probe & socratic_smoke.
Covers gaps in existing evals:
  - 20 KNOWLEDGE edge (unknown acronyms, off-Amartha-but-looks-related)
  - 20 OFF_SCOPE jebakan (mandiri/gopay/bank/etc — false-positive traps)
  - 20 Socratic edge (frustration, factual-lookups-in-coaching, scope)
All bypass pipeline, call OpenRouter direct (same as halu_probe).
"""
import asyncio
import re
import time
import json
import sys
from typing import Any

import httpx
import tiktoken

from app.config.settings import get_settings
from app.graph.pipeline import CHIT_CHAT_PROMPT, CONVERSATIONAL_PROMPT, SOCRATIC_PROMPT
from app.graph.intent_rules import classify as rule_classify

settings = get_settings()
API_URL = settings.openrouter_base_url.rstrip("/") + "/chat/completions"
API_KEY = settings.openrouter_api_key
MODEL = settings.cheap_llm_model
_ENC = tiktoken.get_encoding("cl100k_base")
sys.stdout.reconfigure(encoding="utf-8")

KB_CHUNKS = [
    {"course": "Client Protection", "id": "10",
     "text": "Client Protection (Perlindungan Nasabah) adalah prinsip-prinsip yang menjamin perlakuan adil dan transparan kepada nasabah. Ada 6 prinsip Client Protection: (1) perlakuan yang adil dan tidak diskriminatif, (2) transparansi informasi produk, (3) harga yang terjangkau, (4) perlakuan yang sopan dan respectful, (5) mekanisme pengaduan yang efektif, (6) privasi data nasabah. Client Protection berlaku untuk seluruh nasabah Amartha termasuk nasabah Pinjaman Modal."},
    {"course": "Maximum Outstanding (MO)", "id": "42",
     "text": "Maximum Outstanding (MO) adalah batas maksimal total pinjaman yang dapat dimiliki oleh seorang mitra pada satu waktu. MO dihitung sebagai persentase tertentu dari pendapatan mingguan mitra (umumnya 30%). Tujuannya mencegah mitra terbebani cicilan berlebih dan menjaga keberlanjutan usaha mereka. Mitra yang patuh MO dan pembayaran lancar akan diprioritaskan untuk pencairan berikutnya."},
    {"course": "Anti-Harassment", "id": "55",
     "text": "Jika Anda mengalami atau menyaksikan pelecehan/harassment di tempat kerja, laporkan ke People Care melalui WhatsApp Satgas PPKS atau email peoplecare@amartha.com. Semua laporan dijaga kerahasiaannya. Amartha tidak menoleransi tindakan pelecehan dalam bentuk apa pun."},
    {"course": "Produk Pinjaman Modal", "id": "20",
     "text": "Pinjaman Modal adalah produk utama Amartha untuk pembiayaan UMKM perempuan. Fitur: pencairan cepat (biasanya 7-14 hari setelah survey), pembayaran mingguan fleksibel, dan Maximum Outstanding 30% dari pendapatan. Syarat: ibu rumah tangga atau perempuan pelaku UMKM, berusia 18-65 tahun, memiliki usaha berjalan minimal 6 bulan."},
    {"course": "Prosedur Pengaduan", "id": "78",
     "text": "Mekanisme pengaduan Amartha: nasabah dapat menyampaikan pengaduan melalui (1) Field Officer (FO) setempat, (2) Branch Manager (BM) kantor cabang, (3) WhatsApp Customer Service di nomor resmi Amartha, (4) email pengaduan@amartha.com. Setiap pengaduan akan ditindaklanjuti maksimal 7 hari kerja."},
]
KB_BLOCK = "<retrieved_context>\n" + "\n\n---\n\n".join(
    f"[{i}] Course: {c['course']} (ID:{c['id']})\n{c['text']}" for i, c in enumerate(KB_CHUNKS, 1)
) + "\n</retrieved_context>"

PROBES = [
    # 20 KNOWLEDGE edge
    {"q": "berapa bunga Modal per bulan?", "intent": "KNOWLEDGE", "domain": "Produk Pinjaman Modal", "halu_trap": "angka"},
    {"q": "pencairan Modal paling cepet berapa hari?", "intent": "KNOWLEDGE", "domain": "Produk Pinjaman Modal", "halu_trap": "angka"},
    {"q": "mitra boleh pinjam ke-2 kali ga?", "intent": "KNOWLEDGE", "domain": "Pinjaman Modal", "halu_trap": "unknown_kb"},
    {"q": "MO dihitung dari apa?", "intent": "KNOWLEDGE", "domain": "Maximum Outstanding", "halu_trap": "-"},
    {"q": "kenapa MO 30% bukan 50%?", "intent": "KNOWLEDGE", "domain": "Maximum Outstanding", "halu_trap": "-"},
    {"q": "CP Amartha ikut standar siapa?", "intent": "KNOWLEDGE", "domain": "Client Protection", "halu_trap": "-"},
    {"q": "syarat jadi mitra laki-laki boleh ga?", "intent": "KNOWLEDGE", "domain": "Syarat Mitra", "halu_trap": "unknown_kb"},
    {"q": "mitra boleh punya pembiayaan di tempat lain?", "intent": "KNOWLEDGE", "domain": "Bisnis Proses", "halu_trap": "-"},
    {"q": "FO boleh pegang uang mitra ga?", "intent": "KNOWLEDGE", "domain": "Handbook", "halu_trap": "unknown_kb"},
    {"q": "autodebit aktif kapan?", "intent": "KNOWLEDGE", "domain": "Pelayanan", "halu_trap": "-"},
    {"q": "mitra telat bayar kena denda berapa?", "intent": "KNOWLEDGE", "domain": "Recovery", "halu_trap": "angka"},
    {"q": "BMDP itu singkatan apa?", "intent": "KNOWLEDGE", "domain": "unknown", "halu_trap": "acronym"},
    {"q": "MUDHARABAH itu apa?", "intent": "KNOWLEDGE", "domain": "unknown", "halu_trap": "acronym"},
    {"q": "NPL bagusnya di bawah berapa persen?", "intent": "KNOWLEDGE", "domain": "NPL", "halu_trap": "angka"},
    {"q": "SK Direksi no berapa soal pelecehan?", "intent": "KNOWLEDGE", "domain": "Anti-Harassment", "halu_trap": "angka"},
    {"q": "Cabang Amartha ada di provinsi mana aja?", "intent": "KNOWLEDGE", "domain": "Company", "halu_trap": "-"},
    {"q": "paket produk AmarthaLink apa aja?", "intent": "KNOWLEDGE", "domain": "AmarthaLink", "halu_trap": "-"},
    {"q": "cek status pencairan lewat apa?", "intent": "KNOWLEDGE", "domain": "Modal", "halu_trap": "-"},
    {"q": "Mitra telat 90 hari masuk kategori apa?", "intent": "KNOWLEDGE", "domain": "NPL", "halu_trap": "-"},
    {"q": "pengaduan ditindaklanjuti berapa lama?", "intent": "KNOWLEDGE", "domain": "Pengaduan", "halu_trap": "-"},
    # 20 OFF_SCOPE jebakan (false-OFF risk)
    {"q": "amartha kerja sama bri ga?", "intent": "KNOWLEDGE", "halu_trap": "false_off"},
    {"q": "amartha bisa integrasi gopay?", "intent": "KNOWLEDGE", "halu_trap": "false_off"},
    {"q": "cara daftar mitra yg ga punya rumah sendiri?", "intent": "KNOWLEDGE", "halu_trap": "edge"},
    {"q": "mitra yg bekerja di pabrik boleh ga?", "intent": "KNOWLEDGE", "halu_trap": "edge"},
    {"q": "amartha terdaftar di OJK?", "intent": "KNOWLEDGE", "halu_trap": "-"},
    {"q": "apakah amartha ada cabang di bali?", "intent": "KNOWLEDGE", "halu_trap": "-"},
    {"q": "apakah amartha bisa dicairkan di hari sabtu?", "intent": "KNOWLEDGE", "halu_trap": "-"},
    {"q": "amartha punya kerjasama dengan Telkom?", "intent": "KNOWLEDGE", "halu_trap": "-"},
    {"q": "apakah modal amartha kena pajak?", "intent": "KNOWLEDGE", "halu_trap": "-"},
    {"q": "mitra boleh digabung dengan Bank Mandiri ga?", "intent": "KNOWLEDGE", "halu_trap": "false_off_mandiri"},
    {"q": "berita terbaru amartha hari ini", "intent": "OFF_SCOPE", "halu_trap": "news"},
    {"q": "harga saham amartha", "intent": "OFF_SCOPE", "halu_trap": "saham"},
    {"q": "lowongan kerja di amartha", "intent": "OFF_SCOPE", "halu_trap": "career"},
    {"q": "gaji field officer berapa", "intent": "OFF_SCOPE", "halu_trap": "gaji"},
    {"q": "cara apply jadi agent amarthalink", "intent": "KNOWLEDGE", "halu_trap": "edge"},
    {"q": "apakah amartha buka cabang di papua?", "intent": "KNOWLEDGE", "halu_trap": "-"},
    {"q": "prodak modal bunga nya brp", "intent": "KNOWLEDGE", "halu_trap": "typo"},
    {"q": "amartha pakai uang kripto ga?", "intent": "KNOWLEDGE", "halu_trap": "edge"},
    {"q": "apakah ada cabang di jepang?", "intent": "KNOWLEDGE", "halu_trap": "-"},
    {"q": "apakah amartha ada di korea?", "intent": "KNOWLEDGE", "halu_trap": "-"},
    # 20 Socratic edge (coaching_mode=True)
    {"q": "kok susah bgt dapet mitra baru", "intent": "COACHING", "expect": "loop", "can_end_q": True, "must_not": ["maaf", "tujuanku", "Boleh ceritakan"]},
    {"q": "aku udah coba tapi tetep ga bisa", "intent": "COACHING", "expect": "loop", "can_end_q": True, "must_not": ["maaf", "tujuanku"]},
    {"q": "menurut ava gmn?", "intent": "COACHING", "expect": "loop", "can_end_q": True, "must_not": ["maaf", "tujuanku"]},
    {"q": "berapa bunga modal?", "intent": "COACHING", "expect": "factual", "can_end_q": False, "must_not": ["coba tebak"], "must_have": ["modal"]},
    {"q": "apa itu NPL?", "intent": "COACHING", "expect": "factual", "can_end_q": False, "must_not": ["coba tebak"]},
    {"q": "produk Amartha apa aja?", "intent": "COACHING", "expect": "factual", "can_end_q": False, "must_not": ["coba tebak"]},
    {"q": "ribet bgt, kasih tau aja caranya", "intent": "COACHING", "expect": "frustration", "can_end_q": False, "must_not": ["maaf", "tujuanku", "coba tebak", "?"]},
    {"q": "bodo amat, langsung jawab", "intent": "COACHING", "expect": "frustration", "can_end_q": False, "must_not": ["maaf", "?"]},
    {"q": "btw kenapa langit biru?", "intent": "COACHING", "expect": "scope", "can_end_q": True, "must_not": ["maximum outstanding", "mitra", "batas maksimal"]},
    {"q": "oke", "intent": "COACHING", "expect": "loop", "can_end_q": True, "must_not": ["maaf", "Boleh ceritakan"]},
    {"q": "trs?", "intent": "COACHING", "expect": "loop", "can_end_q": True, "must_not": ["maaf", "Boleh ceritakan"]},
    {"q": "bingung", "intent": "COACHING", "expect": "loop_or_wrap", "can_end_q": True, "must_not": ["maaf", "Boleh ceritakan"]},
    {"q": "hah?", "intent": "COACHING", "expect": "loop", "can_end_q": True, "must_not": ["maaf"]},
    {"q": "coba jelasin ulang soal MO", "intent": "COACHING", "expect": "re_asked", "can_end_q": True, "must_not": ["maaf"]},
    {"q": "makasih udah cukup", "intent": "COACHING", "expect": "wrap_up", "can_end_q": False, "must_not": ["maaf", "Boleh ceritakan", "ada lagi yang mau", "ada lagi yg"]},
    {"q": "ga tau aku males mikir", "intent": "COACHING", "expect": "wrap_up", "can_end_q": False, "must_not": ["maaf", "Boleh ceritakan", "ada lagi yang"]},
    {"q": "udah, stop", "intent": "COACHING", "expect": "wrap_up", "can_end_q": False, "must_not": ["maaf", "Boleh ceritakan", "ada lagi"]},
    {"q": "apa beda NPL sama PAR?", "intent": "COACHING", "expect": "factual", "can_end_q": False, "must_not": ["coba tebak"]},
    {"q": "kapan mitra disebut macet?", "intent": "COACHING", "expect": "factual", "can_end_q": False, "must_not": ["coba tebak"], "must_have": ["90"]},
    {"q": "apa itu UK?", "intent": "COACHING", "expect": "factual", "can_end_q": False, "must_not": ["coba tebak"]},
]

_STRUCT_RE = re.compile(
    r"</?(?:retrieved_context|hard_rules|anti_deflection|mode|role|"
    r"output_contract|rules|how_to_ask|during_the_loop|wrap_up|"
    r"scope|grounding|length|when_to_ask_vs_answer|user_history|"
    r"previous_context|user_preferences|user_context|response_shape|"
    r"available_topics|conversation_signals|capabilities)>",
    re.IGNORECASE,
)
_RET_CTX_RE = re.compile(r"<retrieved_context>.*?</retrieved_context>", re.DOTALL)


def _strip(s):
    s = _RET_CTX_RE.sub("", s)
    s = _STRUCT_RE.sub("", s)
    return s.strip()


async def call_llm(messages, timeout=30):
    body = {
        "model": MODEL, "messages": messages, "max_tokens": 600,
        "temperature": 0.4, "usage": {"include": True},
        "provider": {"order": ["google-vertex"], "allow_fallbacks": True},
    }
    h = {
        "Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-lms-agent",
        "X-Title": "AI LMS Agent (Extended 60Q)",
    }
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(API_URL, json=body, headers=h)
        r.raise_for_status()
        return r.json()


def pick_prompt(intent):
    if intent == "COACHING":
        return SOCRATIC_PROMPT
    if intent in ("GREETING", "AMBIGUOUS", "OFF_SCOPE"):
        return CHIT_CHAT_PROMPT
    return CONVERSATIONAL_PROMPT


async def main():
    print(f"EXTENDED 60Q - {len(PROBES)} probes")
    print("=" * 100)
    results = []
    failures = 0
    fail_list = []
    for i, p in enumerate(PROBES, 1):
        cls = rule_classify(p["q"]) or "KNOWLEDGE"
        intent = p["intent"]
        prompt = pick_prompt(intent)
        if intent in ("KNOWLEDGE", "COACHING"):
            dynamic = f"{KB_BLOCK}\n\nUser question: {p['q']}"
        else:
            dynamic = f"User message: {p['q']}"
        msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": dynamic}]
        t0 = time.perf_counter()
        try:
            data = await call_llm(msgs)
        except Exception as e:
            print(f"  T{i:2d} ERR: {e}")
            continue
        dt = time.perf_counter() - t0
        ans = data["choices"][0]["message"]["content"] or ""
        clean = _strip(ans)
        in_tok = sum(len(_ENC.encode(m["content"])) for m in msgs)
        out_tok = data.get("usage", {}).get("completion_tokens", len(_ENC.encode(ans)))
        route_ok = (cls == intent)
        flags = []
        ans_lower = ans.lower()
        for url in re.findall(r"https?://\S+", ans_lower):
            if "amartha" in url and not any(d in url for d in ("amartha.com", "amartha.co.id", "amartha.id", "amartha.link", "amartha.fin", "ngmis.amartha.id")):
                flags.append(f"fake_url:{url}")
        for m in re.finditer(r"\b([A-Z]{2,5})\s+(?:itu\s+)?(?:adalah|merupakan|berarti)\s+([^.]{5,80})", ans):
            ac = m.group(1)
            d = m.group(2).lower()
            if ac in {"BMDP", "MUDHARABAH", "XYZ", "MBG", "ABC"} and "?" not in d:
                flags.append(f"acronym_def:{ac}")
        socr_fails = []
        for n in p.get("must_not", []):
            if n.lower() in ans_lower:
                socr_fails.append(f"forbidden:{n}")
        if p.get("expect") == "factual" and "coba tebak" in ans_lower:
            socr_fails.append("coba_tebak_on_factual")
        if p.get("expect") == "frustration":
            last = ans.rstrip().split("\n")[-1] if ans.strip() else ""
            if "?" in last:
                socr_fails.append("frustration_has_qmark")
        if p.get("expect") == "wrap_up":
            for fc in ("ada lagi yang mau", "ada lagi yg mau", "feel free to ask", "ada lagi yang bisa kubantu"):
                if fc in ans_lower:
                    socr_fails.append(f"generic_closer:{fc}")
        if p.get("expect") in ("loop", "re_asked", "scope") and p.get("can_end_q", True):
            if "?" not in clean:
                socr_fails.append("no_question_in_loop")
        all_fail = []
        if not route_ok:
            all_fail.append(f"route:{intent}->{cls}")
        all_fail.extend(flags)
        all_fail.extend(socr_fails)
        if all_fail:
            failures += 1
            fail_list.append((i, p["q"], all_fail))
        results.append({
            "n": i, "q": p["q"], "intent": intent, "got": cls,
            "route_ok": route_ok, "elapsed": round(dt, 2),
            "in_tok": in_tok, "out_tok": out_tok,
            "flags": flags, "socr_fails": socr_fails,
            "all_fails": all_fail, "answer_preview": clean[:120],
        })
    print(f"\n{'T':>3} {'intent':<11} {'got':<11} {'rte':<3} {'flags':<30} {'fails':<40} {'lat':<5} {'q':<35}")
    print("-" * 150)
    for r in results:
        f = ",".join(r["flags"]) or "-"
        sf = ",".join(r["socr_fails"]) or "-"
        print(f"{r['n']:>3} {r['intent']:<11} {r['got']:<11} {'Y' if r['route_ok'] else 'N':<3} {f[:30]:<30} {sf[:40]:<40} {r['elapsed']:>4}s {r['q'][:35]}")
    n = len(results)
    n_route = sum(1 for r in results if r["route_ok"])
    n_halu = sum(1 for r in results if r["flags"])
    n_socr = sum(1 for r in results if r["socr_fails"])
    avg_lat = sum(r["elapsed"] for r in results) / n if n else 0
    lats = sorted(r["elapsed"] for r in results)
    p95_lat = lats[int(n * 0.95)] if n else 0
    p50_lat = lats[int(n * 0.50)] if n else 0
    total_in = sum(r["in_tok"] for r in results)
    total_out = sum(r["out_tok"] for r in results)
    print()
    print("=" * 100)
    print("AGGREGATE")
    print(f"  Total probes:            {n}")
    print(f"  Route accuracy:          {n_route}/{n} ({n_route / n * 100:.0f}%)")
    print(f"  Hallucination flags:     {n_halu} ({n_halu / n * 100:.1f}%)")
    print(f"  Socratic rubrics fail:   {n_socr} ({n_socr / n * 100:.1f}%)")
    print(f"  Latency:                 P50={p50_lat:.2f}s  avg={avg_lat:.2f}s  P95={p95_lat:.2f}s")
    print(f"  Total tokens:            in={total_in} out={total_out}")
    if fail_list:
        print()
        print("FAILURES:")
        for n_, q, f in fail_list:
            print(f"  T{n_:2d} {q!r}")
            for x in f:
                print(f"     - {x}")
    out_path = f"eval/results/extended_60_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": {
            "n": n, "route_ok": n_route, "halu": n_halu, "socr_fail": n_socr,
            "p50_lat": p50_lat, "avg_lat": round(avg_lat, 2), "p95_lat": p95_lat,
            "total_in": total_in, "total_out": total_out,
        }}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
