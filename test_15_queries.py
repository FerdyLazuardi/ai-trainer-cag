import asyncio
import httpx
import json

queries = [
    "Mitra saya ngeluh pencairan dananya ngga sesuai sama akad pas survei, dia harus lapor kemana ya?",
    "Bro, data pendaftaran Modal mitra gue ada yang salah nih, gue mesti hubungin siapa?",
    "Halo, saya mau join grup WA buat koordinasi Modal, kontaknya ke siapa ya?",
    "Titik lokasi di sistem belum berubah juga padahal udah dinonaktifkan di HP mitra, ini lapornya lewat apa?",
    "Ada mitra yang mau komplain karena udah 2 minggu sejak survei belum ada kejelasan cair apa ngga, lapor kemana?",
    "Eh, kalo mau cek mitra yang mundur pencairan itu kita dapet infonya dari mana sih?",
    "Gue liat ada indikasi fraud nih pas di lapangan, harus gimana dan lapor ke siapa?",
    "Kalo mitra komplain pencairan Modal bermasalah, gue lapornya ke Amartha Care atau kemana?",
    "Kalo titik Point masih belum update di aplikasi, pas mau lapor itu ngelampirin apa aja?",
    "Siapa sih PIC buat masukin saya ke WAG Modal?",
    "Dana yang diterima mitra kurang dari yang di akad, darurat nih, bisa minta nomernya ga?",
    "Kalau nemu pelanggaran integritas atau fraud, apa saya boleh selesaikan sendiri sama mitranya?",
    "Buat tau update pemberitahuan mitra yang mundur pencairan, BM ngeceknya di mana?",
    "Saya mau gabung koordinasi tim FO buat produk Modal, kontaknya ada ngga?",
    "Lapor ke PST soal titik Point itu detailnya apa aja yang dikirim?"
]

async def fetch(client, i, q):
    try:
        resp = await client.post("http://localhost:8001/api/v1/chat", json={"query": q}, timeout=30.0)
        data = resp.json()
        return i, q, data.get("answer", "ERROR"), data.get("latency_ms", 0)
    except Exception as e:
        return i, q, str(e), 0

async def main():
    async with httpx.AsyncClient() as client:
        tasks = [fetch(client, i, q) for i, q in enumerate(queries)]
        results = await asyncio.gather(*tasks)
        
    for i, q, ans, lat in sorted(results):
        print(f"**Q{i+1}: {q}**")
        print(f"> *Ava:* {ans}")
        print(f"_(Latency: {lat:.0f}ms)_")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(main())
