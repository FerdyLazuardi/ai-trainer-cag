#!/bin/bash
# Test 40 studi case questions against RAG API
# Usage: bash data/test-questions/run_test.sh > data/test-questions/test_results.jsonl 2>&1

API="http://localhost:8001/api/v1/chat"
HEADERS="-H Content-Type:application/json"

# Scenario 1: FO baru bingung cara rekrut mitra
echo "=== SCENARIO 1 ==="
curl -s $API $HEADERS -d '{"query":"Saya BP baru di Amartha, ditugasin BM buat rekrut mitra baru di desa A. Jujur saya bingung harus mulai dari mana — apa yang harus saya lakukan pas pertama kali dateng ke lokasi?","conversation_id":"test-sc1","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Oke, saya udah nemu ibu-ibu yang mau dikasih tau soal produk Amartha. Tapi pas saya coba jelasin, mereka kelihatan bingung dan curiga. Gimana cara saya nge-pitch yang bener supaya mereka percaya? Struktur ngomongnya gimana?","conversation_id":"test-sc1","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Misal nanti ada yang setuju gabung, terus saya harus ngapain setelah itu? Proses seleksi sampe pencairannya gimana? Siapa aja yang terlibat?","conversation_id":"test-sc1","course_id":3}'
echo ""
sleep 2

# Scenario 2: Ada mitra telat bayar terus
echo "=== SCENARIO 2 ==="
curl -s $API $HEADERS -d '{"query":"Saya BM, cabang saya ada beberapa mitra yang mulai telat bayar angsuran 1-2 minggu. Saya liat di dashboard ada istilah DPD, Flowrate, NPL — sebenernya ini ngukur apa aja sih? Yang mana yang harus saya panik-in?","conversation_id":"test-sc2","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Saya udah liat data DPD 30+ naik dari bulan lalu. Tapi data kan cuma angka — saya perlu cari tau kenapa. Metode apa yang bisa saya pake buat nyari akar masalahnya lapangan?","conversation_id":"test-sc2","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Udah dapet akar masalahnya — ternyata BP di cabang saya jarang home visit mitra yang absen kumpulan. Sekarang gue harus bikin action plan. Formatnya gimana biar eksekusinya jelas dan terukur?","conversation_id":"test-sc2","course_id":3}'
echo ""
sleep 2

# Scenario 3: Briefing pagi gak efektif
echo "=== SCENARIO 3 ==="
curl -s $API $HEADERS -d '{"query":"Saya BM, briefing pagi saya tiap hari rasanya cuma seremonial. BP pada ngantuk, saya bacain data, selesai. Padahal waktu berharga. Sebenernya briefing pagi tuh harus diisi apa aja sih biar efektif?","conversation_id":"test-sc3","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Saya dengar ada 3 agenda utama briefing pagi. Dua udah saya jalanin (evaluasi kemaren + target hari ini), tapi yang ketiga soal target digitalisasi saya suka skip karena gak ngerti. Kira-kira penting gak sih bahas target digitalisasi tiap pagi? Contohnya ngomongin apa?","conversation_id":"test-sc3","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Berapa lama idealnya briefing pagi? Soalnya kalo kelamaan, tim lapangan jadi molor berangkatnya.","conversation_id":"test-sc3","course_id":3}'
echo ""
sleep 2

# Scenario 4: Mitra minta pencairan tapi riwayatnya jelek
echo "=== SCENARIO 4 ==="
curl -s $API $HEADERS -d '{"query":"Saya BM, ada mitra lama mau ngajuin pinjaman lanjutan. Tapi pas saya cek dashbooard, riwayat pembayaran dia sebelumnya sering telat. Tapi alesannya masuk akal — dulu pas lagi susah. Kira-kira saya kasih atau enggak?","conversation_id":"test-sc4","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Kalo saya tolak, BP saya ngotot karena mitra ini udah lama dan kasihan. Tapi kalo saya kasih, resiko NPL naik. Gimana cara saya ngejelasin ke BP tanpa bikin dia sakit hati?","conversation_id":"test-sc4","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Terus strategi buat naikin disbursement dari mitra yang bener-bener berkualitas itu gimana? Saya selama ini cuma fokus ngejar target mitra baru doang, mitra lanjutan saya diemin.","conversation_id":"test-sc4","course_id":3}'
echo ""
sleep 2

# Scenario 5: Ada oknum fraud di cabang
echo "=== SCENARIO 5 ==="
curl -s $API $HEADERS -d '{"query":"Saya BM, baru tau ada bp di cabang saya yang diduga narik uang dari mitra tapi gak disetor ke sistem. Jumlahnya udah puluhan juta. Ini pertama kali, saya panik dan bingung harus ngapain. Tolong kasih tau langkah-langkah yang harus saya lakukan.","conversation_id":"test-sc5","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Saya udah laporkan ke RM dan tim RCT. Tapi masalah lain muncul — mitra-mitra lain pada tau dan mulai panik. Ada yang ngamuk, ada yang mau narik semua dananya. Saya harus ngomong apa ke mereka?","conversation_id":"test-sc5","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Krisis ini katanya ada fase-fasenya. Kira-kira setelah situasi reda, apa yang harus saya lakukan biar kejadian serupa gak terulang? Sistem apa yang harus diperkuat?","conversation_id":"test-sc5","course_id":3}'
echo ""
sleep 2

# Scenario 6: Saya diminta jadi BM, tapi gak punya pengalaman leader
echo "=== SCENARIO 6 ==="
curl -s $API $HEADERS -d '{"query":"Saya baru naik jabatan dari BP jadi BM. Anak buah saya dulu temen satu tim. Saya bingung: saya harus bersikap keras biar dihormati, atau tetap santai biar mereka nyaman? Gimana dong?","conversation_id":"test-sc6","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Terus ada satu BP di tim saya yang kinerjanya jeblok. Tapi kalo saya tegur, dia defensif dan nyalahin faktor eksternal terus — mitranya gaptek, jalannya rusak, gak ada sinyal. Gimana saya bisa ngubah pola pikir dia?","conversation_id":"test-sc6","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Ada BP lain — sebut saja si B — kinerjanya mantap, udah senior. Saya malah bingung: saya perlu supervisi dia atau diemin aja? Soalnya takut kalo dicampuri malah bikin dia risih.","conversation_id":"test-sc6","course_id":3}'
echo ""
sleep 2

# Scenario 7: Mitra mengeluh ditagih kasar
echo "=== SCENARIO 7 ==="
curl -s $API $HEADERS -d '{"query":"Saya BP, dapat laporan dari ketua majlis kalau ada mitra yang ngeluh saya dianggap terlalu keras pas nagih. Padahal saya cuma ngingetin aja. Saya jadi takut kena tegur BM. Sebenernya aturan main tagih di Amartha itu kayak gimana?","conversation_id":"test-sc7","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Selain cara nagih, saya juga sering ditanya mitra soal biaya dan bunga. Kadang saya jawab sekenanya. Apa sih prinsip Amartha soal transparansi harga dan produk? Yang wajib saya sampaikan ke mitra apa aja?","conversation_id":"test-sc7","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Terus soal data mitra — kadang mitra nitipin HP sama saya minta diisiin aplikasi. Boleh gak sih saya bantu? Atau ini resiko?","conversation_id":"test-sc7","course_id":3}'
echo ""
sleep 2

# Scenario 8: FOKUS target mitra RF Poket turun drastis
echo "=== SCENARIO 8 ==="
curl -s $API $HEADERS -d '{"query":"Saya BM, lagi pusing — bulan ini jumlah mitra yang bayar pake Poket turun drastis dibanding bulan lalu. Padahal bulan lalu sempet naik. Saya curiga ada yang salah. Kira-kira dari mana saya mulai investigasi?","conversation_id":"test-sc8","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Udah saya trace pake 5 Whys. Ketemu akar masalahnya: ternyata BP saya selama ini yang ngeoperasiin HP mitra pas bayar, bukan mitranya langsung. Sekarang bulan ini mitra disuruh mandiri, jadinya pada gak bisa. Gimana solusinya?","conversation_id":"test-sc8","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Saya juga lagi dikejar target 3 pilar operasional — Growth, Recovery, Digitalisasi. Tapi mana yang harus saya prioritasin kalo sumber daya terbatas? Kadang saya ngerasa semuanya penting.","conversation_id":"test-sc8","course_id":3}'
echo ""
sleep 2

# Scenario 9: Anti-Harassment training wajib
echo "=== SCENARIO 9 ==="
curl -s $API $HEADERS -d '{"query":"Saya di People Care, ditugasin sosialisasi Anti-Harassment ke semua karyawan. Tapi responnya biasa aja — ah training mandatory lagi. Padahal ini penting. Menurut dokumen yang ada, pelecehan itu sebenernya definisinya apa dan bentuknya apa aja?","conversation_id":"test-sc9","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Terus saya denger ada Satgas PPKS. Itu tugasnya ngapain? Kalo ada korban, lapor kemana aja sih salurannya?","conversation_id":"test-sc9","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Satu hal yang bikin saya bingung: katanya kalo pelaku terbukti bersalah di-PHK. Tapi kalo gak terbukti gimana? Selain itu, apa kewajiban saya sebagai pimpinan buat nyegah pelecehan di tim saya?","conversation_id":"test-sc9","course_id":3}'
echo ""
sleep 2

# Scenario 10: Tim gak kompak, achievement turun
echo "=== SCENARIO 10 ==="
curl -s $API $HEADERS -d '{"query":"Saya BM, tim saya akhir-akhir ini gak kompak. BP pada jalan masing-masing, gak ada kerjasama. Yang senior sombong, yang junior minder. Target disbursement mulai meleset. Gimana caranya bikin tim solid?","conversation_id":"test-sc10","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Kalo soal hierarki — saya dengar di Amartha ada prinsip Power Distance. Itu apa? Apakah saya harus menjaga jarak sama anak buah atau justru sebaliknya?","conversation_id":"test-sc10","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Saya lihat ada BP yang keliatan lagi punya masalah pribadi — sering telat, kurang fokus, performa drop. Saya pengen bantu tapi gak enak, takur dianggap terlalu masuk urusan orang. Emangnya itu urusan saya sebagai leader?","conversation_id":"test-sc10","course_id":3}'
echo ""
sleep 2

# Scenario 11: Problem solving muter-muter doang
echo "=== SCENARIO 11 ==="
curl -s $API $HEADERS -d '{"query":"Saya BM, setiap ada masalah di cabang, saya dan tim rapat muter-muter doang — bahas hal yang sama berulang, gak ada solusi konkret. Katanya ada siklus PDCA. Itu apa dan gimana cara terapin biar masalah gue beres?","conversation_id":"test-sc11","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Saya juga suka bingung nentuin target — kadang target saya abstrak kayak tingkatkan kualitas portofolio. Ada metode SMART di dokumen. Contoh konkretnya gimana buat target cabang saya?","conversation_id":"test-sc11","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Kalo udah dapet solusi, cara milih solusi mana yang jalanin duluan gimana? Saya kadang suka ambil yang gampang tapi dampaknya kecil.","conversation_id":"test-sc11","course_id":3}'
echo ""
sleep 2

# Scenario 12: BP gak bisa komunikasi
echo "=== SCENARIO 12 ==="
curl -s $API $HEADERS -d '{"query":"Saya BM, BP baru saya pinter soal produk tapi kalo ngomong sama mitra kaku dan gak enak didenger. Mitra pada gak respect. Padahal secara teknis dia mampu. Kira-kira apa yang kurang dari dia?","conversation_id":"test-sc12","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Kalo lewat WhatsApp juga sering disalahpahamin mitra karena pesannya kepanjangan dan pake istilah internal kantor. Etika chat WA yang bener gimana? Kasih contoh yang bener sama yang salah.","conversation_id":"test-sc12","course_id":3}'
echo ""
sleep 2

# Scenario 13: Mitra meninggal dunia
echo "=== SCENARIO 13 ==="
curl -s $API $HEADERS -d '{"query":"Saya BP, mitra saya meninggal dunia. Keluarganya bingung — katanya ada asuransi yang bisa nutup sisa pinjaman. Apa benar? Terus apa aja yang harus disiapin keluarganya?","conversation_id":"test-sc13","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Siapa yang berhak ngajuin klaim? Apakah keluarganya bisa ngurus sendiri atau harus lewat FO? Terus kalo ada masalah di tengah jalan — misal dokumen gak lengkap — saya harus ngelapor kemana?","conversation_id":"test-sc13","course_id":3}'
echo ""
sleep 2

# Scenario 14: Target disbursement gak nyampe
echo "=== SCENARIO 14 ==="
curl -s $API $HEADERS -d '{"query":"Saya BM, udah 2 bulan target disbursement cabang saya gak nyampe. Rasanya semua udah saya coba — BP saya suruh sosialisasi gencar — tapi hasilnya gitu-gitu aja. Cara baca Sales Funnel gimana biar saya tau bocornya di tahap mana?","conversation_id":"test-sc14","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Saya udah analisa funnel BP saya. Contoh kasus: BP A konversi dari potensi ke onboarding bagus banget, tapi drop di tahap cair (cuma 30% yang approve). Ini diagnose-nya apa? Coaching yang tepat sambil apa?","conversation_id":"test-sc14","course_id":3}'
echo ""
sleep 2
curl -s $API $HEADERS -d '{"query":"Saya juga dengar ada target 3 pilar operasional di Amartha. Growth, Recovery, Digitalisasi. Kira-kira dari dashboard, metrik apa yang harus saya monitor tiap hari biar gak telat ngambil tindakan?","conversation_id":"test-sc14","course_id":3}'
echo ""
