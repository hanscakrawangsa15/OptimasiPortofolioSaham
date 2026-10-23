# Revisi Bab 3 Laporan ETS (audit terhadap repo, 23 Okt 2026)

Dasar: `Laporan ETS.pdf` (Bab 3) dibandingkan dengan kode dan hasil di repo (`src/`, `hasil/csv/`, `data/laporan_data.md`).
Angka tabel di bawah berasal dari `hasil/csv/fase1_parameter_beku.json`, `fase1_*_grid.csv`, dan `fase2_hasil.csv`.

## A. Hasil audit

**Sudah sesuai repo (tidak perlu diubah):** 46 ticker awal dan 42 saham akhir, ticker yang dibuang (AMMN, GOTO, MBMA, PGEO),
1.163 hari harga dan 1.162 hari return, 35 hari awal dibuang (MTEL), ffill 3 hari, 8 stock split, Tabel 3.1 (semua periode,
volatilitas 1/N, σ_max 1,5×, return 1/N), eigenvalue minimum 1,98×10⁻², kromosom 2N+1 = 85 gen, langkah dekoder, aturan Deb,
Evaluator, anggaran 20.000, SBX/turnamen/elitisme GA, gerakan tetangga dan penerimaan SA, mekanisme DE/rand/1/bin, 30 masalah × 4 × 30 seed = 3.600 run.

**Perlu diubah:**

| # | Bagian | Isi di laporan | Kenyataan di repo | Tindakan |
|---|---|---|---|---|
| 1 | 3.3 akhir (populasi awal) | Hanya GA dan PSO yang disemai | Keempat algoritma disemai 10% populasi (GA 10, PSO 4, DE 10, SA 1) | Ganti paragraf (B.1) |
| 2 | 3.4.1 Tabel 3.7 | Hanya parameter awal GA; judul tabel salah ("Kode Evaluator") | GA ditala: populasi 400, laju mutasi 2×(1/N) | Ganti tabel (B.2) |
| 3 | 3.4.2 PSO | Terpilih w=1,0, c=1,0, v_max=1,0; 3 tahap (540+180+150 run); median R 0,4315; `pso_parameter_beku.json` | Penalaan diulang dengan semai: w=1,0, c₁=c₂=2,5, v_max=0,2; 39 titik (1.170 run); median R 0,4396; `fase1_parameter_beku.json` | Ganti (B.3) |
| 4 | 3.4.3 SA | Parameter dasar saja | Ditala: rasio suhu akhir 10⁻⁴ (awal 10⁻³); tidak berbeda nyata dari awal (p = 0,33) | Tambah (B.4) |
| 5 | 3.4.4 Tabel 3.11 DE | F=0,5, CR=0,9, **0 individu tersemai** | Ditala: F=0,1, CR=0,5; semai 10 individu | Ganti tabel (B.5) |
| 6 | 3.5 Fase 1 | "GA, SA, dan DE menggunakan parameter dasar" | Keempat algoritma ditala dengan protokol sama (103 titik, 3.090 run) | Ganti (B.6) |
| 7 | 3.5 uji statistik | Wilcoxon "berpasangan berdasarkan seed yang sama" | Wilcoxon berpasangan **antar masalah** (30 pasang median R), Holm 6 pasangan; Mann–Whitney untuk terpilih vs awal tiap algoritma | Ganti kalimat (B.7) |
| 8 | 3.5 Tabel 3.14 | Sharpe, CVaR 95%, shrinkage, selisih ke R acuan, keberhasilan | Semua kini dihitung (`fase2_hasil.csv`, `stats.ringkasan_latih`); definisi perlu ditulis | Tambah definisi (B.7) |
| 9 | 3.5 (profil risiko) | Tidak didefinisikan, padahal Bab 4.6.3 memakainya | Konservatif 1,5×, Moderat 2,0×, Agresif 2,5× (via GUI, bukan bagian 3.600 run) | Subbab baru (B.8) |
| 10 | 3.3 alasan cap 20% | "pada K kecil mustahil dipenuhi" | Tidak mustahil (kas bebas): cap 20% hanya membatasi dana terinvestasi ≤ 20% × K | Ganti kalimat (B.9) |
| 11 | 3.6 | 8 tab | 9 tab (tambah **Fase 2**); pilihan parameter terpilih/awal berlaku untuk keempat algoritma | Ganti (B.10) |
| 12 | Kecil | "suku bunga berbasis risiko"; "sukar menukar"; "tidal 30 masalah"; "optimum sejadi"; "evaluiasi"; Tabel 3.7 berjudul "Kode Evaluator"; Daftar Isi tidak memuat 3.3 | — | Perbaiki (B.11) |

## B. Teks pengganti (siap tempel)

### B.1 Paragraf populasi awal (akhir 3.3)

> Populasi awal dibuat acak seragam pada [0,1], dengan g₀ pada [0; 0,5]. Pada keempat algoritma, 10% individu awal disemai dengan
> heuristik: kunci = skor (μᵢ − r_f)/σᵢ yang dinormalkan ke [0,1] ditambah derau N(0; 0,05), bobot mentah U(0,3; 1), dan g₀ ∼ U(0; 0,1).
> Jumlah individu tersemai mengikuti proporsi yang sama: GA 10 dari 100 (40 dari 400 pada konfigurasi terpilih), PSO 4 dari 40,
> DE 10 dari 100, dan SA memulai dari satu solusi tersemai. Perlakuan awal yang sama ini menjaga agar perbedaan hasil berasal dari
> strategi pencarian, bukan dari kualitas populasi awal.

### B.2 Tabel 3.7 Parameter GA (ganti judul dan isi)

| Parameter | GA awal (PDF) | GA terpilih (Fase 1) |
|---|---|---|
| Ukuran populasi | 100 | 400 |
| Jumlah generasi (anggaran 20.000) | 200 (100 + 199 × 100) | 50 (400 + 49 × 400) |
| Seleksi | Turnamen k = 3, peringkat Deb | sama |
| Pindah silang | SBX, peluang 0,9, η = 15 | sama |
| Mutasi bobot dan g₀ | Gaussian σ = 0,1, peluang 1/N per gen | σ = 0,1, peluang 2/N per gen |
| Mutasi kunci | Tukar, peluang 0,2 | sama |
| Elitisme | 2 | 2 |
| Individu awal tersemai | 10 | 40 |

### B.3 PSO (ganti paragraf penalaan, Tabel 3.8, dan tahapan)

> Parameter PSO ditala pada Fase 1 (Subbab 3.5) dengan jendela 1, K = 7, σ_max 1,5×, dan seed 100–129 (di luar seed eksperimen utama
> 0–29). Penalaan memakai populasi awal yang disemai seperti pada Subbab 3.3, sehingga seluruh algoritma ditala dalam kondisi yang sama.

| Parameter | PSO awal | PSO terpilih |
|---|---|---|
| Ukuran kawanan | 40 | 40 |
| Bobot inersia w | 0,7 | 1,0 |
| Koefisien c₁ = c₂ | 1,5 | 2,5 |
| Kecepatan maksimum v_max | 0,2 | 0,2 |
| Jumlah iterasi | 500 | 500 |
| Individu awal tersemai | 4 | 4 |

> Penalaan berlangsung dalam tiga tahap: (A) w ∈ {0,4; 0,5; …; 1,0} × c ∈ {1,0; 1,5; 2,0} (21 titik); (A2) perluasan satu kali karena
> nilai terbaik berada di tepi (w ∈ {0,2; 0,3}, c ∈ {0,5; 2,5; 3,0}; 14 titik tambahan); (B) v_max ∈ {0,05; 0,1; 0,2; 0,5; 1,0} pada (w, c)
> terbaik (4 titik tambahan). Total 39 titik × 30 seed = 1.170 run. Kriteria: median R tertinggi; bila selisih ≤ 0,001 dipilih simpangan baku
> terkecil. Konfigurasi terpilih memiliki median R = 0,4396 dibanding 0,4358 pada parameter awal (Mann–Whitney p = 0,002). Nilai w = 1,0 adalah batas
> praktis atas (w > 1 tidak meredam kecepatan) sehingga tidak diperluas. Hasil dibekukan pada `hasil/csv/fase1_parameter_beku.json`.

(Catatan: penalaan PSO sebelumnya, tanpa semai, menghasilkan w = 1,0, c = 1,0, v_max = 1,0 dan median R 0,4315. Hasil itu tetap tersimpan di
`pso_parameter_beku.json` dan notebook PSO Fase 1, tetapi tidak dipakai di Fase 2. Bila ingin dipertahankan sebagai catatan, tulis sebagai
"penalaan pendahuluan tanpa semai".)

### B.4 SA (tambahan setelah Tabel 3.10)

> Pada Fase 1, SA ditala pada dua parameter: peluang penerimaan awal yang menentukan T₀ (target_acceptance ∈ {0,5; 0,8; 0,95}) dan laju pendinginan
> yang dinyatakan sebagai rasio suhu akhir terhadap awal (T_akhir/T₀ ∈ {10⁻²; 10⁻³; 10⁻⁴}, diperluas ke 10⁻⁵ karena terbaik di tepi), total 12 titik × 30 seed.
> Konfigurasi terpilih: peluang penerimaan awal 0,8 dan T_akhir = 10⁻⁴·T₀. Median R 0,4370 dibanding 0,4358 pada parameter awal (T_akhir = 10⁻³·T₀),
> selisih tidak bermakna (p = 0,33): SA relatif tidak sensitif terhadap kedua parameter ini pada rentang yang diuji.

### B.5 Tabel 3.11 Parameter DE

| Parameter | DE awal (PDF) | DE terpilih (Fase 1) |
|---|---|---|
| Ukuran populasi | 100 | 100 |
| Faktor mutasi F | 0,5 | 0,1 |
| Peluang pindah silang CR | 0,9 | 0,5 |
| Individu awal tersemai | 10 | 10 |
| Jumlah generasi | 200 | 200 |

> Penalaan DE: F ∈ {0,3; 0,5; 0,7; 0,9} × CR ∈ {0,5; 0,7; 0,9; 1,0} (16 titik), diperluas satu kali ke F ∈ {0,1; 0,2}, CR ∈ {0,1; 0,3} (20 titik tambahan);
> total 36 titik × 30 seed = 1.080 run. Median R terpilih 0,4423 dibanding 0,4366 pada parameter awal (p ≈ 6×10⁻⁹). F = 0,1 masih berada di tepi grid
> (nilai yang lebih kecil membuat vektor mutan nyaris tidak berbeda dari donornya), sehingga optimum F yang sebenarnya dapat lebih kecil; hal ini dicatat sebagai keterbatasan.

### B.6 Fase 1 (ganti paragraf Fase 1 di 3.5, dan butir 5–6 di 3.1)

> **Fase 1: Penalaan parameter** dilakukan untuk keempat algoritma dengan protokol yang sama: satu masalah representatif (jendela 1, K = 7, σ_max 1,5×), 30 run per titik
> grid pada seed 100–129, anggaran 20.000 evaluasi, populasi awal tersemai seperti pada Subbab 3.3. Grid kecil per algoritma mengikuti rencana penelitian
> (Tabel 3.15a) dan diperluas satu kali bila nilai terbaik berada di tepi. Pemilihan: median R tertinggi; selisih ≤ 0,001 dianggap seri dan dipilih simpangan baku terkecil.
> Parameter terpilih dibekukan sebelum Fase 2 dan dibandingkan dengan parameter awal pada seed penalaan yang sama (Mann–Whitney).

Tabel 3.15a Grid dan hasil Fase 1

| Algoritma | Parameter yang ditala | Titik (run) | Terpilih | Median R terpilih | Median R awal | p |
|---|---|---|---|---|---|---|
| GA | populasi × laju mutasi | 16 (480) | pop 400, laju mutasi 2/N | 0,4360 | 0,4315 | 0,005 |
| PSO | w × c, lalu v_max | 39 (1.170) | w 1,0, c 2,5, v_max 0,2 | 0,4396 | 0,4358 | 0,002 |
| SA | peluang penerimaan awal (T₀) × rasio suhu akhir | 12 (360) | 0,8; 10⁻⁴ | 0,4370 | 0,4358 | 0,33 |
| DE | F × CR | 36 (1.080) | F 0,1, CR 0,5 | 0,4423 | 0,4366 | 6×10⁻⁹ |

> Total 103 titik atau 3.090 run penalaan, terpisah dari 3.600 run Fase 2.

### B.7 Fase 2, metrik, dan uji statistik (ganti paragraf terkait di 3.5)

> **Fase 2** menjalankan setiap algoritma pada 30 masalah (6 data × 5 nilai K) dengan seed 0–29 yang sama, parameter beku Fase 1, dan anggaran 20.000 evaluasi:
> 30 × 4 × 30 = 3.600 run (72 juta evaluasi). Hasil per run disimpan dalam `hasil/csv/fase2_hasil.csv`.

Definisi metrik (tambahkan di bawah Tabel 3.14):

- **R acuan** per masalah: optimum brute force untuk K = 2; untuk K ≥ 3, R terbaik yang pernah ditemukan oleh seluruh run keempat algoritma pada masalah itu.
- **Selisih** = R acuan − median R. **Tingkat keberhasilan** = proporsi run layak dengan R ≥ 0,99 × R acuan (selisih relatif ≤ 1%).
- **Sharpe** (data uji) = (return setara tahunan − r_f) / volatilitas uji tahunan. **CVaR 95%** = rata-rata 5% return harian terburuk pada periode uji.
- **Shrinkage** = return setara tahunan uji − R latih.

Uji statistik (ganti kalimat "Wilcoxon berpasangan berdasarkan seed yang sama"):

> Perbedaan antar-algoritma diuji dengan uji peringkat bertanda Wilcoxon berpasangan atas 30 masalah, dengan median R antar 30 seed sebagai satu pasangan nilai
> per masalah, dan koreksi Holm untuk enam pasangan algoritma. Seed yang sama pada semua algoritma memastikan setiap algoritma menghadapi kondisi acak yang setara.
> Uji Mann–Whitney dipakai pada Fase 1 untuk membandingkan konfigurasi terpilih dengan parameter awal pada tiap algoritma. Perbedaan dinyatakan bermakna bila p terkoreksi < 0,05.

### B.8 Subbab baru: Skenario profil risiko (untuk Subbab 4.6.3)

> Profil risiko didefinisikan melalui pengali batas risiko terhadap volatilitas portofolio equal-weight: Konservatif 1,5×, Moderat 2,0×, dan Agresif 2,5×.
> Pada kasus tetap, σ_max masing-masing 0,2390; 0,3186; dan 0,3983 (untuk jendela lain lihat `data/laporan_data.md`). Skenario ini dijalankan melalui tab
> "Profil risiko" pada aplikasi Streamlit untuk data, K, dan jumlah seed yang dipilih, dan **bukan** bagian dari 3.600 run Fase 2 (yang memakai 1,5×).

(Temuan awal untuk Bab 4: pada kasus tetap, K = 7, σ solusi hanya 0,26–0,28, di bawah batas Moderat dan Agresif, sehingga kedua profil menghasilkan portofolio dan R
yang identik. Batas risiko mengikat hanya pada profil Konservatif. Ini perlu dijelaskan, atau gunakan K lebih kecil.)

### B.9 Alasan batas bobot (ganti kalimat di 3.3)

> Batas bobot 20% per saham hanya diberlakukan untuk K ≥ 7. Pada K yang lebih kecil, batas tersebut membatasi dana yang dapat diinvestasikan paling banyak
> 20% × K dari modal (misalnya K = 3 hanya 60%), sehingga sisanya terpaksa menjadi kas dan perbandingan antar-K tidak lagi setara.

### B.10 Subbab 3.6 (ganti kalimat daftar tab)

> Aplikasi Streamlit menyediakan sembilan tab: Ringkasan, Portofolio, Proses, Perbandingan, Data uji, Profil risiko, Hasil penelitian, Fase 2, dan Tentang.
> Pengguna memilih jendela data, K, modal, batas risiko, algoritma (GA, PSO, SA, DE), jumlah seed, serta parameter terpilih (Fase 1) atau parameter awal (PDF) untuk
> keempat algoritma. Tab Fase 2 menampilkan peringkat, uji Wilcoxon–Holm, dan sebaran hasil dari `fase2_hasil.csv`.

### B.11 Perbaikan kecil

- 3.3: "suku bunga berbasis risiko" → "suku bunga bebas risiko"; "mutasi sukar menukar" → "mutasi tukar (swap) menukar"; "evaluiasi" → "evaluasi".
- Tabel 3.7 berjudul "Kode Evaluator" → "Parameter GA".
- 3.5: "sehingga tidal 30 masalah" → "sehingga terdapat 30 masalah"; "optimum sejadi" → "optimum sejati".
- 3.4.1: "(100 evaluasi populasi awal + 199 * 100 evaluasi anak)/" → tutup kalimat dengan titik.
- Daftar Isi: tambahkan "3.3. Formulasi Masalah dan Penanganan Kendala" (nomor 3.3 ada di isi, tidak di daftar isi); tambahkan subbab baru dari B.8.
- Bab 1.1 dan Tabel 3.13: sudah konsisten; tambahkan baris "Parameter algoritma: dibekukan dari Fase 1" bila ingin lengkap.

## C. Catatan untuk Bab 4

Sumber angka Bab 4 yang kini tersedia di repo:

- Fase 2 (semua algoritma): `hasil/csv/fase2_hasil.csv`; ringkasan lewat `stats.ringkasan_latih`, `stats.peringkat_rata`, `stats.wilcoxon_holm`.
- Hasil in-sample: peringkat rata-rata DE 1,60; PSO 2,15; SA 2,78; GA 3,47. Semua pasangan berbeda nyata (Holm) kecuali PSO–SA.
- Hasil out-of-sample: median Sharpe uji negatif untuk keempat algoritma (kisaran −0,20 sampai −0,28); kepatuhan risiko uji sekitar 41–43%; ini perlu dibahas sebagai batas
  generalisasi (estimasi μ dari data historis noisy).
- Bab 4.3 (PSO) dan 4.5 (DE) sudah berisi tabel dan gambar, tetapi berasal dari notebook sebelum protokol disamakan (PSO dengan w = 1,0, c = 1,0,
  v_max = 1,0 tanpa semai; DE dengan F = 0,5, CR = 0,9 tanpa semai). Angka-angka itu tidak akan sama dengan Fase 2 yang baru dan perlu dibuat ulang
  dari `fase2_hasil.csv` agar konsisten dengan Bab 3. Bab 4.2 (GA) dan 4.4 (SA) masih kosong; datanya juga ada di berkas yang sama.
