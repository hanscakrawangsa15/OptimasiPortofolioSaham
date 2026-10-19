# Optimasi Alokasi Dana Saham LQ45 dengan Batas Risiko dan Kardinalitas

Menggantikan spesifikasi awal maksimasi Sharpe kontinu (dihapus; ada di riwayat git).

**Sumber:** `Optimasi Alokasi Dana Saham LQ45 dengan Batas Risiko dan Kardinalitas.pdf` (rencana, otoritas
utama) dan Ashraf dkk. (2026), *Invasive weed optimization based metaheuristic approach for solving
constrained risk budgeted portfolio selection problem*, Alexandria Eng. J. 135, 273-317 (rujukan metodologi
eksperimen; modelnya berbeda dan IWO tidak dipakai).

## 1. Tujuan

Memaksimalkan return ekspektasi portofolio saham LQ45 di bawah batas risiko, dengan batas jumlah saham K,
batas 20% per saham untuk K >= 7, dan pembelian dalam lot. Membandingkan GA, SA, PSO, DE dengan acuan eksak
dan heuristik sederhana pada anggaran evaluasi sama, in-sample (kualitas algoritma) dan out-of-sample
walk-forward (kualitas strategi).

## 2. Formulasi

- N = 42 saham. Variabel: lot n_i >= 0 bulat. Nilai posisi v_i = 100 * n_i * P_i (P_i = harga RAW pada hari
  terakhir jendela latih). Bobot w_i = v_i / D. Kas w_0 = 1 - sum w_i, return kas = r_f. Kas boleh sengaja ditahan (gen g_0, keputusan 11) untuk memenuhi
  batas risiko; ia bukan hanya sisa pembulatan.
- Maksimalkan R = sum w_i mu_i + w_0 r_f.
- Kendala: sqrt(w' Sigma w) <= sigma_max; w_i <= c_K (c_K = 0,20 bila K >= 7, tanpa batas bila K < 7);
  w_0 >= 0; paling banyak K saham dengan n_i >= 1 (saham terpilih = n_i >= 1; saham yang kebagian 0 lot
  dianggap tidak terpilih).
- mu, Sigma tahunan dari return SEDERHANA jendela latih (x252). sigma_max = 1,5 x volatilitas portofolio 1/N
  (42 saham) pada jendela latih. r_f = 5%/tahun. Kas <= 5% D adalah kendala lunak: hanya dilaporkan.
- Tanpa short selling, leverage, biaya transaksi, rebalancing dalam jendela uji (buy-and-hold 126 hari).

## 3. Keputusan yang sudah disepakati

| # | Keputusan |
|---|---|
| 1 | Skala: K in {2,3,5,7,10}; sigma_max 1,5x saja; D = Rp 5 juta. `D_LEVELS` lain hanya untuk ekstensi sensitivitas. |
| 2 | Walk-forward memakai **5 jendela** (504/126/126) = 25 masalah. Kasus tetap (latih s.d. 2025-09-30, 923 hari, uji 239 hari) = 5 masalah tambahan untuk konvergensi dan perbandingan in-sample. Total 30 masalah. |
| 3 | Kas bernilai r_f; bobot terhadap D (batas 20% = 20% dari modal). |
| 4 | Acuan eksak utama: **SCIP** (MIQCP lewat PySCIPOpt, optimum lot bulat dengan status dan gap). Enumerasi himpunan + QP kontinu sebagai batas atas relaksasi dan cek silang untuk K <= 5. Tanpa cvxpy. |
| 5 | 30 seed (0..29) per (algoritma, masalah); seed sama antar algoritma. |
| 6 | Anggaran 20.000 evaluasi per run untuk semua metaheuristik; konvergensi dicatat tiap 200 evaluasi. |
| 7 | IWO tidak dimasukkan (bisa ditambah kemudian sebagai `iwo.py`). |
| 8 | Statistik: deskriptif + Wilcoxon berpasangan dengan koreksi Holm antar masalah; Kruskal-Wallis dalam satu masalah (ala makalah) opsional. Friedman dan block bootstrap ditunda. |
| 9 | Dekoder memfilter saham yang 1 lot-nya melebihi D x c_K (tidak mungkin dibeli) sebelum memilih top-K. Penyimpangan kecil dari pseudo-code PDF supaya slot K tidak terbuang. |
| 10 | Dekoder TIDAK membagi sisa dana greedy; sisa uang setelah pembulatan ke bawah menjadi kas (sesuai PDF). |
| 11 | Kromosom berpanjang **2N+1**: N kunci, N bobot mentah, dan satu gen kas g_0 in [0,1] (porsi kas awal). Alasan: pada K kecil batas risiko hanya bisa dipenuhi dengan menahan kas (volatilitas saham tunggal terkecil 0,226 vs sigma_max 0,239). Dengan 2N, metaheuristik tidak dapat mewakili sebagian optimum model (jendela 3, K=2: R maksimum 0,241 vs 0,282 dengan kas bebas). Berlaku untuk GA, SA, PSO, dan DE. |
| 12 | Saham terpilih = n_i >= 1. Harga beli = Close raw pada hari terakhir jendela latih (setara beli-dan-tahan sejak penutupan itu; konsisten dengan return uji hari pertama). |

Penyesuaian angka terhadap PDF: data kini 1.162 hari return (bukan sekitar 1.167) karena 5 hari fiktif
terbuang; uji kasus tetap 239 hari.

## 4. Struktur dan sub-proyek

```
config.py, main.py
src/ scrappingdata.py  data.py  problem.py  decoder.py  fitness.py
     algorithms/ exact_scip.py  enum_relax.py  greedy.py  topk_qp.py  random_equal.py  ga.py  sa.py  pso.py  de.py
     evaluasi.py  stats.py  visualisasi.py  laporan_data.py (tetap)
tests/  hasil/
```

Sub-proyek (masing-masing spesifikasi dan rencana sendiri):

- **A. Fondasi** (spesifikasi rinci di bagian 5): `config`, `data`, `problem`, `decoder`, `fitness`.
- **B. Acuan:** SCIP eksak, enumerasi relaksasi, greedy forward, top-K + QP, acak valid (1.000 kali).
- **C. Metaheuristik:** GA (dengan uji brute force K=2), SA, PSO, DE; perbandingan adil lewat dekoder dan
  `Evaluator` bersama.
- **D. Eksperimen dan analisis:** Fase 1 penalaan (jendela 1, K=7, 1,5x; 30 run per titik grid), Fase 2
  (CSV long), `evaluasi`, `stats`, `visualisasi`, `main`.

Berkas pipeline lama (maksimasi Sharpe: `portofolio.py`, `baseline.py`, `GeneticAlgorithm.py`, tes dan
hasilnya, spesifikasi dan rencana lama) sudah dihapus; `main.py`, `evaluasi.py`, `visualisasi.py` menjadi stub
yang ditulis ulang di D. `walkforward.py` dilebur ke `data.py` di A.

## 5. Sub-proyek A: antarmuka

**config.py.** `K_LIST=[2,3,5,7,10]`, `SIGMA_MULT=[1.5]`, `D_UTAMA=5e6`, `D_LEVELS` dipertahankan,
`MAX_WEIGHT=0.20`, `CAP_BERLAKU_MULAI_K=7`, `LOT_SIZE=100`, `N_RUNS=30`, `BUDGET_EVAL=20000`,
`HISTORY_STEP=200`, `N_JENDELA=5`, parameter walk-forward, `RISK_FREE`.

**data.py.**
- `muat_data(data_dir) -> (return_df, harga_raw_df)`.
- `buat_jendela`, `kasus_tetap`, `simpan_jendela`, `muat_jendela` (dipindah dari `walkforward.py`).
- `muat_skenario(data_dir, K_list, sigma_mult, D) -> list[Skenario]` untuk 5 jendela dan kasus tetap.
- `Skenario`: `id` (mis. `w3_K7_s1.5_D5000000`), `jendela`, `K`, `sigma_mult`, `D`, `problem`, `return_uji`,
  `tanggal_uji`, `tanggal_beli`. Hanya `problem` yang diberikan ke algoritma, supaya data uji tidak bocor.

**problem.py.**
- `Problem` (frozen dataclass): `mu, cov, price, D, K, sigma_max, tickers, cap, lot=100, rf=0.05`.
  `cap = 0.20` bila K >= 7, selain itu 1,0. Properti `harga_lot = lot*price`, `layak_beli`
  (`harga_lot <= D*cap`), dan `n_genes = 2N+1` (semua algoritma membaca dimensi dari sini).
- `weights(lots)`, `cash(lots)`, `evaluate(lots) -> (R, sigma, viol)` (juga untuk batch baris).
  `R = w@mu + cash*rf`, `sigma = sqrt(w@cov@w)`, `viol = max(0, sigma - sigma_max)`.
- `Result`: `lots, R, sigma, viol, feasible, history (n_eval, best_R_layak tiap 200), n_eval, seconds, seed`
  dan `status` (opsional, untuk solver eksak). Antarmuka algoritma: `run(problem, budget, seed) -> Result`.

**decoder.py.** Vektor x berpanjang 2N+1: `keys = x[:N]`, `raw = x[N:2N]`, `g0 = x[2N]`.
1. Hanya saham `layak_beli` yang boleh dipilih; ambil paling banyak K dengan kunci terbesar.
2. Kas awal w_0 = g0 (dipotong ke [0,1]). Bobot mentah dipotong ke [0,1] dan dinormalisasi di himpunan
   terpilih (semua nol -> sama rata) sehingga berjumlah 1 - g0.
3. Bila cap < 1: potong yang melebihi cap, bagikan kelebihan ke yang belum penuh, ulangi sampai stabil;
   sisa yang tidak bisa dibagi menjadi kas.
4. `n_i = floor(w_i * D / (lot * P_i))`. Saham dengan n_i = 0 tidak terpilih. Sisa dana (g0 + kelebihan cap
   + sisa pembulatan) = kas.
5. `decode(x, P)` dan versi vektor `decode_batch(X, P)` yang konsisten bit-per-bit.
Fungsi bersama: `split(x, N) -> (keys, raw, g0)` dan `init_population(problem, rng, n) -> (n, 2N+1)`
(`g0` awal seragam di [0, 0,5]; baris pertama boleh disemai top-K dengan g0 rendah). Semua algoritma memakai
keduanya; tidak ada yang mengiris vektor atau memilih dimensi sendiri.
Keluaran selalu memenuhi: jumlah saham <= K, bobot <= cap, lot bulat, dana terpakai <= D, kas >= 0. Hanya
batas risiko yang bisa dilanggar.

**fitness.py.**
- `Evaluator(problem, budget)`: pintu tunggal evaluasi untuk semua metaheuristik. `evaluate_batch(X)`
  menolak lebar selain `n_genes`, mendekode, menghitung (R, sigma, viol), menambah hitungan evaluasi (satu vektor = satu evaluasi), menolak
  melewati anggaran, melacak solusi terbaik menurut Deb, dan mencatat riwayat tiap `HISTORY_STEP`.
- Aturan Deb: layak (viol == 0) mengalahkan tak layak; sesama layak, R lebih besar menang; sesama tak
  layak, viol lebih kecil menang. Fungsi: `deb_better(a, b)`, `deb_key` (untuk pengurutan), `deb_best_index`.

**Pengujian A.**
- Dekoder: 10.000 vektor acak per K pada jendela nyata -> saham <= K, bobot <= cap, lot bulat, dana <= D,
  saham tak terjangkau tidak pernah terpilih; `decode_batch` sama dengan `decode`; deterministik.
- Kasus K=1 (hanya uji unit): satu saham, dana habis sampai sisa kurang dari satu lot (g0 = 0).
- Gen kas: g0 = 1 -> semua kas (R = r_f, sigma = 0, layak); g0 = 0 setara 2N; kas akhir >= g0 - toleransi
  pembulatan; contoh nyata jendela 3, K=2: kromosom (BRIS, ISAT, v = 0,55:0,45, g0 = 0,30) menghasilkan
  7 dan 6 lot, kas 33,5%, sigma 0,1952, R 0,2817.
- `Evaluator` menolak vektor selain lebar `n_genes`; `split` dan `init_population` berdimensi benar.
- `Problem.evaluate` cocok dengan hitungan tangan pada data sintetis.
- Deb: urutan layak/tak layak/campuran, dan konsistensi `deb_key` dengan `deb_better`.
- `Evaluator`: hitungan evaluasi benar, error bila melewati anggaran, riwayat tiap 200, terbaik monoton.
- `muat_skenario`: 30 skenario (25 + 5), `Problem` tidak memuat return uji, sigma_max = 1,5 x vol 1/N.

## 6. Eksperimen (gambaran untuk sub-proyek C dan D)

Fase 0 kelayakan (K=5 juga dijalankan dengan cap 20% sebagai kontrol; dengan kas bebas, cap 20% juga layak untuk K<5 tetapi
dana terinvestasi paling banyak 20% x K, jadi tidak dijalankan). Fase 1 penalaan hanya di data latih
jendela 1 (K=7, 1,5x), grid kecil per algoritma, 30 run per titik, konfigurasi dibekukan. Fase 2: 30 masalah
x 4 metaheuristik x 30 run = 3.600 run x 20.000 evaluasi, ditambah acuan deterministik dan acak valid
(1.000 kali). Keluaran CSV long: R/sigma in-sample, kelayakan, himpunan, bobot, lot, kas, evaluasi sampai
terbaik, waktu, dan R/sigma/Sharpe/MDD/CVaR95 out-of-sample, shrinkage, realisasi risiko.
Parameter algoritma awal sesuai tabel PDF (GA pop 100 x 200 gen, SBX 0,9 eta 15; SA 20.000 iterasi; PSO 40 x
500; DE 100 x 200, F 0,5, CR 0,9), dengan vektor 2N+1. Hitungan anggaran: GA/DE = 100 populasi awal + 199
generasi x 100 = 20.000; PSO = 40 awal + 499 x 40; SA = 1 awal + 19.999 gerakan. Perturbasi Gaussian SA (iii)
ikut mengganggu g0. Mutasi swap GA hanya pada kunci. Penalaan Fase 1 boleh mengubah sebaran awal g0. Aturan penafsiran ditetapkan sebelum melihat hasil (PDF bagian 4).

## 7. Di luar cakupan

Short selling, leverage, biaya transaksi, rebalancing, dividen sebagai arus kas, IWO, NSGA-II,
estimator shrinkage (ekstensi lanjutan).

## 8. Catatan dan keterbatasan yang diwarisi

- Close Yahoo sudah disesuaikan untuk stock split; "harga raw" hanya tanpa penyesuaian dividen. Untuk saham
  yang split sesudah tanggal beli sebuah jendela (ISAT 4:1 pada 2024-10-14 di jendela 1-2; BRPT sangat
  kecil), lot jendela itu dihitung dari harga yang tampak lebih murah. Ditulis sebagai keterbatasan atau
  diatasi sebagai keputusan terpisah.
- Survivorship bias (konstituen LQ45 saat ini), estimasi mu bising, risk-free konstan.

## 9. Perubahan yang perlu dibuat pada PDF rencana (agar konsisten)

Halaman mengacu ke PDF rencana (23 halaman).

| Hal. | Bagian | Perubahan |
|---|---|---|
| 1 | Rumusan Masalah no. 2 | "kas diusahakan sekecil mungkin" -> kas boleh sengaja ditahan agar batas risiko terpenuhi; selalu dilaporkan. |
| 3 | Tabel Asumsi, baris Kas | Kas = variabel keputusan (gen g0); bukan "diminimalkan". Hapus alasan "dana harus terpakai ... 15% himpunan acak". |
| 5 | Paragraf setelah kendala | Klaim "K=5 terkunci pada bobot sama rata, K<5 tidak layak" gugur (kas bebas); ganti: cap 20% pada K<=5 membatasi dana terinvestasi paling banyak 20% x K. |
| 6 | Sumber dan cakupan | "sekitar 1.167 hari bursa (23 November 2021 ...)" -> 1.162 hari return (1.163 hari harga, 22 Nov 2021 s.d. 30 Sep 2026). |
| 7 | Pembagian data, butir 2 dan 3 | Beli pada penutupan hari terakhir jendela latih (bukan "hari pertama jendela uji"); 5 jendela, hapus "memakai 3 jendela (jendela terakhir ...)". |
| 8 | Parameter GA | Tegaskan mutasi Gaussian 1/N per gen berlaku juga untuk g0; vektor 2N+1. |
| 9-12 | Tabel solver eksak, tabel pembanding, ringkasan, hipotesis H2 | Eksak = MIQCP SCIP (optimum lot bulat dengan status/gap) untuk semua K; enumerasi himpunan + QP kontinu = batas atas dan cek silang. |
| 10 | 2. Decoder bersama | Vektor 2N+1; langkah 2 memakai g0 sebagai w_0; saham tak terjangkau dikeluarkan; n_i=0 = tidak terpilih. |
| 10-11 | GA, SA, PSO, DE | "vektor 2N" -> 2N+1; mutasi Gaussian dan perturbasi SA juga menyentuh g0; semai top-K dengan g0 rendah. |
| 11 | 4. Anggaran evaluasi | 30 run (bukan 10); rincian 100 + 199x100 dst. |
| 12 | Fase 0 | "semua nilai K dapat terinvestasi penuh" -> kas bebas; jelaskan cap 20% pada K<=5. |
| 13 | Fase 1 dan Fase 2 | 5 jendela + 5 kasus tetap = 30 masalah; 30 run x 4 = 120 run/masalah = 3.600 run (72 juta evaluasi). Hapus bullet "K<5 batas 20% tidak layak". |
| 14-15 | Keluaran, metrik, R_acuan | Tambah kolom g0 dan kas akhir; "atas 10 run" -> 30; R_acuan = optimum SCIP (bukan enumerasi); daya uji dihitung ulang (30 masalah). |
| 15 | Aturan penafsiran no. 1 | "solusi eksak (K=2,3,5)" -> solusi eksak SCIP bila terbukti (gap 0). |
| 17 | Struktur kode | `exact_miqp.py` -> `exact_scip.py`, `enum_relax.py`, `greedy.py`; tambah README, CONTRIBUTING, docs. |
| 17-18 | Problem | Tambah `n_genes = 2N+1` dan `tickers`; cap=1,0 bila K<7. |
| 18-19 | Pseudo-code decoder | Pakai x[2N] sebagai g0 (lihat bagian 5 spesifikasi). |
| 19-20 | Pseudo-code eksak | PySCIPOpt (bukan cvxpy); kas = 1 - sum(w) >= 0 dan pos. minimum n_i >= z_i. |
| 20 | Tabel tahap | Tahap 1 selesai; tahap 7 = 3.600 run. |
| 21-22 | Pengujian, risiko | Brute force K=2 dengan kas bebas; tambah uji gen kas; "tidak ada optimum terbukti pada K=7,10" -> bergantung pada gap SCIP. |
| 22 | Keterbatasan | Tambah: Close Yahoo sudah split-adjusted (lot jendela lama untuk ISAT/BRPT tampak lebih murah); kas bernilai r_f konstan. |
