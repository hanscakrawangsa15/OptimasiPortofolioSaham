# Optimasi Alokasi Dana Saham LQ45 dengan Batas Risiko dan Kardinalitas

Menggantikan `2026-10-07-ga-portfolio-lq45-design.md` (maksimasi Sharpe kontinu).

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
  terakhir jendela latih). Bobot w_i = v_i / D. Kas w_0 = 1 - sum w_i, return kas = r_f.
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
- **D. Eksperimen dan analisis:** Fase 1 penalaan (jendela 1, K=7, 1,5x; 10 run per titik grid), Fase 2
  (CSV long), `evaluasi`, `stats`, `visualisasi`, `main`.

Berkas lama (`portofolio.py`, `baseline.py`, `GeneticAlgorithm.py`, `walkforward.py`, `main.py` lama, dan tes
mereka) dihapus hanya setelah penggantinya ada dan tesnya lulus (`portofolio` di A, `baseline` di B, GA di C,
`main` di D). `walkforward.py` dilebur ke `data.py` di A.

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
  `cap = 0.20` bila K >= 7, selain itu 1,0. Properti `harga_lot = lot*price` dan `layak_beli`
  (`harga_lot <= D*cap`).
- `weights(lots)`, `cash(lots)`, `evaluate(lots) -> (R, sigma, viol)` (juga untuk batch baris).
  `R = w@mu + cash*rf`, `sigma = sqrt(w@cov@w)`, `viol = max(0, sigma - sigma_max)`.
- `Result`: `lots, R, sigma, viol, feasible, history (n_eval, best_R_layak tiap 200), n_eval, seconds, seed`
  dan `status` (opsional, untuk solver eksak). Antarmuka algoritma: `run(problem, budget, seed) -> Result`.

**decoder.py.** Vektor x berpanjang 2N (N kunci prioritas, N bobot mentah).
1. Hanya saham `layak_beli` yang boleh dipilih; ambil paling banyak K dengan kunci terbesar.
2. Bobot mentah dipotong ke [0,1], dinormalisasi di himpunan terpilih (semua nol -> sama rata).
3. Bila cap < 1: potong yang melebihi cap, bagikan kelebihan ke yang belum penuh, ulangi sampai stabil;
   sisa yang tidak bisa dibagi menjadi kas.
4. `n_i = floor(w_i * D / (lot * P_i))`. Saham dengan n_i = 0 tidak terpilih. Sisa dana = kas.
5. `decode(x, P)` dan versi vektor `decode_batch(X, P)` yang konsisten bit-per-bit.
Keluaran selalu memenuhi: jumlah saham <= K, bobot <= cap, lot bulat, dana terpakai <= D. Hanya batas risiko
yang bisa dilanggar.

**fitness.py.**
- `Evaluator(problem, budget)`: pintu tunggal evaluasi untuk semua metaheuristik. `evaluate_batch(X)`
  mendekode, menghitung (R, sigma, viol), menambah hitungan evaluasi (satu vektor = satu evaluasi), menolak
  melewati anggaran, melacak solusi terbaik menurut Deb, dan mencatat riwayat tiap `HISTORY_STEP`.
- Aturan Deb: layak (viol == 0) mengalahkan tak layak; sesama layak, R lebih besar menang; sesama tak
  layak, viol lebih kecil menang. Fungsi: `deb_better(a, b)`, `deb_key` (untuk pengurutan), `deb_best_index`.

**Pengujian A.**
- Dekoder: 10.000 vektor acak per K pada jendela nyata -> saham <= K, bobot <= cap, lot bulat, dana <= D,
  saham tak terjangkau tidak pernah terpilih; `decode_batch` sama dengan `decode`; deterministik.
- Kasus K=1 (hanya uji unit): satu saham, dana habis sampai sisa kurang dari satu lot.
- `Problem.evaluate` cocok dengan hitungan tangan pada data sintetis.
- Deb: urutan layak/tak layak/campuran, dan konsistensi `deb_key` dengan `deb_better`.
- `Evaluator`: hitungan evaluasi benar, error bila melewati anggaran, riwayat tiap 200, terbaik monoton.
- `muat_skenario`: 30 skenario (25 + 5), `Problem` tidak memuat return uji, sigma_max = 1,5 x vol 1/N.

## 6. Eksperimen (gambaran untuk sub-proyek C dan D)

Fase 0 kelayakan (K=5 juga dijalankan dengan cap 20% sebagai kontrol). Fase 1 penalaan hanya di data latih
jendela 1 (K=7, 1,5x), grid kecil per algoritma, 10 run per titik, konfigurasi dibekukan. Fase 2: 30 masalah
x 4 metaheuristik x 30 run = 3.600 run x 20.000 evaluasi, ditambah acuan deterministik dan acak valid
(1.000 kali). Keluaran CSV long: R/sigma in-sample, kelayakan, himpunan, bobot, lot, kas, evaluasi sampai
terbaik, waktu, dan R/sigma/Sharpe/MDD/CVaR95 out-of-sample, shrinkage, realisasi risiko.
Parameter algoritma awal sesuai tabel PDF (GA pop 100 x 200 gen, SBX 0,9 eta 15; SA 20.000 iterasi; PSO 40 x
500; DE 100 x 200, F 0,5, CR 0,9). Aturan penafsiran ditetapkan sebelum melihat hasil (PDF bagian 4).

## 7. Di luar cakupan

Short selling, leverage, biaya transaksi, rebalancing, dividen sebagai arus kas, IWO, NSGA-II,
estimator shrinkage (ekstensi lanjutan).

## 8. Catatan dan keterbatasan yang diwarisi

- Close Yahoo sudah disesuaikan untuk stock split; "harga raw" hanya tanpa penyesuaian dividen. Untuk saham
  yang split sesudah tanggal beli sebuah jendela (ISAT 4:1 pada 2024-10-14 di jendela 1-2; BRPT sangat
  kecil), lot jendela itu dihitung dari harga yang tampak lebih murah. Ditulis sebagai keterbatasan atau
  diatasi sebagai keputusan terpisah.
- Survivorship bias (konstituen LQ45 saat ini), estimasi mu bising, risk-free konstan.
