# Optimasi Alokasi Portofolio Saham LQ45 dengan Genetic Algorithm (Maksimasi Sharpe Ratio)

## Tujuan
Mencari bobot portofolio saham LQ45 yang memaksimalkan Sharpe ratio memakai Genetic Algorithm (GA) real-coded, lalu membandingkannya dengan baseline (QP/SLSQP max-Sharpe dan 1/N) pada data uji out-of-sample. Keluaran akhir: tabel metrik dan enam visual untuk laporan.

## Keputusan yang sudah disepakati
- Bobot long-only, jumlah 1, **batas atas per saham `MAX_WEIGHT` = 10%** (parameter di `config.py`, bisa 15%). Batas yang sama dipakai GA, QP, dan 1/N.
- **Universe diperluas ke konstituen LQ45** (target sekitar 30-45 saham). Versi awal 10 ticker tidak dipakai karena cap 10% pada 10 saham memaksa bobot sama rata (identik dengan 1/N). Ticker yang datanya kurang dibuang dan dilaporkan.
- Penanganan batas di GA: **operator repair** (clip ke [0, cap], normalisasi, ulangi sampai valid), bukan penalti atau decoder.
- Urutan kerja: baseline dulu, baru GA.

## Aliran data
```
config.py -> scrappingdata.py -> split latih/uji -> mu, Sigma (latih saja)
                                          |-> baseline.py (SLSQP max-Sharpe, 1/N)
                                          |-> GeneticAlgorithm.py (30 run, seed 0..29)
                                                   |
                  evaluasi.py (bobot dari latih, dinilai di uji) -> visualisasi.py -> hasil/
main.py menjalankan semuanya berurutan.
```

## Komponen
Struktur memakai nama file yang sudah ada di `src/`.

| File | Tugas | Bergantung pada |
|---|---|---|
| `config.py` (baru) | Ticker, tanggal, `MAX_WEIGHT`, risk-free, parameter GA, seed | - |
| `src/scrappingdata.py` | Unduh adj. close (yfinance, `.JK`), cek data kosong, buang ticker bermasalah, log return, split, hitung mu/Sigma tahunan; simpan `harga_adjclose.csv`, `log_return.csv`, `mu_cov_train.npz` | config |
| `src/baseline.py` | Bobot SLSQP max-Sharpe (long-only, sum=1, <= cap) dan 1/N | mu, Sigma, config |
| `src/GeneticAlgorithm.py` | Kromosom, fitness, repair, seleksi, SBX, mutasi, elitisme, satu run GA, 30 run | mu, Sigma, config |
| `src/evaluasi.py` | Metrik uji: return tahunan, volatilitas, Sharpe, max drawdown; nilai kumulatif | log return uji, bobot |
| `src/visualisasi.py` | Enam gambar PNG dan tabel CSV ke `hasil/` | keluaran evaluasi/GA |
| `main.py` | Menjalankan pipeline | semua |

## GA
- Kromosom: vektor N bobot real. Populasi awal dari Dirichlet lalu di-repair.
- Fitness: `Sharpe = (w.mu - rf) / sqrt(w' Sigma w)` dengan mu dan Sigma tahunan dari data latih saja.
- Seleksi turnamen, crossover SBX, mutasi polinomial/Gaussian, elitisme, repair setelah variasi.
- Default: populasi 200, 300 generasi, rf 5%/tahun (semua di config).
- Setiap run menyimpan kurva konvergensi (terbaik dan rata-rata per generasi) dan bobot terbaik.
- Bobot GA terbaik dipilih dari run dengan fitness latih tertinggi, tidak pernah dari performa uji.
- RNG: `numpy.random.default_rng(seed)`, hasil reprodusibel.

## Baseline dan evaluasi
- SLSQP max-Sharpe dengan batasan identik dengan GA, dipakai sebagai acuan optimum latih.
- 1/N sebagai pembanding naif (valid selama N >= 1/MAX_WEIGHT).
- Evaluasi di data uji untuk GA (terbaik dan median 30 run), SLSQP, 1/N. Bobot selalu dari data latih.

## Enam visual (hasil/)
1. Kurva konvergensi GA (pita 30 run).
2. Sebaran Sharpe latih 30 run (boxplot) dengan garis SLSQP.
3. Bobot GA vs SLSQP per saham.
4. Nilai kumulatif data uji: GA, SLSQP, 1/N.
5. Perbandingan metrik uji (bar).
6. Drawdown data uji.

## Penanganan galat dan pengujian
- Ticker gagal unduh atau data kosong > 10% dibuang dan dicatat. Program berhenti dengan pesan jelas bila saham tersisa < ceil(1/MAX_WEIGHT).
- Tes unit: repair (jumlah 1, semua dalam [0, cap]), fungsi Sharpe, batasan hasil SLSQP dan 1/N, tidak ada kebocoran data uji ke mu/Sigma, reprodusibilitas seed.

## Di luar cakupan
Short selling, batas jumlah saham (cardinality), biaya transaksi, rebalancing berkala.
