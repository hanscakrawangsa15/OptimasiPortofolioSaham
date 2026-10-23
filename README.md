# Optimasi Alokasi Dana Saham LQ45 dengan Batas Risiko dan Kardinalitas

Memaksimalkan return ekspektasi portofolio 42 saham LQ45 (modal Rp 5 juta) di bawah batas risiko σ_max = 1,5 × volatilitas
portofolio 1/N, dengan batas jumlah saham K (2, 3, 5, 7, 10), batas 20% per saham untuk K ≥ 7, pembelian dalam lot
(100 lembar), dan kas yang bernilai r_f = 5% per tahun. Empat metaheuristik, yaitu **Genetic Algorithm (GA)**,
**Particle Swarm Optimization (PSO)**, **Simulated Annealing (SA)**, dan **Differential Evolution (DE)**, dibandingkan pada
dekoder, fungsi evaluasi, dan anggaran yang sama (20.000 evaluasi per run), in-sample dan out-of-sample (walk-forward).
Untuk K = 2, hasilnya dibandingkan dengan optimum sejati dari brute force.

Rencana penelitian dan makalah rujukan ada di [docs/rujukan/](docs/rujukan/).

## Mulai cepat

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest tests -q          # semua tes harus hijau
```

Data sudah ada di `data/`. Mengunduh ulang dari Yahoo Finance hanya dilakukan satu orang
(lihat [CONTRIBUTING.md](CONTRIBUTING.md)):

```bash
python3 src/scrappingdata.py
```

## Menjalankan GUI

```bash
streamlit run app.py          # jalankan dari folder utama repo
```

Di sisi kiri pilih data (kasus tetap atau jendela 1–5), K, modal, batas risiko, algoritma (GA, PSO, SA, DE), jumlah seed,
dan jenis parameter (terpilih hasil Fase 1 atau awal dari rencana), lalu klik **Jalankan**.

| Tab | Isi |
|---|---|
| Ringkasan | Return per tahun, nilai akhir, sisa kas, risiko terhadap batas |
| Portofolio | Saham, lot, bobot, dan rincian sisa kas |
| Proses | Kurva konvergensi, sebaran risiko–return, perilaku populasi/pendinginan |
| Perbandingan | Statistik antar-seed, boxplot, uji Mann–Whitney |
| Data uji | Nilai portofolio beli-dan-tahan pada periode uji |
| Profil risiko | Konvergensi untuk profil Konservatif (1,5×), Moderat (2,0×), Agresif (2,5×) |
| Hasil penelitian | Hasil penalaan Fase 1, hasil SA 30 seed, dan hasil notebook DE |
| Fase 2 | Peringkat, uji Wilcoxon–Holm, dan hasil out-of-sample keempat algoritma |
| Tentang | Ringkasan masalah dan parameter |

## Eksperimen (Fase 1 dan Fase 2)

Keempat metaheuristik memakai dekoder, `Evaluator`, aturan Deb, anggaran 20.000 evaluasi, dan semai populasi awal yang
sama (10% populasi disemai top-K dengan g₀ rendah; SA memulai dari satu solusi tersemai).

```bash
python3 src/penalaan.py      # Fase 1: tala tiap algoritma (jendela 1, K=7, 30 run per titik), bekukan ke hasil/csv/fase1_parameter_beku.json
python3 src/eksperimen.py    # Fase 2: 4 algoritma x 30 masalah x 30 seed = 3.600 run -> hasil/csv/fase2_hasil.csv
```

- **Fase 1** (penalaan): GA populasi × laju mutasi, SA peluang penerimaan awal (T₀) × laju pendinginan, PSO w × c lalu v_max,
  DE F × CR. Grid diperluas satu kali bila nilai terbaik berada di tepi. Seed penalaan 100–129, terpisah dari seed Fase 2.
- **Fase 2** (perbandingan): 30 masalah (5 jendela walk-forward + kasus tetap, masing-masing 5 nilai K), seed 0–29 sama untuk
  semua algoritma, parameter yang sudah dibekukan.

## Struktur folder

```
.
├── app.py                     GUI Streamlit
├── config.py                  Parameter global (K, sigma, modal, lot, jendela, anggaran, seed, penalaan)
├── requirements.txt, pytest.ini
├── src/
│   ├── scrappingdata.py       Unduh data, bersihkan, hitung return sederhana
│   ├── walkforward.py         Pembentukan jendela walk-forward dan kasus tetap
│   ├── laporan_data.py        Pemeriksaan anomali data -> data/laporan_data.md
│   ├── data.py                Muat jendela data dan bentuk Problem
│   ├── problem.py             Problem dan Result
│   ├── decoder.py             Kromosom 2N+1 -> lot bulat yang layak; populasi awal tersemai
│   ├── fitness.py             Evaluator (pintu tunggal evaluasi) dan aturan Deb
│   ├── evaluasi.py            Rincian kas, penguraian return/risiko, metrik data uji
│   ├── runner.py              Jalankan banyak seed, ringkas, bandingkan, profil risiko
│   ├── penalaan.py            Fase 1: penalaan parameter keempat algoritma
│   ├── eksperimen.py          Fase 2: semua algoritma x 30 masalah x 30 seed
│   ├── parameter.py           Pembaca parameter beku Fase 1
│   ├── stats.py               Peringkat, Wilcoxon berpasangan + Holm, ringkasan latih
│   ├── visualisasi.py         Grafik untuk GUI
│   └── algorithms/
│       ├── ga.py  pso.py  sa.py  de.py     Empat metaheuristik: run(problem, budget, seed, trace, **parameter)
│       └── brute_force_k2.py               Optimum sejati untuk K <= 2
├── notebooks/
│   ├── GA_Portofolio_LQ45 fix.ipynb        GA: dekoder, fitness, hasil per K, data uji
│   ├── PSO_Portofolio_LQ45.ipynb           PSO dan perbandingannya dengan GA
│   ├── PSO_Fase1_Penalaan_LQ45.ipynb       Penalaan PSO awal (sebelum semai disamakan)
│   ├── SA_Portofolio_LQ45.ipynb            SA: kalibrasi suhu, konvergensi, 30 seed
│   └── DE_Portofolio_LQ45.ipynb            DE: kasus tetap dan walk-forward jendela 1–3
├── tests/                     Tes pytest untuk data, dekoder, fitness, algoritma, evaluasi, penalaan, eksperimen, dan GUI
├── data/
│   ├── harga_close_raw.csv, harga_adjclose.csv, return_sederhana.csv, stock_splits.csv, ticker_dibuang.txt, metadata.json
│   ├── laporan_data.md        Laporan kualitas data
│   └── windows/               jendela_1..5.npz dan kasus_tetap.npz (statistik latih, harga beli, data uji)
├── hasil/
│   ├── csv/                   Keluaran eksperimen: fase1_*.csv, fase1_parameter_beku.json, fase2_hasil.csv, de_*.csv, pso_*.csv
│   ├── hasil_sa_30_seed.csv   Hasil notebook SA
│   └── gambar/                Visual untuk laporan
└── docs/
    ├── rujukan/               PDF rencana dan makalah sumber
    └── revisi_bab3_laporan.md Catatan penyesuaian Bab 3 laporan terhadap hasil eksperimen
```

## Cara kerja singkat

- **Kromosom** berpanjang 2N+1 = 85 gen pada [0,1]: 42 kunci prioritas (memilih K saham), 42 bobot mentah, dan satu gen kas g₀.
- **Dekoder** deterministik mengubah kromosom menjadi lot bulat. Kardinalitas, batas bobot, lot, dan batas dana terpenuhi
  oleh konstruksi; hanya batas risiko yang dapat dilanggar dan ditangani dengan aturan kelayakan Deb (tanpa penalti).
- **`Evaluator`** adalah satu-satunya jalan menghitung fitness. Ia menghitung evaluasi, menolak melewati anggaran, dan mencatat
  riwayat konvergensi tiap 200 evaluasi.
- **Data uji** tidak pernah masuk ke `Problem`; hanya dipakai untuk menilai portofolio beli-dan-tahan setelah optimasi.

## Status

| Bagian | Status |
|---|---|
| Data, jendela walk-forward, laporan data | Selesai |
| Dekoder, `Evaluator`, aturan Deb | Selesai |
| GA, PSO, SA, DE | Selesai, perlakuan disamakan (semai, anggaran, seed) |
| Fase 1 (penalaan) dan Fase 2 (3.600 run) | Selesai |
| GUI Streamlit | Selesai |
| Acuan eksak untuk K ≥ 3 (SCIP) dan heuristik pembanding sederhana | Belum dikerjakan |

Panduan kerja bersama (cabang, commit, aturan kode) ada di [CONTRIBUTING.md](CONTRIBUTING.md).
