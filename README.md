# Optimasi Alokasi Dana Saham LQ45 dengan Batas Risiko dan Kardinalitas

Memaksimalkan return ekspektasi portofolio 42 saham LQ45 di bawah batas risiko, dengan batas jumlah saham
K (2, 3, 5, 7, 10), batas 20% per saham untuk K >= 7, dan pembelian dalam lot (100 lembar). GA, SA, PSO, dan DE
dibandingkan dengan acuan eksak (SCIP) dan heuristik sederhana, in-sample dan out-of-sample (walk-forward).

Spesifikasi lengkap: [docs/superpowers/specs/2026-10-14-optimasi-lq45-metaheuristik-design.md](docs/superpowers/specs/2026-10-14-optimasi-lq45-metaheuristik-design.md)
Rencana penelitian dan makalah rujukan: [docs/rujukan/](docs/rujukan/)

## Mulai cepat

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 src/scrappingdata.py        # unduh data (hanya satu orang; lihat CONTRIBUTING.md)
python3 -m pytest tests -q          # semua tes harus hijau
```

## Menjalankan GUI

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pilih data, K, modal, batas risiko, algoritma (GA/PSO), dan jumlah seed di sisi kiri, lalu klik **Jalankan**.

## Eksperimen (Fase 1 dan Fase 2)

Keempat metaheuristik (GA, PSO, SA, DE) memakai dekoder, `Evaluator`, aturan Deb, anggaran 20.000 evaluasi, dan
semai populasi awal yang sama (10% populasi disemai top-K dengan g0 rendah; SA memulai dari satu individu semai).

```bash
python3 src/penalaan.py      # Fase 1: tala tiap algoritma (jendela_1, K=7, 30 run/titik), bekukan ke hasil/csv/fase1_parameter_beku.json
python3 src/eksperimen.py    # Fase 2: 4 algoritma x 30 masalah x 30 seed -> hasil/csv/fase2_hasil.csv
```

Grid Fase 1: GA populasi x laju mutasi, SA peluang penerimaan awal (T0) x rasio suhu akhir (laju pendinginan),
PSO w x c lalu v_maks, DE F x CR; grid diperluas satu kali bila nilai terbaik berada di tepi.

## Struktur folder

```
.
├── config.py                  Parameter global (K, sigma, D, lot, jendela, anggaran, seed)
├── app.py                     GUI Streamlit untuk hasil GA dan PSO
├── requirements.txt, pytest.ini
├── src/
│   ├── scrappingdata.py       Unduh data, bersihkan, return sederhana, jendela
│   ├── laporan_data.py        Anomali data + data/laporan_data.md
│   ├── walkforward.py         Jendela walk-forward
│   ├── data.py                Muat jendela data dan bentuk Problem
│   ├── problem.py             Problem dan Result
│   ├── decoder.py             Kunci + bobot mentah -> lot layak
│   ├── fitness.py             Evaluator + aturan Deb
│   ├── evaluasi.py            Rincian kas, analisis, uji pada data uji
│   ├── runner.py              Jalankan banyak seed, ringkas, bandingkan
│   ├── penalaan.py            Fase 1: penalaan parameter semua algoritma
│   ├── eksperimen.py          Fase 2: semua algoritma x 30 masalah x 30 seed
│   ├── stats.py               Peringkat, Wilcoxon berpasangan + Holm
│   ├── parameter.py           Pembaca parameter beku Fase 1
│   ├── visualisasi.py         Grafik untuk GUI
│   └── algorithms/            ga.py, sa.py, pso.py, de.py, brute_force_k2.py
├── notebooks/                 Notebook eksperimen: GA_Portofolio_LQ45.ipynb (dekoder, fitness, GA)
│                              dan PSO_Portofolio_LQ45.ipynb (PSO, perbandingan PSO vs GA)
│                              dan PSO_Fase1_Penalaan_LQ45.ipynb (Fase 1: penalaan PSO, parameter beku)
├── tests/                     Satu berkas test_<modul>.py per modul
├── data/                      Harga raw/adj, return sederhana, windows/, laporan_data.md
├── hasil/
│   ├── csv/                   Keluaran eksperimen (CSV long)
│   └── gambar/                Visual untuk laporan
└── docs/
    ├── rujukan/               PDF rencana dan makalah sumber
    └── superpowers/           specs/ dan plans/
```

"Stub" = berkas berisi docstring dan tanda tangan fungsi (`NotImplementedError`), supaya antarmuka jelas
sebelum diisi.

## Pembagian pekerjaan (usulan, isi kolom PIC)

| Sub-proyek | Isi | Syarat | PIC |
|---|---|---|---|
| A. Fondasi | `config`, `data`, `problem`, `decoder`, `fitness` | Data selesai (ada) | |
| B. Acuan | `exact_scip`, `enum_relax`, `greedy`, `topk_qp`, `random_equal` | Antarmuka A ter-merge | |
| C. Metaheuristik | `ga`, `sa`, `pso`, `de` | Antarmuka A ter-merge | |
| D. Eksperimen dan analisis | Fase 1 dan 2, `evaluasi`, `stats`, `visualisasi`, `main` | B dan C | |

A adalah prasyarat bagi semuanya. B dan C bisa berjalan paralel setelah `Problem`, `Evaluator`, dan dekoder
ter-merge; C dapat dibagi dua orang (GA + SA, PSO + DE).
