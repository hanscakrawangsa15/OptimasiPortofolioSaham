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

## Struktur folder

```
.
├── config.py                  Parameter global (K, sigma, D, lot, jendela, anggaran, seed)
├── main.py                    Orkestrasi eksperimen                                [D, stub]
├── requirements.txt, pytest.ini
├── src/
│   ├── scrappingdata.py       Unduh data, bersihkan, return sederhana, jendela     [ADA]
│   ├── laporan_data.py        Anomali data + data/laporan_data.md                  [ADA]
│   ├── walkforward.py         Jendela walk-forward                [ADA, dilebur ke data.py di A]
│   ├── data.py                Muat data dan skenario                               [A, stub]
│   ├── problem.py             Problem dan Result                                   [A, stub]
│   ├── decoder.py             Kunci + bobot mentah -> lot layak                    [A, stub]
│   ├── fitness.py             Evaluator + aturan Deb                               [A, stub]
│   ├── algorithms/
│   │   ├── exact_scip.py      Acuan eksak MIQCP (SCIP)                             [B, stub]
│   │   ├── enum_relax.py      Enumerasi himpunan + QP kontinu (batas atas)         [B, stub]
│   │   ├── greedy.py          Greedy forward selection                             [B, stub]
│   │   ├── topk_qp.py         Top-K + QP                                           [B, stub]
│   │   ├── random_equal.py    Acak valid, bobot sama                               [B, stub]
│   │   └── ga.py  sa.py  pso.py  de.py                                             [C, stub]
│   ├── evaluasi.py            Metrik in/out-of-sample                              [D, stub]
│   ├── stats.py               Wilcoxon + Holm, deskriptif                          [D, stub]
│   ├── visualisasi.py         Gambar                                               [D, stub]
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
