"""Fase 2: GA, PSO, SA, DE pada 30 masalah (6 data x K) x 30 seed dengan parameter beku Fase 1.

Seed sama untuk semua algoritma (0..29), anggaran 20.000 evaluasi. Keluaran CSV long:
hasil/csv/fase2_hasil.csv (in-sample dan out-of-sample). Jalankan:  python3 src/eksperimen.py
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

if __package__ is None:                                   # dijalankan sebagai skrip
    for _p in (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))):
        if _p not in sys.path:
            sys.path.insert(0, _p)

import config
from data import NAMA_DATA, buat_problem, muat_jendela
from evaluasi import komponen_akhir, metrik_uji, nilai_uji
from parameter import kw_terpilih
from penalaan import MODUL

ALGORITMA = ("GA", "PSO", "SA", "DE")
BERKAS = os.path.join(config.HASIL_DIR, "csv", "fase2_hasil.csv")


def daftar_masalah(data=NAMA_DATA, ks=config.K_LIST):
    """30 masalah: 5 jendela walk-forward + kasus tetap, masing-masing K dalam K_LIST."""
    return [(d, K) for d in data for K in ks]


@lru_cache(maxsize=None)
def _muat(data):
    return muat_jendela(data)


def _satu_run(argumen):
    algoritma, data, K, seed, budget, kw = argumen
    w = _muat(data)
    P = buat_problem(w, K, mult=config.TUNING_MULT)
    r = MODUL[algoritma].run(P, budget=budget, seed=seed, **kw)
    r_uji = w["return_uji"]
    nilai = nilai_uji(P, r.lots, r_uji)
    m = metrik_uji(nilai, P.D)
    saham_akhir, kas_akhir = komponen_akhir(P, r.lots, r_uji)
    pos = r.lots * P.harga_lot
    return dict(algoritma=algoritma, data=data, K=K, seed=seed, sigma_max=P.sigma_max,
                R=r.R, sigma=r.sigma, viol=r.viol, layak=bool(r.feasible), n_eval=r.n_eval,
                eval_terbaik=r.n_eval_best, detik=r.seconds, jumlah_saham=int((r.lots > 0).sum()),
                kas=float((P.D - pos.sum()) / P.D), g0=float(np.clip(r.x[-1], 0.0, 1.0)),
                lots=";".join(map(str, r.lots)),
                return_uji_total=m["total"], return_uji_tahunan=m["setara_tahun"], vol_uji=m["vol"], mdd_uji=m["mdd"],
                sharpe_uji=m["sharpe"], cvar95_uji=m["cvar95"], shrinkage=m["setara_tahun"] - r.R,
                nilai_akhir_uji=float(nilai[-1]), saham_akhir_uji=saham_akhir, kas_akhir_uji=kas_akhir,
                risiko_uji_memenuhi=bool(m["vol"] <= P.sigma_max))


def jalankan(algoritma=ALGORITMA, masalah=None, seeds=config.SEEDS_FASE2, budget=config.BUDGET_EVAL,
             proses=None, kw=None, cetak=print):
    """Jalankan (algoritma x masalah x seed). kw = {algoritma: parameter}; bawaan: parameter beku Fase 1."""
    masalah = masalah or daftar_masalah()
    kw = kw or {a: kw_terpilih(a) for a in algoritma}
    tugas = [(a, d, K, s, budget, kw[a]) for (d, K) in masalah for a in algoritma for s in seeds]
    t0 = time.perf_counter()
    if proses and proses > 1:
        with ProcessPoolExecutor(max_workers=proses) as ex:
            hasil = list(ex.map(_satu_run, tugas, chunksize=max(1, len(tugas) // (proses * 16))))
    else:
        hasil = [_satu_run(t) for t in tugas]
    cetak(f"{len(hasil)} run selesai dalam {time.perf_counter() - t0:.1f} dtk")
    return pd.DataFrame(hasil)


def main(proses=None):
    kw = {a: kw_terpilih(a) for a in ALGORITMA}
    if not any(kw.values()):
        print("PERINGATAN: parameter beku belum ada; memakai parameter awal PDF (jalankan src/penalaan.py dulu).")
    print("Parameter:", kw)
    df = jalankan(proses=proses, kw=kw)
    os.makedirs(os.path.dirname(BERKAS), exist_ok=True)
    df.to_csv(BERKAS, index=False)
    print("Tersimpan:", BERKAS, df.shape)
    return df


if __name__ == "__main__":
    main(proses=os.cpu_count())
