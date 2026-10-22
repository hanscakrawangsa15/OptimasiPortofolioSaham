"""Fase 1: penalaan parameter tiap metaheuristik dengan protokol yang sama (spesifikasi bagian 6).

Hanya data latih jendela 1, K = 7, sigma_max = 1,5x; grid kecil per algoritma (PDF: GA populasi x laju mutasi,
SA T0 x laju pendinginan, PSO w x c, DE F x CR); 30 run per titik grid (seed 100..129, terpisah dari seed Fase 2);
konfigurasi dengan median R tertinggi dipilih (selisih <= toleransi dianggap seri -> simpangan baku terkecil)
lalu dibekukan. Jalankan:  python3 src/penalaan.py
"""
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

if __package__ is None:                                   # dijalankan sebagai skrip: pasang root dan src ke sys.path
    for _p in (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))):
        if _p not in sys.path:
            sys.path.insert(0, _p)

import config
from algorithms import de, ga, pso, sa
from data import buat_problem, muat_jendela

MODUL = {"GA": ga, "PSO": pso, "SA": sa, "DE": de}

# Parameter awal (PDF) yang harus ikut dalam grid sebagai pembanding. SA: T0 dikalibrasi dari peluang penerimaan awal
# (target_acceptance); laju pendinginan = rasio suhu akhir/awal (T_end_ratio, pendinginan geometrik).
AWAL = {
    "GA": dict(pop=100, laju_mut=1.0),
    "PSO": dict(w=0.7, c=1.5, v_maks=0.2),
    "SA": dict(target_acceptance=0.80, T_end_ratio=0.001),
    "DE": dict(F=0.5, CR=0.9),
}

# tahap = daftar (nama tahap, {parameter: nilai-nilai}); tahap berikutnya memakai nilai terbaik tahap sebelumnya.
RUANG = {
    "GA": [("A", {"pop": [50, 100, 200], "laju_mut": [0.5, 1.0, 2.0]})],
    "SA": [("A", {"target_acceptance": [0.5, 0.8, 0.95], "T_end_ratio": [1e-2, 1e-3, 1e-4]})],
    "PSO": [("A", {"w": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "c": [1.0, 1.5, 2.0]}),
            ("B", {"v_maks": [0.05, 0.1, 0.2, 0.5, 1.0]})],
    "DE": [("A", {"F": [0.3, 0.5, 0.7, 0.9], "CR": [0.5, 0.7, 0.9, 1.0]})],
}


# Perluasan grid satu kali bila nilai terbaik berada di tepi: (nilai tambahan di bawah, nilai tambahan di atas).
# Batas alami tidak diperluas (mis. w > 1 tidak meredam kecepatan, CR <= 1, peluang penerimaan < 1).
PERLUASAN = {
    "GA": {"pop": ([], [400]), "laju_mut": ([0.25], [4.0])},
    "SA": {"target_acceptance": ([0.3], [0.99]), "T_end_ratio": ([1e-5], [1e-1])},
    "PSO": {"w": ([0.2, 0.3], []), "c": ([0.5], [2.5, 3.0]), "v_maks": ([], [2.0])},
    "DE": {"F": ([0.1, 0.2], [1.1, 1.3]), "CR": ([0.1, 0.3], [])},
}


def perluas(algoritma, grid, terbaik):
    """Grid yang diperluas di sisi tepi tempat nilai terbaik berada; {} bila tidak ada tepi yang dapat diperluas."""
    baru = {}
    for k, nilai in grid.items():
        bawah, atas = PERLUASAN.get(algoritma, {}).get(k, ([], []))
        tambah = []
        if np.isclose(terbaik[k], min(nilai)):
            tambah += [v for v in bawah if v not in nilai]
        if np.isclose(terbaik[k], max(nilai)):
            tambah += [v for v in atas if v not in nilai]
        if tambah:
            baru[k] = sorted(set(nilai) | set(tambah))
    return {k: (baru[k] if k in baru else list(v)) for k, v in grid.items()} if baru else {}


def ke_kw(algoritma, titik):
    """Parameter grid -> argumen keyword run(). PSO: c = c1 = c2."""
    kw = dict(titik)
    if algoritma == "PSO" and "c" in kw:
        kw["c1"] = kw["c2"] = kw.pop("c")
    return kw


@lru_cache(maxsize=None)
def _problem(data, K, mult):
    return buat_problem(muat_jendela(data), K, mult=mult)


def _satu_run(argumen):
    """Satu run (dipanggil di proses terpisah): kembalikan metrik ringkas."""
    algoritma, titik, data, K, mult, seed, budget = argumen
    P = _problem(data, K, mult)
    r = MODUL[algoritma].run(P, budget=budget, seed=seed, **ke_kw(algoritma, dict(titik)))
    return dict(seed=seed, R=r.R, sigma=r.sigma, viol=r.viol, layak=bool(r.feasible),
                eval_terbaik=r.n_eval_best, detik=r.seconds)


def jalankan_titik(algoritma, titik_list, seeds, budget, data, K, mult, proses):
    """Jalankan semua (titik x seed). Mengembalikan {indeks_titik: [metrik per seed]} (urut seed)."""
    tugas = [(algoritma, tuple(t.items()), data, K, mult, s, budget) for t in titik_list for s in seeds]
    if proses and proses > 1:
        with ProcessPoolExecutor(max_workers=proses) as ex:
            hasil = list(ex.map(_satu_run, tugas, chunksize=max(1, len(tugas) // (proses * 8))))
    else:
        hasil = [_satu_run(t) for t in tugas]
    n = len(seeds)
    return {i: hasil[i * n:(i + 1) * n] for i in range(len(titik_list))}


def _r_efektif(baris):
    return np.array([b["R"] if b["layak"] else -1.0 for b in baris])


def ringkas_titik(baris):
    r = _r_efektif(baris)
    return dict(median=float(np.median(r)), std=float(r.std()), layak=float(np.mean([b["layak"] for b in baris])),
                maks=float(r.max()))


def pilih(tabel, toleransi=config.TUNING_TOLERANSI):
    """Baris terpilih: median R tertinggi; yang selisihnya <= toleransi dianggap seri -> std terkecil, lalu median tertinggi."""
    m = tabel["median"].max()
    seri = tabel[tabel["median"] >= m - toleransi]
    return seri.sort_values(["std", "median"], ascending=[True, False]).iloc[0]


def tala(algoritma, seeds=config.TUNING_SEEDS, budget=config.BUDGET_EVAL, ruang=None, proses=None, cetak=print):
    """Penalaan satu algoritma. Mengembalikan (DataFrame long semua run, dict parameter beku)."""
    ruang = ruang or RUANG[algoritma]
    data, K, mult = config.TUNING_DATA, config.TUNING_K, config.TUNING_MULT
    awal = AWAL[algoritma]
    semua, baris_run = [], []                                 # semua: ringkasan per titik; baris_run: seluruh run
    terbaik_sejauh = dict(awal)
    sudah = set()
    def _jalankan_tahap(tahap, grid):
        nonlocal terbaik_sejauh
        nama = list(grid)
        kombinasi = [dict(zip(nama, v)) for v in itertools.product(*grid.values())]
        titik_list = [{**terbaik_sejauh, **k} for k in kombinasi]
        titik_list = [t for t in titik_list if tuple(sorted(t.items())) not in sudah]
        t0 = time.perf_counter()
        per = jalankan_titik(algoritma, titik_list, seeds, budget, data, K, mult, proses)
        for i, t in enumerate(titik_list):
            sudah.add(tuple(sorted(t.items())))
            ring = ringkas_titik(per[i])
            semua.append(dict(tahap=tahap, **t, **ring))
            for b in per[i]:
                baris_run.append(dict(algoritma=algoritma, tahap=tahap, **t, **b))
            cetak(f"  {algoritma} {tahap} {t}  median R = {ring['median']:.4f}  std = {ring['std']:.4f}  layak = {ring['layak']:.0%}")
        cetak(f"  tahap {tahap}: {len(titik_list)} titik x {len(seeds)} run dalam {time.perf_counter() - t0:.1f} dtk")

    for tahap, grid in ruang:
        _jalankan_tahap(tahap, grid)
        tabel = pd.DataFrame([s for s in semua if s["tahap"] in (tahap, tahap + "2")])
        terbaik = {k: pilih(tabel)[k] for k in grid}
        terbaik = {k: (v.item() if hasattr(v, "item") else v) for k, v in terbaik.items()}
        besar = perluas(algoritma, grid, terbaik)
        if besar:                                             # satu kali perluasan grid (sama untuk semua algoritma)
            cetak(f"  nilai terbaik di tepi grid {terbaik}: grid tahap {tahap} diperluas")
            _jalankan_tahap(tahap + "2", besar)
            tabel = pd.DataFrame([s for s in semua if s["tahap"] in (tahap, tahap + "2")])
            terbaik = {k: pilih(tabel)[k] for k in grid}
            terbaik = {k: (v.item() if hasattr(v, "item") else v) for k, v in terbaik.items()}
        terbaik_sejauh = {**terbaik_sejauh, **terbaik}
    tabel_semua = pd.DataFrame(semua)
    parameter_kolom = list(awal)
    final = pilih(tabel_semua)
    pilihan = {k: (final[k].item() if hasattr(final[k], "item") else final[k]) for k in parameter_kolom}
    # pembanding: parameter awal PDF pada seed penalaan yang sama (selalu ada dalam grid)
    df_run = pd.DataFrame(baris_run)
    def _sama(df, p):
        m = np.ones(len(df), bool)
        for k, v in p.items():
            m &= np.isclose(df[k].astype(float), float(v))
        return m
    r_pilih = _r_efektif(df_run[_sama(df_run, pilihan)].to_dict("records"))
    r_awal = _r_efektif(df_run[_sama(df_run, awal)].to_dict("records"))
    p_mw = float(mannwhitneyu(r_pilih, r_awal, alternative="two-sided")[1]) if len(r_awal) and r_pilih.std() + r_awal.std() > 0 else 1.0
    tepi = []
    for _, g in ruang:
        for k, nilai in g.items():
            semua_nilai = sorted(set(tabel_semua[k]))
            if k in pilihan and (np.isclose(pilihan[k], semua_nilai[0]) or np.isclose(pilihan[k], semua_nilai[-1])) and len(semua_nilai) > 1:
                tepi.append(k)
    beku = dict(parameter=ke_kw(algoritma, pilihan), grid_parameter=pilihan, parameter_awal=ke_kw(algoritma, dict(awal)),
                median_R_penalaan=float(np.median(r_pilih)), std_R_penalaan=float(r_pilih.std()),
                median_R_awal=float(np.median(r_awal)), p_mannwhitney_vs_awal=p_mw,
                di_tepi_grid=sorted(set(tepi)), titik_diuji=int(len(tabel_semua)), run_per_titik=len(seeds),
                data_penalaan=data, K=K, pengali_risiko=mult, anggaran=budget,
                seed_penalaan=[int(min(seeds)), int(max(seeds))],
                aturan=f"median R tertinggi; selisih <= {config.TUNING_TOLERANSI} -> std terkecil")
    return df_run, beku


def main(algoritma=None, proses=None):
    algoritma = algoritma or list(RUANG)
    os.makedirs(os.path.join(config.HASIL_DIR, "csv"), exist_ok=True)
    from parameter import BERKAS_BEKU, muat_beku
    beku = muat_beku()
    for a in algoritma:
        print(f"== Fase 1 {a}: {config.TUNING_DATA}, K = {config.TUNING_K}, {len(config.TUNING_SEEDS)} run per titik")
        df, b = tala(a, proses=proses)
        df.to_csv(os.path.join(config.HASIL_DIR, "csv", f"fase1_{a.lower()}_grid.csv"), index=False)
        beku[a] = b
        with open(BERKAS_BEKU, "w", encoding="utf-8") as f:
            json.dump(beku, f, indent=2, ensure_ascii=False)
        print(f"  TERPILIH {a}: {b['parameter']}  median R {b['median_R_penalaan']:.4f} (awal {b['median_R_awal']:.4f}, p = {b['p_mannwhitney_vs_awal']:.3g})")
    return beku


if __name__ == "__main__":
    main(proses=os.cpu_count())
