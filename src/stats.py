"""Statistik antar algoritma pada Fase 2: peringkat per masalah, Wilcoxon berpasangan + koreksi Holm."""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def median_per_masalah(df, kolom="R"):
    """Tabel (data, K) x algoritma berisi median kolom antar seed; run tak layak dihitung terburuk untuk R."""
    d = df.assign(_v=np.where(df["layak"], df[kolom], -1.0) if kolom == "R" else df[kolom])
    return d.groupby(["data", "K", "algoritma"])["_v"].median().unstack("algoritma")


def peringkat_rata(tabel):
    """Peringkat rata-rata tiap algoritma (1 = terbaik, seri = rata-rata) atas semua masalah."""
    return tabel.rank(axis=1, ascending=False, method="average").mean().sort_values()


def wilcoxon_holm(tabel, alpha=0.05):
    """Wilcoxon berpasangan tiap pasangan algoritma atas masalah (median R); p disesuaikan Holm."""
    baris = []
    for a, b in itertools.combinations(tabel.columns, 2):
        selisih = (tabel[a] - tabel[b]).to_numpy()
        p = 1.0 if np.allclose(selisih, 0.0) else float(wilcoxon(tabel[a], tabel[b], zero_method="wilcox")[1])
        baris.append(dict(A=a, B=b, median_selisih=float(np.median(selisih)), menang_A=int((selisih > 1e-12).sum()),
                          menang_B=int((selisih < -1e-12).sum()), seri=int((np.abs(selisih) <= 1e-12).sum()), p=p))
    hasil = pd.DataFrame(baris).sort_values("p").reset_index(drop=True)
    m = len(hasil)
    holm = np.maximum.accumulate([min(1.0, (m - i) * p) for i, p in enumerate(hasil["p"])])
    hasil["p_holm"] = holm
    hasil["signifikan"] = hasil["p_holm"] < alpha
    return hasil


def acuan_masalah(df, brute=True, brute_force=None):
    """R acuan per (data, K): optimum brute force untuk K = 2; selain itu R terbaik yang pernah ditemukan semua run."""
    ac = df.groupby(["data", "K"])["R"].max()
    if brute_force is None and brute:
        from algorithms.brute_force_k2 import brute_force_k2
        from data import buat_problem, muat_jendela
        brute_force = {(d, 2): brute_force_k2(buat_problem(muat_jendela(d), 2))["kas bebas"][0]
                       for d in df.loc[df["K"] == 2, "data"].unique()}
    for kunci, nilai in (brute_force or {}).items():
        if kunci in ac.index:
            ac.loc[kunci] = nilai
    return ac


def ringkasan_latih(df, acuan, toleransi=0.01):
    """Metrik data latih per (data, K, algoritma): R (mean, median, std, min, max), kelayakan, selisih ke R acuan,
    tingkat keberhasilan (R >= (1 - toleransi) x R acuan), evaluasi hingga terbaik dan waktu (median)."""
    d = df.assign(Reff=df["R"].where(df["layak"], -1.0))
    d["acuan"] = [acuan.loc[(a, b)] for a, b in zip(d["data"], d["K"])]
    d["ok"] = d["layak"] & (d["R"] >= (1.0 - toleransi) * d["acuan"])
    g = d.groupby(["data", "K", "algoritma"])
    hasil = g.agg(mean=("Reff", "mean"), median=("Reff", "median"), std=("Reff", "std"), min=("Reff", "min"),
                  max=("Reff", "max"), layak=("layak", "mean"), acuan=("acuan", "first"), sukses=("ok", "mean"),
                  eval_terbaik=("eval_terbaik", "median"), detik=("detik", "median")).reset_index()
    hasil["selisih"] = hasil["acuan"] - hasil["median"]
    return hasil
