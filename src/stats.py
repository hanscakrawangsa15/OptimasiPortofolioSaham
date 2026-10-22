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
