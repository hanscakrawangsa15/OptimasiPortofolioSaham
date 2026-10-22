"""Menjalankan beberapa seed, meringkas, dan membandingkan dua kelompok run."""
import numpy as np
from scipy.stats import mannwhitneyu

from algorithms import ga, pso, sa, de
from fitness import deb_better

METODE = {
    "DE": lambda P, seed, trace: de.run(P, seed=seed, trace=trace),
    "GA": lambda P, seed, trace: ga.run(P, seed=seed, trace=trace),
    "PSO awal": lambda P, seed, trace: pso.run(
        P, seed=seed, trace=trace, **pso.PSO_AWAL
    ),
    "PSO terpilih": lambda P, seed, trace: pso.run(
        P, seed=seed, trace=trace, **pso.PSO_TERPILIH
    ),
    "SA": lambda P, seed, trace: sa.run(
        P, seed=seed, trace=trace
    ),
}


# Profil risiko investor -> pengali batas risiko (sigma_max = pengali x volatilitas portofolio 1/N)
PROFIL_RISIKO = {"Konservatif": 1.5, "Moderat": 2.0, "Agresif": 2.5}


def jalankan(metode, P, seeds, trace=False):
    if metode not in METODE:
        raise ValueError(f"metode tidak dikenal: {metode}")
    return [METODE[metode](P, int(s), trace) for s in seeds]


def terbaik(runs):
    """Run terbaik menurut aturan Deb."""
    b = runs[0]
    for r in runs[1:]:
        if deb_better((r.R, r.viol), (b.R, b.viol)):
            b = r
    return b


def ringkas(runs, R_ref=None, tol=1e-9):
    """Statistik R antar seed. sukses = pangsa run layak yang mencapai R_ref."""
    layak = np.array([r.feasible for r in runs])
    R = np.array([r.R for r in runs])
    Rl = R[layak] if layak.any() else np.array([np.nan])
    sukses = float(np.mean(layak & (R >= R_ref - tol))) if R_ref is not None else np.nan
    return dict(median=float(np.median(Rl)), std=float(np.std(Rl)), minimum=float(Rl.min()), maks=float(Rl.max()),
                layak=float(layak.mean()), sukses=sukses, detik=float(np.mean([r.seconds for r in runs])),
                eval_terbaik=float(np.median([r.n_eval_best for r in runs])))


def bandingkan(a, b, alpha=0.05):
    """Uji Mann-Whitney dua sisi atas R (run tak layak dihitung terburuk)."""
    Ra = [r.R if r.feasible else -1.0 for r in a]
    Rb = [r.R if r.feasible else -1.0 for r in b]
    if np.ptp(np.concatenate([Ra, Rb])) == 0:
        return dict(p=1.0, lebih_baik="tidak berbeda nyata")
    p = float(mannwhitneyu(Ra, Rb, alternative="two-sided")[1])
    if p >= alpha:
        return dict(p=p, lebih_baik="tidak berbeda nyata")
    return dict(p=p, lebih_baik="A" if np.median(Ra) > np.median(Rb) else "B")


def konvergensi(runs, fraksi=(0.9, 0.99)):
    """Kurva konvergensi antar seed (median dan kuartil R terbaik yang layak) dan titik capai.

    capai_X = evaluasi pertama saat median mencapai R_awal + X (R_akhir - R_awal), dengan R_awal = nilai median
    pertama yang terdefinisi (minimal separuh run sudah layak). layak_pertama = evaluasi saat titik itu muncul.
    """
    x = np.array([h[0] for h in runs[0].history])
    Y = np.array([[h[1] for h in r.history] for r in runs], float)
    ada = (~np.isnan(Y)).mean(axis=0) >= 0.5
    med, q1, q3 = (np.where(ada, np.nanpercentile(np.where(np.isnan(Y), np.nan, Y), q, axis=0), np.nan)
                   if ada.any() else np.full(len(x), np.nan) for q in (50, 25, 75))
    hasil = dict(evaluasi=x, median=med, q1=q1, q3=q3)
    if not ada.any():
        return {**hasil, "R_awal": np.nan, "R_akhir": np.nan, "layak_pertama": np.nan,
                "capai_90": np.nan, "capai_99": np.nan}
    i0 = int(np.argmax(ada))
    R0, R1 = float(med[i0]), float(med[-1])
    hasil.update(R_awal=R0, R_akhir=R1, layak_pertama=int(x[i0]))
    for f in fraksi:
        ambang = R0 + f * (R1 - R0)
        idx = np.where(ada & (med >= ambang - 1e-12))[0]
        hasil[f"capai_{int(round(f * 100))}"] = int(x[idx[0]])
    return hasil
