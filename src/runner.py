"""Menjalankan beberapa seed, meringkas, dan membandingkan dua kelompok run."""
import numpy as np
from scipy.stats import mannwhitneyu

from algorithms import ga, pso
from fitness import deb_better

METODE = {
    "GA": lambda P, seed, trace: ga.run(P, seed=seed, trace=trace),
    "PSO awal": lambda P, seed, trace: pso.run(P, seed=seed, trace=trace, **pso.PSO_AWAL),
    "PSO terpilih": lambda P, seed, trace: pso.run(P, seed=seed, trace=trace, **pso.PSO_TERPILIH),
}


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
