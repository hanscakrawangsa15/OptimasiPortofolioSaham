"""Particle Swarm Optimization pada kromosom 2N+1 (posisi dijepit di [0,1])."""
import time

import numpy as np

import config
from decoder import decode, decode_batch, init_population
from fitness import Evaluator, deb_lebih_baik, deb_urut
from parameter import kw_terpilih

PSO_AWAL = dict(n=40, w=0.7, c1=1.5, c2=1.5, v_maks=0.2, n_seed=None)   # parameter awal (PDF)
# Hasil penalaan Fase 1 yang dibekukan (hasil/csv/fase1_parameter_beku.json); cadangan: penalaan lama tanpa semai.
PSO_TERPILIH = kw_terpilih("PSO") or dict(w=1.0, c1=1.0, c2=1.0, v_maks=1.0)


def run(P, budget=config.BUDGET_EVAL, seed=0, trace=False, **kw):
    """Satu run PSO. trace=True menyimpan keadaan swarm tiap iterasi (untuk visualisasi proses)."""
    prm = {**PSO_AWAL, **kw}
    rng = np.random.default_rng(seed)
    ev = Evaluator(P, budget)
    t0 = time.perf_counter()
    n, d, vm = prm["n"], P.n_genes, prm["v_maks"]
    kunci = ("R", "sigma", "viol", "g0", "kas", "n_unik", "R_gbest", "gb_R", "gb_sigma", "g0_gbest", "kas_gbest",
             "lots_terbaik", "kecepatan", "jarak", "x_min", "x_max")
    rec = {k: [] for k in kunci} if trace else None

    n_seed = config.n_semai(n) if prm["n_seed"] is None else prm["n_seed"]
    x = init_population(P, rng, n, n_seed=n_seed)             # posisi awal acak seragam, sebagian disemai top-K
    vel = rng.uniform(-vm, vm, (n, d))
    R, s, vi = ev.evaluate(x)
    pb_x, pb_R, pb_s, pb_v = x.copy(), R.copy(), s.copy(), vi.copy()    # pbest tiap partikel
    g = int(deb_urut(pb_R, pb_v)[0])                                    # gbest = pbest terbaik menurut Deb

    def catat():
        if rec is None:
            return
        lots, _, kas = decode_batch(x, P)
        lg, _, kg = decode(pb_x[g], P)
        rec["R"].append(R.copy()); rec["sigma"].append(s.copy()); rec["viol"].append(vi.copy())
        rec["g0"].append(np.clip(x[:, -1], 0.0, 1.0)); rec["kas"].append(kas)
        rec["n_unik"].append(len({tuple(np.where(l > 0)[0]) for l in lots}))
        rec["R_gbest"].append(pb_R[g] if pb_v[g] <= 0 else np.nan)
        rec["gb_R"].append(pb_R[g]); rec["gb_sigma"].append(pb_s[g])
        rec["g0_gbest"].append(float(np.clip(pb_x[g][-1], 0.0, 1.0))); rec["kas_gbest"].append(kg)
        rec["lots_terbaik"].append(lg)
        rec["kecepatan"].append(float(np.abs(vel).mean()))
        rec["jarak"].append(float(np.linalg.norm(x - pb_x[g], axis=1).mean()))
        rec["x_min"].append(float(x.min())); rec["x_max"].append(float(x.max()))

    catat()
    while ev.n_eval + n <= budget:
        r1, r2 = rng.random((n, d)), rng.random((n, d))
        vel = prm["w"] * vel + prm["c1"] * r1 * (pb_x - x) + prm["c2"] * r2 * (pb_x[g] - x)
        vel = np.clip(vel, -vm, vm)
        x = x + vel
        luar = (x < 0.0) | (x > 1.0)
        x = np.clip(x, 0.0, 1.0)
        vel = np.where(luar, 0.0, vel)                         # menabrak batas: kecepatan komponen itu dinolkan
        R, s, vi = ev.evaluate(x)
        lebih = deb_lebih_baik(R, vi, pb_R, pb_v)              # pbest diperbarui menurut aturan Deb
        pb_x[lebih], pb_R[lebih], pb_s[lebih], pb_v[lebih] = x[lebih], R[lebih], s[lebih], vi[lebih]
        g = int(deb_urut(pb_R, pb_v)[0])
        catat()

    hasil = ev.hasil(seed, time.perf_counter() - t0)
    if trace:
        hasil.trace = {k: np.array(val) for k, val in rec.items()}
    return hasil
