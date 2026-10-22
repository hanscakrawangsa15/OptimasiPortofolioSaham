"""Genetic Algorithm: turnamen, SBX, mutasi Gaussian + swap kunci, elitisme."""
import time

import numpy as np

import config
from decoder import decode_batch, init_population
from fitness import Evaluator, deb_rank, deb_urut

GA_PARAM = dict(pop=100, turnamen=3, p_sbx=0.9, eta=15.0, sigma_mut=0.1,
                p_swap=0.2, elitisme=2, n_seed=None, laju_mut=1.0)   # laju_mut: pengali peluang mutasi per gen (1/N)


def turnamen(rank, rng, n, k):
    """Seleksi turnamen ukuran k berdasarkan peringkat Deb (rank kecil = baik)."""
    cand = rng.integers(0, len(rank), size=(n, k))
    return cand[np.arange(n), np.argmin(rank[cand], axis=1)]


def sbx(p1, p2, eta, rng):
    u = rng.random(p1.shape)
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    return (0.5 * ((1 + beta) * p1 + (1 - beta) * p2),
            0.5 * ((1 - beta) * p1 + (1 + beta) * p2))


def mutasi_swap(kids, P, p_swap, rng):
    """Tukar kunci satu saham terpilih dengan satu saham terjangkau di luar himpunan terpilih."""
    K = P.K
    kand = np.where(P.layak_beli)[0]
    if len(kand) <= K:
        return
    for i in np.where(rng.random(len(kids)) < p_swap)[0]:
        urut = kand[np.argsort(-kids[i, kand], kind="stable")]
        a, b = urut[rng.integers(0, K)], urut[rng.integers(K, len(urut))]
        kids[i, a], kids[i, b] = kids[i, b], kids[i, a]


def run(P, budget=config.BUDGET_EVAL, seed=0, trace=False, **kw):
    """Satu run GA. trace=True menyimpan keadaan populasi tiap generasi (untuk visualisasi proses)."""
    prm = {**GA_PARAM, **kw}
    rng = np.random.default_rng(seed)
    ev = Evaluator(P, budget)
    t0 = time.perf_counter()
    n, N, E = prm["pop"], P.n, prm["elitisme"]
    rec = {k: [] for k in ("R", "sigma", "viol", "g0", "kas", "n_unik", "lots_terbaik")} if trace else None

    def catat(pop, R, s, v):
        if rec is None:
            return
        lots, _, kas = decode_batch(pop, P)
        rec["R"].append(R.copy()); rec["sigma"].append(s.copy()); rec["viol"].append(v.copy())
        rec["g0"].append(np.clip(pop[:, -1], 0.0, 1.0)); rec["kas"].append(kas)
        rec["n_unik"].append(len({tuple(np.where(l > 0)[0]) for l in lots}))
        rec["lots_terbaik"].append(lots[deb_urut(R, v)[0]])

    n_seed = config.n_semai(n) if prm["n_seed"] is None else prm["n_seed"]
    pop = init_population(P, rng, n, n_seed)
    R, s, v = ev.evaluate(pop)
    catat(pop, R, s, v)

    while ev.n_eval + n <= budget:
        rank = deb_rank(R, v)
        elit = np.argsort(rank)[:E]                              # elit generasi ini
        par = pop[turnamen(rank, rng, n, prm["turnamen"])]
        p1, p2 = par[0::2], par[1::2]
        c1, c2 = sbx(p1, p2, prm["eta"], rng)
        pakai = (rng.random(len(p1)) < prm["p_sbx"])[:, None]    # SBX hanya pada sebagian pasangan
        kids = np.empty_like(pop)
        kids[0::2], kids[1::2] = np.where(pakai, c1, p1), np.where(pakai, c2, p2)

        mask = rng.random((n, N + 1)) < prm["laju_mut"] / N                  # mutasi Gaussian: bobot mentah dan g0
        kids[:, N:] += mask * rng.normal(0.0, prm["sigma_mut"], (n, N + 1))
        mutasi_swap(kids, P, prm["p_swap"], rng)
        kids = np.clip(kids, 0.0, 1.0)

        Rk, sk, vk = ev.evaluate(kids)
        terburuk = deb_urut(Rk, vk)[::-1][:E]                    # elitisme: elit menggantikan anak terburuk
        kids[terburuk], Rk[terburuk], sk[terburuk], vk[terburuk] = pop[elit], R[elit], s[elit], v[elit]
        pop, R, s, v = kids, Rk, sk, vk
        catat(pop, R, s, v)

    hasil = ev.hasil(seed, time.perf_counter() - t0)
    if trace:
        hasil.trace = {k: np.array(val) for k, val in rec.items()}
    return hasil
