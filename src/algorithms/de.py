"""Differential Evolution (DE/rand/1/bin) untuk kromosom portofolio 2N+1."""

import time
import numpy as np
import config
from decoder import decode_batch, init_population
from fitness import Evaluator, deb_lebih_baik, deb_urut

DE_PARAM = dict(pop=100, F=0.5, CR=0.9, n_seed=None)


def run(P, budget=config.BUDGET_EVAL, seed=0, trace=False, **kw):
    """Satu run dengan anggaran evaluasi persis 'budget', jika budget >= pop.

    Untuk setiap target, pilih tiga indeks berbeda; mutant = a+F*(b-c).
    Crossover binomial menjamin >=1 gen berasal dari mutant. Seleksi memakai Deb."""
    
    prm = {**DE_PARAM, **kw}
    pop_size = int(prm['pop'])
    F, CR = float(prm['F']), float(prm['CR'])
    if pop_size < 4 or budget < pop_size:
        raise ValueError('DE memerlukan pop >= 4 dan budget >= pop')
    if not (0 < F <= 2 and 0 <= CR <= 1):
        raise ValueError('Harus 0 < F <= 2 dan 0 <= CR <= 1')
    rng = np.random.default_rng(seed)
    ev = Evaluator(P, budget)
    start = time.perf_counter()
    n, d = pop_size, P.n_genes
    n_seed = config.n_semai(n) if prm['n_seed'] is None else int(prm['n_seed'])
    x = init_population(P, rng, n, n_seed=n_seed)
    R, sigma, viol = ev.evaluate(x)
    trace_data = {k: [] for k in ('R', 'sigma', 'viol', 'g0', 'kas', 'n_unik', 'lots_terbaik')} if trace else None

    def catat():
        if trace_data is None:
            return
        lots, _, kas = decode_batch(x, P)
        trace_data['R'].append(R.copy())
        trace_data['sigma'].append(sigma.copy())
        trace_data['viol'].append(viol.copy())
        trace_data['g0'].append(x[:, -1].copy())
        trace_data['kas'].append(kas.copy())
        trace_data['n_unik'].append(len({tuple(np.flatnonzero(l > 0)) for l in lots}))
        trace_data['lots_terbaik'].append(lots[deb_urut(R, viol)[0]].copy())

    catat()
    # Sisa anggaran dibagi dalam batch; mendukung budget yang tidak kelipatan pop.
    while ev.n_eval < budget:
        m = min(n, budget - ev.n_eval)
        # Sampling 3 donor tanpa pengembalian dan tanpa indeks target.
        # Untuk setiap target i, ambil 3 dari [0,n) \ {i} dengan permutation.
        scores = rng.random((m, n))
        scores[np.arange(m), np.arange(m)] = np.inf
        donors = np.argpartition(scores, kth=2, axis=1)[:, :3]
        a, b, c = donors.T
        mutant = np.clip(x[a] + F * (x[b] - x[c]), 0.0, 1.0)
        cross = rng.random((m, d)) < CR
        cross[np.arange(m), rng.integers(0, d, size=m)] = True
        trial = np.where(cross, mutant, x[:m])
        Rt, st, vt = ev.evaluate(trial)
        accept = deb_lebih_baik(Rt, vt, R[:m], viol[:m])
        idx = np.flatnonzero(accept)
        x[idx] = trial[idx]
        R[idx], sigma[idx], viol[idx] = Rt[idx], st[idx], vt[idx]
        catat()

    result = ev.hasil(seed, time.perf_counter() - start)
    if trace:
        result.trace = {k: np.asarray(v) for k, v in trace_data.items()}
    return result