"""Genetic Algorithm real-coded untuk memaksimalkan Sharpe ratio portofolio.

Kromosom = vektor bobot. Operator: seleksi turnamen, SBX, mutasi Gaussian,
elitisme, dan repair (potong ke cap lalu normalisasi) setelah variasi.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from portofolio import repair_population, sharpe_population  # noqa: E402


@dataclass
class GAResult:
    seed: int
    weights: np.ndarray
    fitness: float
    hist_best: np.ndarray
    hist_mean: np.ndarray


def _seleksi_turnamen(fit, k, rng):
    p = len(fit)
    cand = rng.integers(0, p, size=(p, k))
    return cand[np.arange(p), np.argmax(fit[cand], axis=1)]


def _sbx(p1, p2, eta, rng):
    u = rng.random(p1.shape)
    beta = np.where(u <= 0.5,
                    (2 * u) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return c1, c2


def run_ga(mu, cov, cap, rf, seed, pop_size=config.POP_SIZE,
           generations=config.GENERATIONS, tournament_k=3,
           crossover_prob=0.9, eta_c=15.0, mutation_prob=None,
           mutation_sigma=0.02, elite=2):
    if pop_size % 2:
        raise ValueError("pop_size harus genap")
    n = len(mu)
    if n * cap < 1 - 1e-12:
        raise ValueError(
            f"Cap {cap} tidak layak untuk {n} saham (perlu n x cap >= 1)")
    pm = 1.0 / n if mutation_prob is None else mutation_prob
    rng = np.random.default_rng(seed)

    pop = repair_population(rng.dirichlet(np.ones(n), size=pop_size), cap)
    fit = sharpe_population(pop, mu, cov, rf)
    hist_best = [fit.max()]
    hist_mean = [fit.mean()]

    for _ in range(generations):
        elit_idx = np.argsort(fit)[-elite:]
        elit = pop[elit_idx].copy()

        parents = pop[_seleksi_turnamen(fit, tournament_k, rng)]
        p1, p2 = parents[0::2], parents[1::2]
        c1, c2 = _sbx(p1, p2, eta_c, rng)
        do_x = (rng.random(len(p1)) < crossover_prob)[:, None]
        c1 = np.where(do_x, c1, p1)
        c2 = np.where(do_x, c2, p2)
        child = np.empty_like(pop)
        child[0::2], child[1::2] = c1, c2

        mask = rng.random(child.shape) < pm
        child = child + mask * rng.normal(0.0, mutation_sigma, child.shape)
        child = repair_population(child, cap)

        child_fit = sharpe_population(child, mu, cov, rf)
        worst = np.argsort(child_fit)[:elite]
        child[worst] = elit
        child_fit[worst] = fit[elit_idx]

        pop, fit = child, child_fit
        hist_best.append(fit.max())
        hist_mean.append(fit.mean())

    i = int(np.argmax(fit))
    return GAResult(seed=seed, weights=pop[i].copy(), fitness=float(fit[i]),
                    hist_best=np.array(hist_best), hist_mean=np.array(hist_mean))


def run_ga_batch(mu, cov, cap, rf, seeds, **kwargs):
    return [run_ga(mu, cov, cap, rf, seed=s, **kwargs) for s in seeds]


def pilih_terbaik(results):
    """Run dengan fitness LATIH tertinggi (bukan berdasarkan data uji)."""
    return max(results, key=lambda r: r.fitness)


def pilih_median(results):
    urut = sorted(results, key=lambda r: r.fitness)
    return urut[len(urut) // 2]
