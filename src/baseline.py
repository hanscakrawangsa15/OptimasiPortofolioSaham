"""Baseline pembanding: 1/N dan max-Sharpe (SLSQP) dengan batasan sama seperti GA."""
import numpy as np
from scipy.optimize import minimize

from portofolio import repair_weights, sharpe_ratio


def _cek_cap(n, cap):
    if n * cap < 1 - 1e-12:
        raise ValueError(
            f"Cap {cap} tidak layak untuk {n} saham (perlu n x cap >= 1)")


def equal_weight(n, cap):
    _cek_cap(n, cap)
    return np.full(n, 1.0 / n)


def max_sharpe_slsqp(mu, cov, rf, cap, n_starts=6, seed=0):
    """Maksimalkan Sharpe, long-only, jumlah 1, bobot <= cap.

    Dicoba dari beberapa titik awal (1/N dan acak yang di-repair); diambil
    hasil terbaik karena Sharpe tidak konveks pada w.
    """
    n = len(mu)
    _cek_cap(n, cap)
    rng = np.random.default_rng(seed)
    starts = [np.full(n, 1.0 / n)]
    starts += [repair_weights(rng.dirichlet(np.ones(n)), cap)
               for _ in range(n_starts - 1)]

    def neg_sharpe(w):
        return -sharpe_ratio(w, mu, cov, rf)

    best_w, best_val = None, np.inf
    for x0 in starts:
        res = minimize(neg_sharpe, x0, method="SLSQP",
                       bounds=[(0.0, cap)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       options={"maxiter": 500, "ftol": 1e-12})
        if res.success or res.fun < best_val:
            w = repair_weights(res.x, cap)       # rapikan galat numerik kecil
            val = neg_sharpe(w)
            if val < best_val:
                best_w, best_val = w, val
    return best_w
