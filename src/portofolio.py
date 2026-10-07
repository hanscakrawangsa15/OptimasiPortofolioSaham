"""Fungsi inti portofolio: Sharpe ratio dan operator repair batas bobot."""
import numpy as np

_EPS = 1e-12


def sharpe_ratio(w, mu, cov, rf):
    var = float(w @ cov @ w)
    vol = np.sqrt(max(var, _EPS))
    return float((w @ mu - rf) / vol)


def sharpe_population(pop, mu, cov, rf):
    ret = pop @ mu
    var = np.einsum("ij,jk,ik->i", pop, cov, pop)
    return (ret - rf) / np.sqrt(np.maximum(var, _EPS))


def repair_weights(w, cap):
    """Buat bobot valid: >= 0, jumlah 1, tiap bobot <= cap.

    Potong ke [0, cap], lalu bagikan kelebihan ke saham yang masih di bawah
    cap (sebanding bobotnya), ulangi sampai tidak ada yang melebihi cap.
    """
    n = len(w)
    if n * cap < 1 - 1e-12:
        raise ValueError(
            f"Cap {cap} tidak layak untuk {n} saham (perlu n x cap >= 1)")
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = w.sum()
    w = w / total if total > 0 else np.full(n, 1.0 / n)

    for _ in range(n + 1):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = w < cap - _EPS
        if not free.any():
            break
        base = w[free] if w[free].sum() > 0 else np.ones(free.sum())
        w[free] += excess * base / base.sum()
    return np.clip(w, 0.0, cap)


def repair_population(pop, cap):
    """Repair seluruh populasi; baris yang sudah valid hanya dinormalisasi."""
    pop = np.clip(pop, 0.0, None)
    sums = pop.sum(axis=1, keepdims=True)
    zero = (sums <= 0).ravel()
    pop = np.where(sums > 0, pop / np.where(sums > 0, sums, 1), 1.0 / pop.shape[1])
    bad = (pop > cap + _EPS).any(axis=1) | zero
    for i in np.where(bad)[0]:
        pop[i] = repair_weights(pop[i], cap)
    return pop
