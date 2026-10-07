import numpy as np
import pytest

from baseline import equal_weight, max_sharpe_slsqp
from portofolio import sharpe_ratio


def _masalah(n=15, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    cov = A @ A.T / n * 0.1 + 0.02 * np.eye(n)
    mu = rng.normal(0.12, 0.08, n)
    return mu, cov


def test_equal_weight():
    w = equal_weight(20, 0.10)
    assert np.allclose(w, 0.05) and w.sum() == pytest.approx(1)


def test_equal_weight_cap_tidak_layak():
    with pytest.raises(ValueError, match="tidak layak"):
        equal_weight(5, 0.10)


def test_slsqp_memenuhi_batasan():
    mu, cov = _masalah()
    w = max_sharpe_slsqp(mu, cov, 0.05, 0.10)
    assert w.sum() == pytest.approx(1, abs=1e-6)
    assert (w >= -1e-9).all() and (w <= 0.10 + 1e-9).all()


def test_slsqp_lebih_baik_dari_sama_rata_dan_acak():
    mu, cov = _masalah()
    w = max_sharpe_slsqp(mu, cov, 0.05, 0.10)
    s = sharpe_ratio(w, mu, cov, 0.05)
    assert s >= sharpe_ratio(equal_weight(15, 0.10), mu, cov, 0.05) - 1e-9
    rng = np.random.default_rng(5)
    for _ in range(200):
        r = rng.dirichlet(np.ones(15))
        r = np.minimum(r, 0.10)
        r /= r.sum()
        if r.max() <= 0.10 + 1e-9:
            assert s >= sharpe_ratio(r, mu, cov, 0.05) - 1e-6


def test_slsqp_cap_tidak_layak():
    mu, cov = _masalah(5)
    with pytest.raises(ValueError, match="tidak layak"):
        max_sharpe_slsqp(mu, cov, 0.05, 0.10)
