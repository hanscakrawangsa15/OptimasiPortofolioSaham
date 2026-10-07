import numpy as np
import pytest

from portofolio import (repair_population, repair_weights, sharpe_population,
                        sharpe_ratio)


def _valid(w, cap):
    return (abs(w.sum() - 1) < 1e-9) and (w >= -1e-12).all() and (w <= cap + 1e-9).all()


def test_sharpe_manual():
    mu = np.array([0.10, 0.20])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    w = np.array([0.5, 0.5])
    expected = (0.15 - 0.05) / np.sqrt(0.25 * 0.04 + 0.25 * 0.09)
    assert sharpe_ratio(w, mu, cov, 0.05) == pytest.approx(expected)


def test_sharpe_population_sama_dengan_satuan():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    cov = A @ A.T / 5 + 0.01 * np.eye(5)
    mu = rng.normal(0.1, 0.05, 5)
    pop = rng.dirichlet(np.ones(5), size=7)
    got = sharpe_population(pop, mu, cov, 0.05)
    want = [sharpe_ratio(w, mu, cov, 0.05) for w in pop]
    assert np.allclose(got, want)


def test_sharpe_varians_nol_tidak_nan():
    mu = np.array([0.1, 0.1])
    cov = np.zeros((2, 2))
    val = sharpe_ratio(np.array([0.5, 0.5]), mu, cov, 0.05)
    assert np.isfinite(val)


@pytest.mark.parametrize("seed", range(20))
def test_repair_acak_selalu_valid(seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.05, 0.3, 15)          # ada negatif dan > cap
    out = repair_weights(w, 0.10)
    assert _valid(out, 0.10)


def test_repair_semua_nol_jadi_sama_rata():
    out = repair_weights(np.zeros(10), 0.10)
    assert np.allclose(out, 0.1)


def test_repair_terkonsentrasi_dipotong():
    w = np.array([1.0] + [0.0] * 19)
    out = repair_weights(w, 0.10)
    assert _valid(out, 0.10)
    assert out.max() <= 0.10 + 1e-9


def test_repair_cap_tidak_layak_error():
    with pytest.raises(ValueError, match="tidak layak"):
        repair_weights(np.ones(5) / 5, 0.10)   # 5 x 0.10 < 1


def test_repair_cap_pas_sama_rata():
    out = repair_weights(np.random.default_rng(0).random(10), 0.10)
    assert np.allclose(out, 0.1)


def test_repair_population_valid_dan_tidak_mengubah_yang_sudah_valid():
    rng = np.random.default_rng(3)
    pop = rng.normal(0.05, 0.2, (50, 20))
    out = repair_population(pop, 0.10)
    assert all(_valid(r, 0.10) for r in out)
    again = repair_population(out.copy(), 0.10)
    assert np.allclose(out, again)
