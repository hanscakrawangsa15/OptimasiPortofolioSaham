import numpy as np
import pytest

from baseline import max_sharpe_slsqp
from GeneticAlgorithm import (pilih_median, pilih_terbaik, run_ga,
                              run_ga_batch)
from portofolio import sharpe_ratio


def _masalah(n=15, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    cov = A @ A.T / n * 0.1 + 0.02 * np.eye(n)
    mu = rng.normal(0.12, 0.08, n)
    return mu, cov


KW = dict(pop_size=60, generations=80)


def test_hasil_valid_dan_riwayat_benar():
    mu, cov = _masalah()
    r = run_ga(mu, cov, 0.10, 0.05, seed=1, **KW)
    assert r.weights.sum() == pytest.approx(1, abs=1e-9)
    assert (r.weights >= -1e-12).all() and (r.weights <= 0.10 + 1e-9).all()
    assert len(r.hist_best) == 81 and len(r.hist_mean) == 81
    assert r.fitness == pytest.approx(sharpe_ratio(r.weights, mu, cov, 0.05))


def test_fitness_terbaik_tidak_turun_elitisme():
    mu, cov = _masalah()
    r = run_ga(mu, cov, 0.10, 0.05, seed=2, **KW)
    assert (np.diff(r.hist_best) >= -1e-12).all()
    assert r.hist_best[-1] > r.hist_best[0]


def test_reprodusibel_dengan_seed_sama():
    mu, cov = _masalah()
    a = run_ga(mu, cov, 0.10, 0.05, seed=7, **KW)
    b = run_ga(mu, cov, 0.10, 0.05, seed=7, **KW)
    c = run_ga(mu, cov, 0.10, 0.05, seed=8, **KW)
    assert np.array_equal(a.weights, b.weights)
    assert not np.array_equal(a.weights, c.weights)


def test_populasi_ganjil_ditolak():
    mu, cov = _masalah()
    with pytest.raises(ValueError, match="genap"):
        run_ga(mu, cov, 0.10, 0.05, seed=0, pop_size=51, generations=5)


def test_cap_tidak_layak_ditolak():
    mu, cov = _masalah(5)
    with pytest.raises(ValueError, match="tidak layak"):
        run_ga(mu, cov, 0.10, 0.05, seed=0, pop_size=20, generations=5)


def test_mendekati_slsqp():
    mu, cov = _masalah()
    ref = sharpe_ratio(max_sharpe_slsqp(mu, cov, 0.05, 0.10), mu, cov, 0.05)
    r = run_ga(mu, cov, 0.10, 0.05, seed=3, pop_size=200, generations=300)
    assert r.fitness >= 0.95 * ref


def test_batch_dan_pemilihan():
    mu, cov = _masalah()
    rs = run_ga_batch(mu, cov, 0.10, 0.05, seeds=range(5), pop_size=40,
                      generations=20)
    assert [r.seed for r in rs] == [0, 1, 2, 3, 4]
    best = pilih_terbaik(rs)
    assert best.fitness == max(r.fitness for r in rs)
    med = pilih_median(rs)
    urut = sorted(r.fitness for r in rs)
    assert med.fitness == urut[len(urut) // 2]
