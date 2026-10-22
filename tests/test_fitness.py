"""Tes aturan Deb dan Evaluator."""
import numpy as np
import pytest

from data import buat_problem, muat_jendela
from fitness import (AnggaranHabis, Evaluator, deb_better, deb_lebih_baik,
                     deb_rank, deb_urut, hitung)
from decoder import decode_batch


def _P(K=5):
    return buat_problem(muat_jendela("kasus_tetap"), K)


def test_urutan_deb():
    R = np.array([0.30, 0.20, 0.50, 0.40, 0.10])
    v = np.array([0.0, 0.0, 0.05, 0.01, 0.0])
    assert list(deb_urut(R, v)) == [0, 1, 4, 3, 2]        # layak (R turun), lalu tak layak (viol naik)
    assert list(deb_rank(R, v)) == [0, 1, 4, 3, 2]


def test_deb_better_skalar():
    assert deb_better((0.2, 0.0), (0.9, 0.01)) and not deb_better((0.9, 0.01), (0.2, 0.0))
    assert deb_better((0.3, 0.0), (0.2, 0.0)) and deb_better((0.1, 0.01), (0.9, 0.05))
    assert not deb_better((0.3, 0.0), (0.3, 0.0))


def test_deb_vektor_sama_dengan_skalar():
    rng = np.random.default_rng(0)
    R1, R2 = rng.random(500).round(1), rng.random(500).round(1)          # nilai kembar sengaja
    v1 = np.where(rng.random(500) < 0.5, 0.0, rng.random(500).round(2))
    v2 = np.where(rng.random(500) < 0.5, 0.0, rng.random(500).round(2))
    vek = deb_lebih_baik(R1, v1, R2, v2)
    assert all(vek[i] == deb_better((R1[i], v1[i]), (R2[i], v2[i])) for i in range(500))


def test_hitung_objektif_dan_pelanggaran():
    P = _P(5)
    X = np.random.default_rng(2).random((20, P.n_genes))
    lots, w, kas = decode_batch(X, P)
    R, s, v = hitung(lots, w, kas, P)
    assert np.allclose(R, w @ P.mu + kas * P.rf)
    assert np.allclose(s, np.sqrt(np.einsum("ij,jk,ik->i", w, P.cov, w)))
    assert np.allclose(v, np.maximum(0, s - P.sigma_max))


def test_evaluator_anggaran_riwayat_dan_lebar():
    P = _P(5)
    ev = Evaluator(P, budget=1000)
    rng = np.random.default_rng(1)
    ev.evaluate(rng.random((450, P.n_genes)))
    ev.evaluate(rng.random((150, P.n_genes)))
    assert ev.n_eval == 600 and [h[0] for h in ev.history] == [200, 400, 600]
    layak = [h[1] for h in ev.history if not np.isnan(h[1])]
    assert all(b >= a - 1e-12 for a, b in zip(layak, layak[1:]))
    with pytest.raises(AnggaranHabis):
        ev.evaluate(rng.random((401, P.n_genes)))
    with pytest.raises(ValueError, match="lebar"):
        ev.evaluate(np.zeros((1, P.n_genes - 1)))


def test_evaluator_hasil():
    P = _P(5)
    ev = Evaluator(P, budget=400)
    ev.evaluate(np.random.default_rng(4).random((400, P.n_genes)))
    r = ev.hasil(seed=9, seconds=0.5)
    assert r.seed == 9 and r.n_eval == 400 and 1 <= r.n_eval_best <= 400
    assert r.feasible == (r.viol <= 0) and (r.lots > 0).sum() <= 5
