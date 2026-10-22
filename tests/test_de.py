"""Pengujian Differential Evolution terintegrasi."""
import numpy as np
import pytest
from algorithms import de
from decoder import decode
from data import buat_problem, muat_jendela


def test_de_seed_budget_constraint():
    P = buat_problem(muat_jendela('kasus_tetap'), 7)
    a = de.run(P, budget=400, seed=4, trace=True)
    b = de.run(P, budget=400, seed=4)
    assert a.R == b.R
    assert np.array_equal(a.lots, b.lots)
    assert a.n_eval == 400
    assert len(a.history) == 2
    assert a.feasible
    assert (a.lots > 0).sum() <= P.K
    assert (a.lots * P.harga_lot).sum() <= P.D + 1e-8
    assert (a.lots * P.harga_lot / P.D).max() <= P.cap + 1e-9
    assert a.trace['R'].shape == (4, 100)
    assert len(a.x) == P.n_genes


def test_de_budget_odd_and_other_seeds():
    P = buat_problem(muat_jendela('jendela_1'), 3)
    a = de.run(P, budget=253, pop=20, seed=0)
    b = de.run(P, budget=253, pop=20, seed=1)
    assert a.n_eval == b.n_eval == 253
    assert not np.array_equal(a.x, b.x)


def test_de_invalid_params():
    P = buat_problem(muat_jendela('kasus_tetap'), 2)
    with pytest.raises(ValueError):
        de.run(P, budget=5, pop=20)
    with pytest.raises(ValueError):
        de.run(P, budget=100, CR=2)