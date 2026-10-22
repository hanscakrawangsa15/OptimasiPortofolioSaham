"""Tes GA, PSO, dan brute force K=2 (ground truth)."""
import numpy as np
import pytest

from algorithms import ga, pso
from algorithms.brute_force_k2 import brute_force_k2
from data import buat_problem, muat_jendela
import config

BUDGET = config.BUDGET_EVAL


def _P(K, nama="kasus_tetap"):
    return buat_problem(muat_jendela(nama), K)


@pytest.mark.parametrize("modul", [ga, pso])
def test_reprodusibel_anggaran_dan_kendala(modul):
    P = _P(7)
    a, b, c = modul.run(P, seed=11), modul.run(P, seed=11), modul.run(P, seed=12)
    assert a.n_eval == BUDGET and a.R == b.R and np.array_equal(a.lots, b.lots)
    assert not np.array_equal(a.x, c.x)
    assert len(a.history) == BUDGET // config.HISTORY_STEP
    assert a.feasible and (a.lots > 0).sum() <= 7
    assert (a.lots * P.harga_lot / P.D).max() <= P.cap + 1e-9
    assert (a.lots * P.harga_lot).sum() <= P.D


@pytest.mark.parametrize("modul", [ga, pso])
def test_trace_tidak_mengubah_hasil(modul):
    P = _P(5)
    a, t = modul.run(P, seed=3), modul.run(P, seed=3, trace=True)
    assert t.R == a.R and np.array_equal(t.lots, a.lots)
    n = ga.GA_PARAM["pop"] if modul is ga else pso.PSO_AWAL["n"]
    assert t.trace["R"].shape == (BUDGET // n, n)
    assert t.trace["lots_terbaik"].shape == (BUDGET // n, P.n)


def test_pso_posisi_di_batas_dan_gbest_monoton():
    P = _P(7)
    t = pso.run(P, seed=11, trace=True)
    assert t.trace["x_min"].min() >= 0.0 and t.trace["x_max"].max() <= 1.0
    rg = t.trace["R_gbest"][~np.isnan(t.trace["R_gbest"])]
    assert (np.diff(rg) >= -1e-12).all()


def test_pso_parameter_terpilih_dapat_dipakai():
    assert set(pso.PSO_TERPILIH) >= {"w", "c1", "c2", "v_maks"}
    r = pso.run(_P(5), seed=0, **pso.PSO_TERPILIH)
    assert r.feasible and r.n_eval == BUDGET


@pytest.mark.parametrize("nama,R_bf", [("kasus_tetap", 0.2966), ("jendela_3", 0.2817), ("jendela_5", 0.5114)])
def test_brute_force_k2_nilai_diketahui(nama, R_bf):
    hasil = brute_force_k2(_P(2, nama))
    assert hasil["kas bebas"][0] == pytest.approx(R_bf, abs=5e-5)


def test_brute_force_kas_dibatasi_lebih_rendah():
    hasil = brute_force_k2(_P(2, "jendela_3"))
    assert hasil["kas <= 5%"][0] < hasil["kas hanya dari pembulatan"][0] < hasil["kas bebas"][0]


@pytest.mark.parametrize("modul", [ga, pso])
def test_tidak_pernah_melampaui_optimum_sejati(modul):
    P = _P(2, "jendela_3")
    R_star = brute_force_k2(P)["kas bebas"][0]
    for s in range(4):
        r = modul.run(P, seed=s)
        assert (not r.feasible) or r.R <= R_star + 1e-9


def test_ga_menemukan_optimum_k2_pada_sebagian_seed():
    P = _P(2, "jendela_3")
    R_star = brute_force_k2(P)["kas bebas"][0]
    assert max(ga.run(P, seed=s).R for s in range(8)) >= R_star - 1e-9
