"""Konsistensi keempat metaheuristik (GA, PSO, SA, DE): perlakuan sama sesuai spesifikasi."""
import numpy as np
import pytest

import config
from algorithms import de, ga, pso, sa
from data import buat_problem, muat_jendela

MODUL = {"GA": ga, "PSO": pso, "SA": sa, "DE": de}
BUDGET = 2000                     # kecil agar tes cepat; semua kelipatan populasi (100, 40) dan SA bebas


def _P(K=7, nama="kasus_tetap"):
    return buat_problem(muat_jendela(nama), K)


def test_proporsi_semai():
    assert [config.n_semai(n) for n in (100, 40, 1, 50, 200)] == [10, 4, 1, 5, 20]


@pytest.mark.parametrize("nama", MODUL)
def test_antarmuka_anggaran_riwayat_dan_kendala_sama(nama):
    P = _P(7)
    r = MODUL[nama].run(P, budget=BUDGET, seed=3)
    assert r.n_eval == BUDGET
    assert len(r.history) == BUDGET // config.HISTORY_STEP
    assert r.feasible and (r.lots > 0).sum() <= P.K
    assert (r.lots * P.harga_lot / P.D).max() <= P.cap + 1e-9
    assert (r.lots * P.harga_lot).sum() <= P.D
    r2 = MODUL[nama].run(P, budget=BUDGET, seed=3)
    assert r.R == r2.R and np.array_equal(r.lots, r2.lots)          # seed sama -> hasil sama


@pytest.mark.parametrize("nama", MODUL)
def test_semai_aktif_secara_bawaan(nama):
    """Parameter bawaan menyemai top-K; n_seed=0 mengubah populasi awal sehingga hasil run berbeda."""
    P = _P(7)
    a = MODUL[nama].run(P, budget=BUDGET, seed=5)
    b = MODUL[nama].run(P, budget=BUDGET, seed=5, n_seed=0)
    assert not np.array_equal(a.x, b.x)


@pytest.mark.parametrize("nama", MODUL)
def test_n_seed_bawaan_sama_dengan_proporsi_config(nama):
    """Menyebut n_seed secara eksplisit sebesar proporsi config harus identik dengan bawaan."""
    P = _P(5)
    ukuran = {"GA": ga.GA_PARAM["pop"], "PSO": pso.PSO_AWAL["n"], "DE": de.DE_PARAM["pop"], "SA": 1}[nama]
    a = MODUL[nama].run(P, budget=BUDGET, seed=1)
    b = MODUL[nama].run(P, budget=BUDGET, seed=1, n_seed=config.n_semai(ukuran))
    assert a.R == b.R and np.array_equal(a.x, b.x)


def test_ga_laju_mutasi_dapat_ditala_dan_bawaan_tidak_berubah():
    P = _P(5)
    a = ga.run(P, budget=BUDGET, seed=2)
    assert a.R == ga.run(P, budget=BUDGET, seed=2, laju_mut=1.0).R
    assert not np.array_equal(a.x, ga.run(P, budget=BUDGET, seed=2, laju_mut=4.0).x)


@pytest.mark.parametrize("pop", [50, 100, 200])
def test_ga_ukuran_populasi_tepat_anggaran(pop):
    r = ga.run(_P(5), budget=config.BUDGET_EVAL, seed=0, pop=pop)
    assert r.n_eval == config.BUDGET_EVAL and r.feasible
