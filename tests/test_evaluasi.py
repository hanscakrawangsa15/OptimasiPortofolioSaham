"""Tes evaluasi (rincian kas, return, data uji) dan runner (ringkasan, perbandingan)."""
import dataclasses

import numpy as np
import pytest

import config
import runner
from algorithms import ga, pso
from data import buat_problem, muat_jendela
from decoder import decode
from evaluasi import (analisis, komponen_akhir, metrik_uji, nilai_1n,
                      nilai_kas, nilai_uji, rincian_kas, rp)


def _P(K, nama="kasus_tetap"):
    return buat_problem(muat_jendela(nama), K)


def test_format_rupiah():
    assert rp(1676000) == "Rp 1.676.000" and rp(0) == "Rp 0" and rp(250000.4) == "Rp 250.000"


def test_rincian_kas_contoh_jendela_3():
    P = _P(2, "jendela_3")
    ib, isa = P.tickers.index("BRIS"), P.tickers.index("ISAT")
    x = np.zeros(P.n_genes)
    x[ib] = x[isa] = 1.0
    x[P.n + ib], x[P.n + isa] = 0.55, 0.45
    x[-1] = 0.30
    r = rincian_kas(x, P)
    assert r["terpakai"] == pytest.approx(3_324_000) and r["kas_akhir"] == pytest.approx(1_676_000)
    assert r["kas_sengaja"] == pytest.approx(1_500_000) and r["kas_cap"] == pytest.approx(0, abs=1e-6)
    assert r["kas_pembulatan"] == pytest.approx(176_000)


def test_rincian_kas_kelebihan_cap():
    P = dataclasses.replace(_P(7), cap=0.10)            # 7 x 10% = 70%: 30% dana tak tertampung
    x = np.random.default_rng(5).random(P.n_genes)
    x[-1] = 0.0
    r = rincian_kas(x, P)
    assert r["kas_cap"] == pytest.approx(0.3 * P.D, rel=1e-6)
    assert r["kas_sengaja"] + r["kas_cap"] + r["kas_pembulatan"] == pytest.approx(r["kas_akhir"])
    assert r["kas_pembulatan"] >= 0


def test_analisis_sama_dengan_hasil_algoritma():
    P = _P(7)
    r = ga.run(P, seed=3)
    A = analisis(P, r.lots)
    assert A["R"] == pytest.approx(r.R, abs=1e-9) and A["sigma"] == pytest.approx(r.sigma, abs=1e-9)
    assert A["R_saham"].sum() + A["R_kas"] == pytest.approx(A["R"])
    assert A["kontr_risiko"].sum() == pytest.approx(1.0)
    assert A["w"].sum() + A["kas"] == pytest.approx(1.0)


def test_nilai_uji_kas_saja_dan_1n():
    r_uji = muat_jendela("kasus_tetap")["return_uji"]
    T = len(r_uji)
    D = config.D_UTAMA
    P = _P(5)
    kas = nilai_kas(D, P.rf, T)
    assert kas[-1] == pytest.approx(D * (1 + P.rf) ** (T / 252))
    nol = nilai_uji(P, np.zeros(P.n, dtype=int), r_uji)                  # tanpa saham = kas saja
    assert np.allclose(nol, kas)
    eq = nilai_1n(D, r_uji)
    assert len(eq) == T and eq[0] == pytest.approx(D * (1 + r_uji[0].mean()))


def test_metrik_uji_dan_komponen_akhir():
    r_uji = muat_jendela("kasus_tetap")["return_uji"]
    P = _P(5)
    r = ga.run(P, seed=3)
    V = nilai_uji(P, r.lots, r_uji)
    saham, kas = komponen_akhir(P, r.lots, r_uji)
    assert saham + kas == pytest.approx(V[-1])
    m = metrik_uji(V, P.D)
    assert m["total"] == pytest.approx(V[-1] / P.D - 1) and m["mdd"] <= 0 and m["vol"] > 0
    assert metrik_uji(nilai_kas(P.D, P.rf, len(r_uji)), P.D)["vol"] == pytest.approx(0, abs=1e-12)


# ------------------------------------------------------------------ runner
def test_jalankan_dan_terbaik():
    P = _P(5)
    runs = runner.jalankan("GA", P, seeds=[0, 1, 2])
    assert [r.seed for r in runs] == [0, 1, 2]
    best = runner.terbaik(runs)
    assert best.R == max(r.R for r in runs if r.feasible)
    runs2 = runner.jalankan("PSO terpilih", P, seeds=[0, 1])
    assert len(runs2) == 2 and all(r.n_eval == config.BUDGET_EVAL for r in runs2)
    with pytest.raises(ValueError):
        runner.jalankan("XYZ", P, seeds=[0])


def test_sa_terdaftar_di_runner():
    P = _P(5)
    runs = runner.jalankan("SA", P, seeds=[0, 1])
    assert len(runs) == 2 and all(r.n_eval == config.BUDGET_EVAL and r.feasible for r in runs)


def test_ringkas_dan_bandingkan():
    P = _P(5)
    a = runner.jalankan("GA", P, seeds=range(6))
    b = runner.jalankan("PSO awal", P, seeds=range(6))
    ring = runner.ringkas(a, R_ref=max(r.R for r in a + b if r.feasible))
    assert {"median", "std", "minimum", "maks", "layak", "sukses", "detik", "eval_terbaik"} <= set(ring)
    assert 0 <= ring["sukses"] <= 1 and ring["layak"] == 1.0
    cmp_ = runner.bandingkan(a, b)
    assert 0 <= cmp_["p"] <= 1 and cmp_["lebih_baik"] in ("A", "B", "tidak berbeda nyata")


# ------------------------------------------------------------------ profil risiko
def test_profil_risiko_memetakan_ke_pengali_sigma():
    assert runner.PROFIL_RISIKO == {"Konservatif": 1.5, "Moderat": 2.0, "Agresif": 2.5}
    assert set(runner.PROFIL_RISIKO.values()) <= set(config.SIGMA_MULT)


def test_konvergensi_kurva_dan_titik_capai():
    P = _P(5)
    runs = runner.jalankan("GA", P, seeds=range(4))
    k = runner.konvergensi(runs)
    n = config.BUDGET_EVAL // config.HISTORY_STEP
    assert len(k["evaluasi"]) == len(k["median"]) == len(k["q1"]) == len(k["q3"]) == n
    assert k["evaluasi"][0] == config.HISTORY_STEP and k["evaluasi"][-1] == config.BUDGET_EVAL
    med = k["median"][~np.isnan(k["median"])]
    assert (np.diff(med) >= -1e-12).all()                         # R terbaik tidak pernah turun
    assert k["R_akhir"] == pytest.approx(k["median"][-1])
    assert k["capai_90"] <= k["capai_99"] <= config.BUDGET_EVAL   # 90% tercapai lebih dulu dari 99%
    assert k["layak_pertama"] <= k["capai_90"]


def test_konvergensi_profil_lebih_longgar_return_tidak_turun():
    hasil = {}
    for nama, mult in runner.PROFIL_RISIKO.items():
        P = buat_problem(muat_jendela("kasus_tetap"), 5, mult=mult)
        hasil[nama] = runner.konvergensi(runner.jalankan("GA", P, seeds=range(3)))["R_akhir"]
    assert hasil["Agresif"] >= hasil["Konservatif"] - 1e-9


def test_metrik_uji_sharpe_cvar_dan_kas_saja():
    r_uji = muat_jendela("kasus_tetap")["return_uji"]
    P = _P(5)
    r = ga.run(P, seed=3)
    V = nilai_uji(P, r.lots, r_uji)
    m = metrik_uji(V, P.D)
    assert m["sharpe"] == pytest.approx((m["setara_tahun"] - P.rf) / m["vol"])
    awal = np.concatenate([[P.D], V])
    harian = np.diff(awal) / awal[:-1]
    n_ekor = max(1, int(np.ceil(0.05 * len(harian))))
    assert m["cvar95"] == pytest.approx(np.sort(harian)[:n_ekor].mean())     # rata-rata 5% hari terburuk (return harian)
    assert m["cvar95"] <= np.percentile(harian, 5) + 1e-12
    kas = metrik_uji(nilai_kas(P.D, P.rf, len(r_uji)), P.D)
    assert np.isnan(kas["sharpe"])                                          # tanpa volatilitas: Sharpe tak terdefinisi
