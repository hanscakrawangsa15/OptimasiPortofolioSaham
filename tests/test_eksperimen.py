"""Tes Fase 2 (eksperimen) dan statistik antar algoritma."""
import numpy as np
import pandas as pd
import pytest

import config
import eksperimen
import stats


def test_30_masalah():
    m = eksperimen.daftar_masalah()
    assert len(m) == 30 and len(set(m)) == 30
    assert {d for d, _ in m} == {"kasus_tetap"} | {f"jendela_{i}" for i in range(1, 6)}
    assert {K for _, K in m} == set(config.K_LIST)


def test_jalankan_kecil_seed_sama_antar_algoritma_dan_kolom_lengkap():
    df = eksperimen.jalankan(masalah=[("jendela_3", 2), ("kasus_tetap", 5)], seeds=(0, 1), budget=2000,
                             proses=1, kw={a: {} for a in eksperimen.ALGORITMA}, cetak=lambda *_: None)
    assert len(df) == 4 * 2 * 2 * 1 and set(df["algoritma"]) == set(eksperimen.ALGORITMA)
    assert (df["n_eval"] == 2000).all() and df["layak"].all()
    assert (df["jumlah_saham"] <= df["K"]).all()
    for kolom in ["return_uji_tahunan", "vol_uji", "mdd_uji", "kas", "g0", "lots", "risiko_uji_memenuhi"]:
        assert kolom in df and df[kolom].notna().all()
    assert (df.groupby(["algoritma", "data", "K"])["seed"].apply(lambda s: sorted(s) == [0, 1])).all()   # seed sama


def test_run_terjamin_deterministik():
    a = eksperimen._satu_run(("GA", "jendela_3", 5, 4, 2000, {}))
    b = eksperimen._satu_run(("GA", "jendela_3", 5, 4, 2000, {}))
    assert a["R"] == b["R"] and a["lots"] == b["lots"]


def test_peringkat_dan_wilcoxon_holm():
    rng = np.random.default_rng(0)
    dasar = rng.random(30)
    tabel = pd.DataFrame({"A": dasar + 0.10, "B": dasar + 0.05, "C": dasar})
    assert list(stats.peringkat_rata(tabel).index) == ["A", "B", "C"]
    w = stats.wilcoxon_holm(tabel)
    assert len(w) == 3 and (w["p_holm"] >= w["p"] - 1e-12).all() and w["signifikan"].all()
    assert (w["p_holm"].diff().dropna() >= -1e-12).all()                 # Holm tidak menurun menurut urutan p


def test_wilcoxon_semua_seri_p_satu():
    tabel = pd.DataFrame({"A": np.ones(30), "B": np.ones(30)})
    w = stats.wilcoxon_holm(tabel)
    assert w.loc[0, "p"] == 1.0 and not w.loc[0, "signifikan"] and w.loc[0, "seri"] == 30


def test_median_per_masalah_run_tak_layak_terburuk():
    df = pd.DataFrame(dict(algoritma=["GA"] * 3, data=["d"] * 3, K=[2] * 3, R=[0.3, 0.4, 0.9], layak=[True, True, False]))
    assert stats.median_per_masalah(df).loc[("d", 2), "GA"] == pytest.approx(0.3)   # median(0.3, 0.4, -1)


@pytest.mark.skipif(not __import__("os").path.exists(eksperimen.BERKAS), reason="fase2_hasil.csv belum dibuat")
def test_integritas_berkas_fase2():
    df = pd.read_csv(eksperimen.BERKAS)
    assert len(df) == 4 * 30 * 30 and df.groupby(["data", "K"]).ngroups == 30
    assert (df["n_eval"] == config.BUDGET_EVAL).all() and set(df["seed"]) == set(config.SEEDS_FASE2)
    assert (df.groupby(["algoritma", "data", "K"])["seed"].nunique() == 30).all()
    assert (df["jumlah_saham"] <= df["K"]).all() and df["layak"].all()


@pytest.mark.skipif(not __import__("os").path.exists(eksperimen.BERKAS), reason="fase2_hasil.csv belum dibuat")
def test_fase2_tidak_melampaui_optimum_k2():
    from algorithms.brute_force_k2 import brute_force_k2
    from data import buat_problem, muat_jendela
    df = pd.read_csv(eksperimen.BERKAS)
    for d in ("kasus_tetap", "jendela_3", "jendela_5"):
        r_star = brute_force_k2(buat_problem(muat_jendela(d), 2))["kas bebas"][0]
        assert df[(df.data == d) & (df.K == 2)]["R"].max() <= r_star + 1e-9


def test_kolom_metrik_tambahan_dan_shrinkage():
    r = eksperimen._satu_run(("GA", "jendela_3", 5, 4, 2000, {}))
    for kolom in ("sharpe_uji", "cvar95_uji", "shrinkage"):
        assert kolom in r
    assert r["shrinkage"] == pytest.approx(r["return_uji_tahunan"] - r["R"])


def _df_latih():
    baris = []
    for a, rs in {"GA": [0.30, 0.29, 0.30, 0.20], "DE": [0.30, 0.30, 0.30, 0.30]}.items():
        for s, R in enumerate(rs):
            baris.append(dict(algoritma=a, data="d", K=2, seed=s, R=R, layak=True, eval_terbaik=100 * (s + 1), detik=1.0))
    return pd.DataFrame(baris)


def test_acuan_dan_ringkasan_latih():
    df = _df_latih()
    ac = stats.acuan_masalah(df, brute=False)
    assert ac.loc[("d", 2)] == pytest.approx(0.30)                          # terbaik yang diketahui
    ac2 = stats.acuan_masalah(df, brute=False, brute_force={("d", 2): 0.31})
    assert ac2.loc[("d", 2)] == pytest.approx(0.31)                         # optimum brute force menggantikan
    r = stats.ringkasan_latih(df, ac).set_index("algoritma")
    assert r.loc["DE", "sukses"] == 1.0 and r.loc["DE", "selisih"] == pytest.approx(0.0)
    assert r.loc["GA", "sukses"] == pytest.approx(0.5)                      # ambang 0,297: hanya 0,30 dan 0,30 yang lolos
    assert r.loc["GA", "median"] == pytest.approx(0.295) and r.loc["GA", "layak"] == 1.0
    assert {"mean", "std", "min", "max", "eval_terbaik", "detik"} <= set(r.columns)
