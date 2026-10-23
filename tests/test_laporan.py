"""Tes bahan Bab 4: validasi constraint dan rincian portofolio."""
import numpy as np
import pandas as pd
import pytest

import eksperimen
import laporan
from data import buat_problem, muat_jendela


@pytest.fixture(scope="module")
def df_kecil():
    return eksperimen.jalankan(masalah=[("jendela_3", 2), ("kasus_tetap", 7)], seeds=(0, 1), budget=2000,
                               proses=1, kw={a: {} for a in eksperimen.ALGORITMA}, cetak=lambda *_: None)


def test_validasi_semua_terpenuhi(df_kecil):
    t = laporan.tabel_validasi(df_kecil)
    assert (t["Pelanggaran"] == 0).all() and (t["Run"] == 2).all()
    assert set(t["Algoritma"]) == set(eksperimen.ALGORITMA)


def test_periksa_run_mendeteksi_pelanggaran():
    P = buat_problem(muat_jendela("kasus_tetap"), 2)
    ok = np.zeros(P.n, dtype=int)
    ok[0] = 1
    assert all(laporan.periksa_run(ok, P).values())
    langgar = np.ones(P.n, dtype=int)                                   # 42 saham > K, dana melebihi modal
    hasil = laporan.periksa_run(langgar, P)
    assert not hasil["saham <= K"] and not hasil["dana <= modal"]
    P7 = buat_problem(muat_jendela("kasus_tetap"), 7)
    besar = np.zeros(P7.n, dtype=int)
    besar[int(np.argmin(P7.harga_lot))] = int(P7.D // P7.harga_lot.min())   # satu saham memakai hampir seluruh modal
    assert not laporan.periksa_run(besar, P7)["bobot <= cap"]


def test_rincian_menutup_modal_dan_return(df_kecil):
    baris = df_kecil[(df_kecil.algoritma == "GA") & (df_kecil.data == "kasus_tetap")].iloc[0]
    rinci, total = laporan.rincian_portofolio(baris)
    P = buat_problem(muat_jendela("kasus_tetap"), int(baris["K"]))
    assert total["dana_saham"] + total["sisa_kas"] == pytest.approx(P.D)
    assert rinci["Bobot (%)"].sum() / 100 + total["sisa_kas"] / P.D == pytest.approx(1.0)
    assert rinci["Kontribusi return (pp)"].sum() / 100 + total["kontribusi_kas"] == pytest.approx(baris["R"])


def test_terbaik_per_k_memilih_r_tertinggi_layak(df_kecil):
    t = laporan.terbaik_per_k(df_kecil, "DE", "kasus_tetap")
    sub = df_kecil[(df_kecil.algoritma == "DE") & (df_kecil.data == "kasus_tetap")]
    assert len(t) == 1 and t.iloc[0]["Expected return (%/tahun)"] == pytest.approx(100 * sub["R"].max())


def test_sebar_label_tidak_bertumpuk_dan_urutan_terjaga():
    y = np.array([4.54, 5.48, 5.27, 5.17, 5.21, 5.24])
    baru = laporan._sebar_label(y, 0.085)
    urut = np.argsort(y)
    assert (np.diff(baru[urut]) >= 0.085 - 1e-9).all()               # jarak minimal antar label
    assert baru[np.argmax(y)] >= baru[np.argmin(y)]
    assert np.mean(baru) == pytest.approx(np.mean(y))                # kelompok label tetap berpusat di data


def test_tabel_metrik_lengkap_sesuai_tabel_3_14(df_kecil):
    d = df_kecil.assign(data="kasus_tetap", K=7)           # satu sel (K=7) agar tabel lengkap dapat diperiksa
    lat = laporan.tabel_metrik_latih(d)
    assert {"Mean R (%)", "Median R (%)", "Std (pp)", "Min R (%)", "Maks R (%)", "Kelayakan (%)",
            "Selisih ke R acuan (pp)", "Keberhasilan gap <= 1% (%)", "Median evaluasi hingga terbaik", "Median waktu (dtk)"} <= set(lat.columns)
    uji = laporan.tabel_metrik_uji(d)
    assert {"Return uji (%/tahun)", "Volatilitas uji (%)", "Sharpe uji", "CVaR 95% harian (%)", "Max drawdown (%)",
            "Shrinkage (pp)", "Volatilitas uji <= sigma_max (%)"} <= set(uji.columns)
    assert len(lat) == len(uji) == len(eksperimen.ALGORITMA)
