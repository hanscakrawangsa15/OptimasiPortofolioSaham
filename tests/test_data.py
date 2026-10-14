"""Tes data dan jendela walk-forward.

Bagian 1 memakai data sintetis (tanpa jaringan). Bagian 2 memeriksa berkas
nyata di data/ yang dibuat oleh `python3 src/scrappingdata.py`.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

import config
import laporan_data as ld
import scrappingdata as sd
import walkforward as wf

TRAIN, TEST, STEP = config.WINDOW_TRAIN, config.WINDOW_TEST, config.WINDOW_STEP


# ---------------------------------------------------------------- sintetis
def _sintetis(T=900, n=6, seed=0):
    """T baris return sederhana, n ticker; raw = adj x faktor tetap per ticker."""
    idx = pd.bdate_range("2022-01-03", periods=T + 1)
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.015, (T + 1, n))
    r[0] = 0.0
    kol = [f"S{i}.JK" for i in range(n)]
    adj = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=kol)
    raw = adj * rng.uniform(1.0, 1.3, n)
    ret = adj.pct_change().dropna()
    return ret, raw, adj


@pytest.mark.parametrize("T", [629, 630, 755, 756, 900, 1167])
def test_jumlah_jendela_sesuai_rumus(T):
    ret, raw, _ = _sintetis(T=T)
    jendela = wf.buat_jendela(ret, raw, TRAIN, TEST, STEP)
    assert len(jendela) == (T - TRAIN - TEST) // STEP + 1
    assert len(jendela) == wf.jumlah_jendela(T, TRAIN, TEST, STEP)


def test_tanggal_jendela_berurutan_dan_ukuran_benar():
    ret, raw, _ = _sintetis(T=900)
    for k, w in enumerate(wf.buat_jendela(ret, raw, TRAIN, TEST, STEP)):
        assert w["tanggal_latih_akhir"] < w["tanggal_uji_awal"]
        sub_latih = ret.loc[w["tanggal_latih_awal"]:w["tanggal_latih_akhir"]]
        sub_uji = ret.loc[w["tanggal_uji_awal"]:w["tanggal_uji_akhir"]]
        assert len(sub_latih) == TRAIN and len(sub_uji) == TEST
        # uji menempel persis setelah latih, dan geser = STEP
        assert ret.index.get_loc(sub_latih.index[0]) == k * STEP
        assert (ret.index.get_loc(sub_uji.index[0])
                == ret.index.get_loc(sub_latih.index[-1]) + 1)
        assert w["return_uji"].shape == (TEST, ret.shape[1])
        assert np.allclose(w["return_uji"], sub_uji.values)


def test_mu_cov_dari_baris_latih_saja_dihitung_ulang_manual():
    ret, raw, _ = _sintetis(T=900)
    for w in wf.buat_jendela(ret, raw, TRAIN, TEST, STEP):
        latih = ret.loc[w["tanggal_latih_awal"]:w["tanggal_latih_akhir"]]
        assert np.allclose(w["mu"], latih.values.mean(axis=0) * 252)
        assert np.allclose(w["cov"], np.cov(latih.values.T, ddof=1) * 252)


def test_data_uji_diubah_tidak_mengubah_mu_cov_sigma_harga_beli():
    ret, raw, _ = _sintetis(T=900)
    asli = wf.buat_jendela(ret, raw, TRAIN, TEST, STEP)[0]
    ret2, raw2 = ret.copy(), raw.copy()
    ret2.iloc[TRAIN:] *= 10.0                      # semua baris setelah latih jendela 0
    raw2.iloc[TRAIN + 1:] *= 7.0                   # harga setelah hari terakhir latih
    ubah = wf.buat_jendela(ret2, raw2, TRAIN, TEST, STEP)[0]
    assert np.array_equal(asli["mu"], ubah["mu"])
    assert np.array_equal(asli["cov"], ubah["cov"])
    assert asli["sigma_max"] == ubah["sigma_max"]
    assert np.array_equal(asli["harga_beli"], ubah["harga_beli"])
    assert not np.allclose(asli["return_uji"], ubah["return_uji"])


def test_sigma_max_dan_vol_benchmark():
    ret, raw, _ = _sintetis(T=900)
    for w in wf.buat_jendela(ret, raw, TRAIN, TEST, STEP):
        n = len(w["tickers"])
        eq = np.full(n, 1.0 / n)
        vol = float(np.sqrt(eq @ w["cov"] @ eq))
        assert w["vol_benchmark"] == pytest.approx(vol)
        assert set(w["sigma_max"]) == set(config.SIGMA_MULT)
        for m in config.SIGMA_MULT:
            assert w["sigma_max"][m] == pytest.approx(m * vol)


def test_harga_beli_adalah_raw_pada_hari_terakhir_latih():
    ret, raw, _ = _sintetis(T=900)
    for w in wf.buat_jendela(ret, raw, TRAIN, TEST, STEP):
        assert np.array_equal(w["harga_beli"], raw.loc[w["tanggal_latih_akhir"]].values)


def test_kolom_raw_beda_urutan_ditolak():
    ret, raw, _ = _sintetis(T=700)
    with pytest.raises(ValueError, match="kolom"):
        wf.buat_jendela(ret, raw[raw.columns[::-1]], TRAIN, TEST, STEP)


def test_kasus_tetap_tanpa_kebocoran():
    ret, raw, _ = _sintetis(T=900)
    train_end = str(ret.index[699].date())
    k = wf.kasus_tetap(ret, raw, train_end)
    latih = ret.loc[:train_end]
    assert len(latih) == 700
    assert k["tanggal_latih_akhir"] == train_end
    assert k["tanggal_uji_awal"] > train_end
    assert k["return_uji"].shape[0] == len(ret) - 700
    assert np.allclose(k["mu"], latih.values.mean(axis=0) * 252)
    assert np.allclose(k["cov"], np.cov(latih.values.T, ddof=1) * 252)
    assert np.array_equal(k["harga_beli"], raw.loc[train_end].values)
    for m in config.SIGMA_MULT:
        assert k["sigma_max"][m] == pytest.approx(m * k["vol_benchmark"])


def test_kasus_tetap_uji_kosong_gagal_jelas():
    ret, raw, _ = _sintetis(T=700)
    with pytest.raises(ValueError, match="uji"):
        wf.kasus_tetap(ret, raw, "2099-01-01")


def test_simpan_dan_muat_jendela(tmp_path):
    ret, raw, _ = _sintetis(T=900)
    jendela = wf.buat_jendela(ret, raw, TRAIN, TEST, STEP)
    kasus = wf.kasus_tetap(ret, raw, str(ret.index[699].date()))
    wf.simpan_jendela(jendela, kasus, str(tmp_path))
    for k in range(1, len(jendela) + 1):
        assert (tmp_path / f"jendela_{k}.npz").exists()
    assert (tmp_path / "kasus_tetap.npz").exists()
    ring = pd.read_csv(tmp_path / "ringkasan_jendela.csv")
    assert len(ring) == len(jendela)
    assert {"jendela", "latih_awal", "latih_akhir", "uji_awal", "uji_akhir",
            "vol_benchmark", "sigma_max_1.5", "sigma_max_2.0",
            "sigma_max_2.5"} <= set(ring.columns)
    w2 = wf.muat_jendela(str(tmp_path / "jendela_1.npz"))
    assert np.array_equal(w2["mu"], jendela[0]["mu"])
    assert w2["sigma_max"] == jendela[0]["sigma_max"]
    assert w2["tickers"] == jendela[0]["tickers"]
    assert w2["tanggal_latih_akhir"] == jendela[0]["tanggal_latih_akhir"]


# ----------------------------------------------------------------- bersihkan
def _nan(df, kolom, a, b):
    df.iloc[a:b, df.columns.get_loc(kolom)] = np.nan


def _mentah():
    ret, raw, adj = _sintetis(T=300, n=5)
    adj = adj.copy()
    raw = raw.copy()
    _nan(adj, "S1.JK", None, 100)                       # IPO belakangan (33%)
    _nan(raw, "S1.JK", None, 100)
    _nan(adj, "S2.JK", None, None)                         # unduhan gagal
    _nan(raw, "S2.JK", None, None)
    _nan(adj, "S3.JK", 150, 152)                    # lubang 2 hari: di-ffill
    _nan(raw, "S3.JK", 150, 152)
    _nan(adj, "S4.JK", 200, 205)                    # lubang 5 hari: baris dibuang
    _nan(raw, "S4.JK", 200, 205)
    return adj, raw


def test_bersihkan_buang_ticker_dengan_alasan_dan_samakan_raw_adj():
    adj, raw = _mentah()
    a, r, dibuang, info = sd.bersihkan_data(adj, raw, config.MAX_MISSING)
    assert set(dibuang) == {"S1.JK", "S2.JK"}
    assert "unduh" in dibuang["S2.JK"].lower()
    assert "%" in dibuang["S1.JK"]
    assert list(a.columns) == list(r.columns) == ["S0.JK", "S3.JK", "S4.JK"]
    assert a.index.equals(r.index)
    assert not a.isna().any().any() and not r.isna().any().any()


def test_bersihkan_ffill_maks_3_hari_dan_lapor_hari_dibuang():
    adj, raw = _mentah()
    a, r, dibuang, info = sd.bersihkan_data(adj, raw, config.MAX_MISSING)
    assert adj.index[150] in a.index and adj.index[151] in a.index   # lubang 2 hari terisi
    tengah = set(info["hari_dibuang_tengah"])
    # lubang 5 hari: 3 hari pertama terisi ffill, 2 hari terakhir dibuang
    assert {adj.index[203], adj.index[204]} <= {pd.Timestamp(d) for d in tengah}
    assert adj.index[200] in a.index
    assert info["n_hari_dibuang_tengah"] == len(tengah)


# -------------------------------------------------------------------- anomali
def test_anomali_split_terdeteksi_dan_data_tidak_diubah():
    _, raw, adj = _sintetis(T=300, n=3)
    raw = raw.copy()
    tgl = raw.index[100]
    raw.loc[tgl:, "S0.JK"] *= 0.5            # raw turun 50% tanpa tercermin di adj
    raw0, adj0 = raw.copy(), adj.copy()
    an = ld.deteksi_anomali(raw, adj)
    assert len(an) == 1
    assert an.iloc[0]["ticker"] == "S0.JK" and pd.Timestamp(an.iloc[0]["tanggal"]) == tgl
    pd.testing.assert_frame_equal(raw, raw0)
    pd.testing.assert_frame_equal(adj, adj0)


def test_lonjakan_nyata_yang_tercermin_di_adj_tidak_dianggap_anomali():
    _, raw, adj = _sintetis(T=300, n=3)
    raw, adj = raw.copy(), adj.copy()
    tgl = raw.index[100]
    raw.loc[tgl:, "S1.JK"] *= 1.5
    adj.loc[tgl:, "S1.JK"] *= 1.5            # tercermin di keduanya
    assert len(ld.deteksi_anomali(raw, adj)) == 0


# ======================================================= berkas nyata di data/
def _butuh(*nama):
    for n in nama:
        p = os.path.join(config.DATA_DIR, n)
        if not os.path.exists(p):
            pytest.fail(f"{p} tidak ada; jalankan dulu: python3 src/scrappingdata.py")


@pytest.fixture(scope="module")
def nyata():
    _butuh("harga_close_raw.csv", "harga_adjclose.csv", "return_sederhana.csv",
           "metadata.json", "ticker_dibuang.txt", "laporan_data.md")
    baca = lambda f: pd.read_csv(os.path.join(config.DATA_DIR, f),  # noqa: E731
                                 index_col=0, parse_dates=True)
    return {"raw": baca("harga_close_raw.csv"), "adj": baca("harga_adjclose.csv"),
            "ret": baca("return_sederhana.csv")}


def test_raw_dan_adj_kolom_indeks_identik_tanpa_nan_dan_positif(nyata):
    raw, adj = nyata["raw"], nyata["adj"]
    assert list(raw.columns) == list(adj.columns)
    assert raw.index.equals(adj.index)
    assert not raw.isna().any().any() and not adj.isna().any().any()
    assert (raw > 0).all().all() and (adj > 0).all().all()


def test_return_sederhana_dapat_direkonstruksi_dari_adj(nyata):
    adj, ret = nyata["adj"], nyata["ret"]
    assert list(ret.columns) == list(adj.columns)
    assert ret.index.equals(adj.index[1:])
    rekonstruksi = (1 + ret).cumprod()
    rasio = adj.iloc[1:] / adj.iloc[0]
    assert np.allclose(rekonstruksi.values, rasio.values, rtol=1e-9)


def test_jendela_nyata_jumlah_konsisten_dan_tanpa_kebocoran(nyata):
    ret, raw = nyata["ret"], nyata["raw"]
    T = len(ret)
    jendela = wf.buat_jendela(ret, raw, TRAIN, TEST, STEP)
    assert len(jendela) == (T - TRAIN - TEST) // STEP + 1 >= 1
    for k, w in enumerate(jendela, start=1):
        assert w["tanggal_latih_akhir"] < w["tanggal_uji_awal"]
        latih = ret.loc[w["tanggal_latih_awal"]:w["tanggal_latih_akhir"]]
        assert np.allclose(w["mu"], latih.values.mean(axis=0) * 252)
        assert np.allclose(w["cov"], np.cov(latih.values.T, ddof=1) * 252)
        tersimpan = wf.muat_jendela(
            os.path.join(config.DATA_DIR, "windows", f"jendela_{k}.npz"))
        assert np.allclose(tersimpan["mu"], w["mu"], rtol=1e-10, atol=0)
        assert np.allclose(tersimpan["cov"], w["cov"], rtol=1e-10, atol=0)
        assert np.allclose(tersimpan["return_uji"], w["return_uji"], rtol=1e-10, atol=0)
    ring = pd.read_csv(os.path.join(config.DATA_DIR, "windows", "ringkasan_jendela.csv"))
    assert len(ring) == len(jendela)


def _semua_jendela_tersimpan():
    d = os.path.join(config.DATA_DIR, "windows")
    _butuh(os.path.join("windows", "ringkasan_jendela.csv"),
           os.path.join("windows", "kasus_tetap.npz"))
    n = len(pd.read_csv(os.path.join(d, "ringkasan_jendela.csv")))
    out = [wf.muat_jendela(os.path.join(d, f"jendela_{k}.npz")) for k in range(1, n + 1)]
    return out + [wf.muat_jendela(os.path.join(d, "kasus_tetap.npz"))]


def test_cov_simetris_dan_semidefinit_positif(nyata):
    # Bila gagal, JANGAN diperbaiki diam-diam: laporkan ke pengguna.
    for w in _semua_jendela_tersimpan():
        c = w["cov"]
        assert np.allclose(c, c.T, atol=1e-12)
        eig_min = np.linalg.eigvalsh((c + c.T) / 2).min()
        assert eig_min >= -1e-10, f"cov tidak PSD (eig min {eig_min:.3e}) pada {w['id']}"


def test_sigma_max_nyata_sama_dengan_kelipatan_vol_benchmark(nyata):
    for w in _semua_jendela_tersimpan():
        n = len(w["tickers"])
        eq = np.full(n, 1.0 / n)
        vol = float(np.sqrt(eq @ w["cov"] @ eq))
        assert w["vol_benchmark"] == pytest.approx(vol)
        for m in config.SIGMA_MULT:
            assert w["sigma_max"][m] == pytest.approx(m * vol)


def test_harga_beli_nyata_sama_dengan_raw_pada_tanggal_tercatat(nyata):
    raw = nyata["raw"]
    for w in _semua_jendela_tersimpan():
        assert w["tickers"] == list(raw.columns)
        assert np.array_equal(w["harga_beli"], raw.loc[w["tanggal_latih_akhir"]].values)


def test_kasus_tetap_nyata_memakai_train_end_config(nyata):
    k = wf.muat_jendela(os.path.join(config.DATA_DIR, "windows", "kasus_tetap.npz"))
    assert k["tanggal_latih_akhir"] <= config.TRAIN_END < k["tanggal_uji_awal"]


def test_anomali_nyata_hanya_dilaporkan_bukan_dihapus(nyata):
    an = ld.deteksi_anomali(nyata["raw"], nyata["adj"])
    assert {"tanggal", "ticker", "ret_raw", "ret_adj"} <= set(an.columns)
    if len(an):
        print("\nANOMALI (raw berubah >35% tanpa tercermin di adj):\n", an.to_string())
    # data tidak boleh sudah dihapus: baris dan kolom raw tetap utuh
    assert nyata["raw"].shape == nyata["adj"].shape


def test_metadata_dan_laporan_ada(nyata):
    with open(os.path.join(config.DATA_DIR, "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    assert {"waktu_unduh", "versi_yfinance", "tanggal_awal", "tanggal_akhir",
            "n_ticker", "n_hari"} <= set(meta)
    assert meta["n_ticker"] == nyata["raw"].shape[1]
    assert meta["n_hari"] == len(nyata["raw"])
    with open(os.path.join(config.DATA_DIR, "laporan_data.md"), encoding="utf-8") as f:
        isi = f.read()
    for judul in ["Ringkasan data", "Jendela walk-forward", "Kelayakan lot"]:
        assert judul in isi


# --------------------------------------------- split sesudah tanggal beli
def test_split_setelah_beli_terdeteksi_per_jendela():
    jendela = [{"id": 1, "tanggal_latih_akhir": "2023-12-14"},
               {"id": 2, "tanggal_latih_akhir": "2025-01-09"}]
    splits = pd.DataFrame({"tanggal": ["2023-04-04", "2024-10-14"],
                           "ticker": ["BMRI.JK", "ISAT.JK"], "rasio": [2.0, 4.0]})
    out = ld.split_setelah_beli(jendela, splits)
    assert len(out) == 1                       # hanya ISAT x jendela 1
    assert out.iloc[0]["jendela"] == 1 and out.iloc[0]["ticker"] == "ISAT.JK"
    assert out.iloc[0]["rasio"] == 4.0


def test_ticker_penyebab_awal_dilaporkan():
    adj, raw = _mentah()
    adj2 = adj.copy()
    _nan(adj2, "S0.JK", None, 3)               # S0 baru mulai di hari ke-4
    _nan(raw, "S0.JK", None, 3)
    _, _, _, info = sd.bersihkan_data(adj2, raw, config.MAX_MISSING)
    assert info["ticker_penyebab_awal"] == ["S0.JK"]
