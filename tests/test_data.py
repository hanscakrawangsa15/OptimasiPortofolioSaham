import numpy as np
import pandas as pd
import pytest

import scrappingdata as sd


def _harga():
    idx = pd.bdate_range("2024-01-01", periods=100)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (100, 3)), axis=0)),
        index=idx, columns=["A.JK", "B.JK", "C.JK"])
    return df


def test_bersihkan_buang_ticker_ipo_belakangan_tanpa_memangkas_tanggal():
    df = _harga()
    df.loc[df.index[:40], "C.JK"] = np.nan          # IPO belakangan
    bersih, dibuang = sd.bersihkan_harga(df, max_missing=0.05)
    assert dibuang == ["C.JK"]
    assert list(bersih.columns) == ["A.JK", "B.JK"]
    assert len(bersih) == 100                        # tanggal tidak terpangkas


def test_bersihkan_isi_lubang_kecil():
    df = _harga()
    df.iloc[50, 0] = np.nan
    bersih, dibuang = sd.bersihkan_harga(df, max_missing=0.05)
    assert dibuang == []
    assert not bersih.isna().any().any()


def test_log_return():
    df = pd.DataFrame({"A": [100.0, 110.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    r = sd.hitung_log_return(df)
    assert r.iloc[0, 0] == pytest.approx(np.log(1.1))


def test_split_tidak_bocor_dan_train_end_bukan_hari_bursa():
    rets = sd.hitung_log_return(_harga())
    # 2024-02-03 adalah Sabtu
    train, test = sd.pisah_latih_uji(rets, "2024-02-03")
    assert train.index.max() <= pd.Timestamp("2024-02-03")
    assert test.index.min() > pd.Timestamp("2024-02-03")
    assert len(train) + len(test) == len(rets)


def test_split_uji_kosong_gagal_jelas():
    rets = sd.hitung_log_return(_harga())
    with pytest.raises(ValueError, match="uji"):
        sd.pisah_latih_uji(rets, "2030-01-01")


def test_mu_cov_tahunan():
    rets = sd.hitung_log_return(_harga())
    mu, cov = sd.hitung_mu_cov(rets, trading_days=252)
    assert mu.shape == (3,) and cov.shape == (3, 3)
    assert np.allclose(mu, rets.mean().values * 252)
    assert np.allclose(cov, rets.cov().values * 252)
