import numpy as np
import pandas as pd
import pytest

import evaluasi as ev


def _rets(vals):
    idx = pd.bdate_range("2025-01-01", periods=len(vals))
    return pd.DataFrame(vals, index=idx, columns=["A", "B"])


def test_return_harian_pakai_expm1():
    rets = _rets([[np.log(1.10), np.log(1.00)]])
    r = ev.portfolio_daily_returns(rets, np.array([0.5, 0.5]))
    assert r.iloc[0] == pytest.approx(0.05)


def test_max_drawdown_manual():
    cum = pd.Series([1.0, 1.2, 0.9, 1.0, 0.6])
    assert ev.max_drawdown(cum) == pytest.approx(0.6 / 1.2 - 1)
    dd = ev.drawdown_series(cum)
    assert dd.iloc[1] == 0 and dd.min() == pytest.approx(-0.5)


def test_metrik_konstan_vol_nol_tidak_nan():
    rets = _rets(np.zeros((50, 2)))
    m = ev.hitung_metrik(rets, np.array([0.5, 0.5]), rf=0.05)
    assert m["volatilitas"] == 0
    assert np.isfinite(m["sharpe"])
    assert m["max_drawdown"] == 0


def test_metrik_dan_bandingkan():
    rng = np.random.default_rng(0)
    rets = _rets(rng.normal(0.0005, 0.01, (200, 2)))
    w = np.array([0.7, 0.3])
    m = ev.hitung_metrik(rets, w, rf=0.05)
    r = ev.portfolio_daily_returns(rets, w)
    assert m["return_tahunan"] == pytest.approx(r.mean() * 252)
    assert m["volatilitas"] == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert m["sharpe"] == pytest.approx((m["return_tahunan"] - 0.05) / m["volatilitas"])
    tabel = ev.bandingkan(rets, {"X": w, "Y": np.array([0.5, 0.5])}, rf=0.05)
    assert list(tabel.index) == ["X", "Y"]
    assert set(tabel.columns) == {"return_tahunan", "volatilitas", "sharpe", "max_drawdown"}
    assert ev.nilai_kumulatif(rets, w).iloc[-1] == pytest.approx((1 + r).prod())
