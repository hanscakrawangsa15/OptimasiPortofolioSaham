"""Evaluasi out-of-sample: bobot dari data latih, dinilai di data uji.

Return portofolio dihitung dari return sederhana (expm1 dari log return) dengan
bobot konstan per hari; metrik tahunan memakai 252 hari bursa.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_daily_returns(rets, w):
    return pd.Series(np.expm1(rets.values) @ np.asarray(w), index=rets.index)


def nilai_kumulatif(rets, w):
    return (1 + portfolio_daily_returns(rets, w)).cumprod()


def drawdown_series(cum):
    return cum / cum.cummax() - 1


def max_drawdown(cum):
    return float(drawdown_series(cum).min())


def hitung_metrik(rets, w, rf, trading_days=TRADING_DAYS):
    r = portfolio_daily_returns(rets, w)
    ret = float(r.mean() * trading_days)
    vol = float(r.std(ddof=1) * np.sqrt(trading_days))
    sharpe = (ret - rf) / vol if vol > 1e-12 else 0.0
    return {"return_tahunan": ret, "volatilitas": vol, "sharpe": sharpe,
            "max_drawdown": max_drawdown((1 + r).cumprod())}


def bandingkan(rets, bobot, rf):
    baris = {nama: hitung_metrik(rets, w, rf) for nama, w in bobot.items()}
    return pd.DataFrame.from_dict(baris, orient="index")
