"""Menyiapkan data untuk GA alokasi portofolio LQ45.

Jalankan dari root proyek:  python3 src/scrappingdata.py
Keluaran di data/: harga_adjclose.csv, log_return.csv, mu_cov_train.npz
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def unduh_harga(tickers, start, end):
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    return raw["Close"].copy()


def bersihkan_harga(prices, max_missing):
    """Buang ticker dengan data kosong > max_missing SEBELUM dropna(),
    supaya satu ticker IPO belakangan tidak memangkas seluruh tanggal."""
    kosong = prices.isna().mean()
    dibuang = [c for c in prices.columns if kosong[c] > max_missing]
    bersih = prices.drop(columns=dibuang).ffill(limit=3).dropna()
    return bersih, dibuang


def hitung_log_return(prices):
    return np.log(prices / prices.shift(1)).dropna()


def pisah_latih_uji(rets, train_end):
    train = rets.loc[rets.index <= pd.Timestamp(train_end)]
    test = rets.loc[rets.index > pd.Timestamp(train_end)]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Data latih ({len(train)} hari) atau data uji ({len(test)} hari) "
            f"kosong untuk TRAIN_END={train_end}")
    return train, test


def hitung_mu_cov(train, trading_days=config.TRADING_DAYS):
    """Return dan kovarians tahunan dari data latih (input fitness GA)."""
    return train.mean().values * trading_days, train.cov().values * trading_days


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    mentah = unduh_harga(config.TICKERS, config.START, config.END)
    prices, dibuang = bersihkan_harga(mentah, config.MAX_MISSING)
    print(f"Ticker dibuang ({len(dibuang)}): {dibuang}")
    print(f"Ticker dipakai: {prices.shape[1]} | hari: {len(prices)} "
          f"({prices.index[0].date()} .. {prices.index[-1].date()})")

    min_saham = math.ceil(1 / config.MAX_WEIGHT)
    if prices.shape[1] < min_saham:
        sys.exit(f"Saham tersisa {prices.shape[1]} < {min_saham}; "
                 f"cap {config.MAX_WEIGHT} tidak layak. Tambah ticker.")

    rets = hitung_log_return(prices)
    train, test = pisah_latih_uji(rets, config.TRAIN_END)
    print(f"Latih: {len(train)} hari | Uji: {len(test)} hari")
    mu, cov = hitung_mu_cov(train)

    prices.to_csv(os.path.join(config.DATA_DIR, "harga_adjclose.csv"))
    rets.to_csv(os.path.join(config.DATA_DIR, "log_return.csv"))
    np.savez(os.path.join(config.DATA_DIR, "mu_cov_train.npz"),
             mu=mu, cov=cov, tickers=np.array(train.columns))
    print("Selesai. File tersimpan di data/.")


if __name__ == "__main__":
    main()
