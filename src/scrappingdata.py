"""
Menyiapkan data untuk GA alokasi portofolio.
Jalankan di komputermu sendiri:
    pip install yfinance pandas numpy
    python siapkan_data_portofolio.py
Output: harga_adjclose.csv, log_return.csv, mu_cov_train.npz
"""
import numpy as np
import pandas as pd
import yfinance as yf

# ---- Pengaturan ----
TICKERS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "ISAT.JK",
    "INDF.JK", "JPFA.JK", "INKP.JK", "INCO.JK", "ADRO.JK",
]
START = "2021-10-01"
END = "2026-10-01"
TRAIN_END = "2025-09-30"   # data latih: START..TRAIN_END, data uji: sesudahnya
TRADING_DAYS = 252


def unduh_harga(tickers, start, end):
    """Unduh adjusted close, cek kualitas, dan bersihkan."""
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    prices = raw["Close"].copy()

    print("Jumlah baris:", len(prices))
    print("Data kosong per ticker:\n", prices.isna().sum(), "\n")

    dead = prices.columns[prices.isna().mean() > 0.10]
    if len(dead):
        print("PERINGATAN: data banyak kosong, pertimbangkan ganti ticker:",
              list(dead))
        prices = prices.drop(columns=dead)

    return prices.ffill(limit=3).dropna()


def hitung_log_return(prices):
    return np.log(prices / prices.shift(1)).dropna()


def pisah_latih_uji(rets, train_end):
    train = rets.loc[:train_end]
    test = rets.loc[train_end:].iloc[1:]
    return train, test


def hitung_mu_cov(train, trading_days=TRADING_DAYS):
    """Return dan kovarians tahunan, input untuk fungsi fitness GA."""
    mu = train.mean().values * trading_days
    cov = train.cov().values * trading_days
    return mu, cov


def main():
    prices = unduh_harga(TICKERS, START, END)
    prices.to_csv("harga_adjclose.csv")

    rets = hitung_log_return(prices)
    rets.to_csv("log_return.csv")

    train, test = pisah_latih_uji(rets, TRAIN_END)
    print(f"Latih: {len(train)} hari | Uji: {len(test)} hari")

    mu, cov = hitung_mu_cov(train)

    print("\nReturn tahunan (latih):")
    print(pd.Series(mu, index=train.columns).round(3))
    print("\nKorelasi (latih):")
    print(train.corr().round(2))

    np.savez("mu_cov_train.npz", mu=mu, cov=cov,
             tickers=np.array(train.columns))
    print("\nSelesai. File tersimpan di folder kerja.")


if __name__ == "__main__":
    main()