"""Jendela walk-forward: latih 504 hari, uji 126 hari, geser 126 hari.

Semua statistik (mu, cov, vol_benchmark, sigma_max) hanya memakai baris latih.
Return = RETURN SEDERHANA. harga_beli = harga RAW pada hari terakhir latih.
"""
import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def jumlah_jendela(n_return, train, test, step):
    """floor((n_return - train - test) / step) + 1, minimal 0."""
    return max(0, (n_return - train - test) // step + 1)


def _cek_masukan(return_df, harga_raw_df):
    if list(return_df.columns) != list(harga_raw_df.columns):
        raise ValueError("Urutan/nama kolom return_df dan harga_raw_df berbeda")
    hilang = return_df.index.difference(harga_raw_df.index)
    if len(hilang):
        raise ValueError(f"{len(hilang)} tanggal return tidak ada di harga_raw_df")


def _bangun(return_df, harga_raw_df, i_latih_awal, i_latih_akhir, i_uji_akhir, id_):
    """Satu jendela dari posisi baris (inklusif i_latih_akhir, i_uji_akhir)."""
    latih = return_df.iloc[i_latih_awal:i_latih_akhir + 1]
    uji = return_df.iloc[i_latih_akhir + 1:i_uji_akhir + 1]
    n = latih.shape[1]
    td = config.TRADING_DAYS

    mu = latih.values.mean(axis=0) * td
    cov = np.cov(latih.values.T, ddof=1) * td
    eq = np.full(n, 1.0 / n)
    vol = float(np.sqrt(eq @ cov @ eq))
    tgl_akhir_latih = latih.index[-1]
    d = lambda t: str(pd.Timestamp(t).date())  # noqa: E731
    return {
        "id": id_,
        "tickers": list(return_df.columns),
        "tanggal_latih_awal": d(latih.index[0]),
        "tanggal_latih_akhir": d(tgl_akhir_latih),
        "tanggal_uji_awal": d(uji.index[0]),
        "tanggal_uji_akhir": d(uji.index[-1]),
        "n_latih": len(latih),
        "n_uji": len(uji),
        "mu": mu,
        "cov": cov,
        "eig_min": float(np.linalg.eigvalsh((cov + cov.T) / 2).min()),
        "vol_benchmark": vol,
        "sigma_max": {m: m * vol for m in config.SIGMA_MULT},
        "harga_beli": harga_raw_df.loc[tgl_akhir_latih].values.astype(float),
        "return_uji": uji.values.copy(),
        "tanggal_uji": [d(t) for t in uji.index],
    }


def buat_jendela(return_df, harga_raw_df, train=config.WINDOW_TRAIN,
                 test=config.WINDOW_TEST, step=config.WINDOW_STEP):
    _cek_masukan(return_df, harga_raw_df)
    n = jumlah_jendela(len(return_df), train, test, step)
    return [_bangun(return_df, harga_raw_df, k * step, k * step + train - 1,
                    k * step + train + test - 1, k + 1) for k in range(n)]


def kasus_tetap(return_df, harga_raw_df, train_end=config.TRAIN_END):
    """Satu kasus: latih dari awal data s.d. train_end, uji = semua hari sesudahnya."""
    _cek_masukan(return_df, harga_raw_df)
    n_latih = int((return_df.index <= pd.Timestamp(train_end)).sum())
    n_uji = len(return_df) - n_latih
    if n_latih == 0 or n_uji == 0:
        raise ValueError(f"Data latih ({n_latih} hari) atau uji ({n_uji} hari) "
                         f"kosong untuk train_end={train_end}")
    return _bangun(return_df, harga_raw_df, 0, n_latih - 1, len(return_df) - 1, "tetap")


_SKALAR = ["tanggal_latih_awal", "tanggal_latih_akhir", "tanggal_uji_awal",
           "tanggal_uji_akhir"]


def _simpan_npz(path, w):
    np.savez(
        path,
        id=np.array(str(w["id"])),
        tickers=np.array(w["tickers"]),
        **{k: np.array(w[k]) for k in _SKALAR},
        n_latih=np.array(w["n_latih"]), n_uji=np.array(w["n_uji"]),
        mu=w["mu"], cov=w["cov"], eig_min=np.array(w["eig_min"]),
        vol_benchmark=np.array(w["vol_benchmark"]),
        sigma_mult=np.array(list(w["sigma_max"].keys())),
        sigma_max=np.array(list(w["sigma_max"].values())),
        harga_beli=w["harga_beli"], return_uji=w["return_uji"],
        tanggal_uji=np.array(w["tanggal_uji"]),
    )


def muat_jendela(path):
    z = np.load(path, allow_pickle=False)
    w = {k: str(z[k]) for k in _SKALAR}
    w.update(
        id=str(z["id"]), tickers=[str(t) for t in z["tickers"]],
        n_latih=int(z["n_latih"]), n_uji=int(z["n_uji"]),
        mu=z["mu"], cov=z["cov"], eig_min=float(z["eig_min"]),
        vol_benchmark=float(z["vol_benchmark"]),
        sigma_max={float(m): float(v) for m, v in zip(z["sigma_mult"], z["sigma_max"])},
        harga_beli=z["harga_beli"], return_uji=z["return_uji"],
        tanggal_uji=[str(t) for t in z["tanggal_uji"]],
    )
    return w


def simpan_jendela(jendela, kasus, out_dir):
    """jendela_{k}.npz (k mulai 1), kasus_tetap.npz, dan ringkasan_jendela.csv.

    ringkasan_jendela.csv hanya berisi jendela walk-forward (satu baris per jendela).
    """
    os.makedirs(out_dir, exist_ok=True)
    for w in jendela:
        _simpan_npz(os.path.join(out_dir, f"jendela_{w['id']}.npz"), w)
    _simpan_npz(os.path.join(out_dir, "kasus_tetap.npz"), kasus)

    kolom = ["jendela", "latih_awal", "latih_akhir", "uji_awal", "uji_akhir",
             "vol_benchmark"] + [f"sigma_max_{m}" for m in config.SIGMA_MULT]
    with open(os.path.join(out_dir, "ringkasan_jendela.csv"), "w", newline="",
              encoding="utf-8") as f:
        tulis = csv.writer(f)
        tulis.writerow(kolom)
        for w in jendela:
            tulis.writerow([w["id"], w["tanggal_latih_awal"], w["tanggal_latih_akhir"],
                            w["tanggal_uji_awal"], w["tanggal_uji_akhir"],
                            repr(w["vol_benchmark"])]
                           + [repr(w["sigma_max"][m]) for m in config.SIGMA_MULT])
