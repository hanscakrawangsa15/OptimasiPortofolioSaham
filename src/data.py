"""Pemuatan jendela data (data/windows/*.npz) dan pembentukan Problem."""
import os

import numpy as np

import config
from problem import Problem

NAMA_DATA = ["kasus_tetap"] + [f"jendela_{i}" for i in range(1, config.N_JENDELA + 1)]


def muat_jendela(nama):
    """Baca data/windows/<nama>.npz: statistik latih, harga beli, sigma_max, dan data uji."""
    z = np.load(os.path.join(config.DATA_DIR, "windows", f"{nama}.npz"), allow_pickle=False)
    return dict(
        tickers=[str(t) for t in z["tickers"]],
        tanggal_latih_awal=str(z["tanggal_latih_awal"]), tanggal_latih_akhir=str(z["tanggal_latih_akhir"]),
        tanggal_uji_awal=str(z["tanggal_uji_awal"]), tanggal_uji_akhir=str(z["tanggal_uji_akhir"]),
        mu=z["mu"], cov=z["cov"], harga_beli=z["harga_beli"],
        return_uji=z["return_uji"], tanggal_uji=[str(t) for t in z["tanggal_uji"]],   # hanya evaluasi; TIDAK masuk Problem
        vol_benchmark=float(z["vol_benchmark"]),
        sigma_max={float(m): float(v) for m, v in zip(z["sigma_mult"], z["sigma_max"])})


def buat_problem(w, K, D=config.D_UTAMA, mult=1.5, label=""):
    return Problem(
        mu=w["mu"], cov=w["cov"], price=w["harga_beli"], D=D, K=K,
        sigma_max=w["sigma_max"][mult],
        cap=config.MAX_WEIGHT if K >= config.CAP_BERLAKU_MULAI_K else 1.0,
        lot=config.LOT_SIZE, rf=config.RISK_FREE,
        tickers=tuple(t.replace(".JK", "") for t in w["tickers"]), label=label)
