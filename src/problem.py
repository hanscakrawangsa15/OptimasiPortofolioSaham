"""Definisi masalah optimasi dan hasil satu run."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Problem:
    mu: np.ndarray            # (N,) return tahunan (return sederhana)
    cov: np.ndarray           # (N,N) kovarians tahunan
    price: np.ndarray         # (N,) harga raw (tanpa penyesuaian) pada hari beli
    D: float                  # modal (Rp)
    K: int                    # batas jumlah saham
    sigma_max: float          # batas volatilitas portofolio
    cap: float = 1.0          # batas bobot per saham; 1.0 = tanpa batas (K < 7)
    lot: int = 100
    rf: float = 0.05
    tickers: tuple = ()
    label: str = ""

    @property
    def n(self) -> int:                 # jumlah saham N
        return len(self.mu)

    @property
    def n_genes(self) -> int:           # panjang kromosom = 2N+1
        return 2 * self.n + 1

    @property
    def harga_lot(self) -> np.ndarray:  # biaya 1 lot tiap saham (Rp)
        return self.lot * self.price

    @property
    def layak_beli(self) -> np.ndarray:  # saham yang 1 lot-nya muat dalam D x cap
        return self.harga_lot <= self.D * self.cap

    @property
    def vol(self) -> np.ndarray:        # volatilitas tahunan tiap saham
        return np.sqrt(np.diag(self.cov))


@dataclass
class Result:
    lots: np.ndarray          # (N,) lot terbaik (bilangan bulat)
    R: float                  # return ekspektasi (in-sample)
    sigma: float              # volatilitas portofolio
    viol: float               # max(0, sigma - sigma_max); 0 bila layak
    feasible: bool
    history: list             # [(n_eval, R terbaik yang layak)] tiap HISTORY_STEP evaluasi
    n_eval: int
    seconds: float
    seed: int
    x: np.ndarray = None      # kromosom terbaik
    n_eval_best: int = 0      # evaluasi ke-berapa solusi terbaik ditemukan
    status: str = ""
    trace: dict = None        # jejak per generasi/iterasi (opsional, untuk visualisasi proses)
