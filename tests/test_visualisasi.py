import os

import numpy as np
import pandas as pd

import visualisasi as vz
from GeneticAlgorithm import GAResult


def _results(k=4, g=10):
    rng = np.random.default_rng(0)
    out = []
    for s in range(k):
        best = np.sort(rng.random(g + 1))
        out.append(GAResult(s, np.full(12, 1 / 12), float(best[-1]), best, best - 0.1))
    return out


def _ok(path):
    assert os.path.exists(path) and os.path.getsize(path) > 1000


def test_semua_gambar_terbuat(tmp_path):
    res = _results()
    tick = [f"T{i}.JK" for i in range(12)]
    w1, w2 = np.full(12, 1 / 12), np.r_[np.full(6, 0.1), np.full(6, 0.4 / 6)]
    idx = pd.bdate_range("2025-10-01", periods=30)
    kum = {n: pd.Series(1 + np.arange(30) * 0.001 * (i + 1), index=idx)
           for i, n in enumerate(["GA", "SLSQP", "1/N"])}
    dd = {n: s / s.cummax() - 1 for n, s in kum.items()}
    tabel = pd.DataFrame({"return_tahunan": [0.1, 0.12, 0.09],
                          "volatilitas": [0.2, 0.19, 0.21],
                          "sharpe": [0.25, 0.37, 0.19],
                          "max_drawdown": [-0.1, -0.08, -0.12]},
                         index=["GA", "SLSQP", "1/N"])
    for f, args in [
        (vz.plot_konvergensi, (res,)),
        (vz.plot_sebaran_sharpe, (res, 0.9)),
        (vz.plot_bobot, (tick, w1, w2)),
        (vz.plot_kumulatif, (kum,)),
        (vz.plot_metrik, (tabel,)),
        (vz.plot_drawdown, (dd,)),
    ]:
        p = str(tmp_path / f"{f.__name__}.png")
        assert f(*args, p) == p
        _ok(p)
