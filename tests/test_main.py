import os

import numpy as np
import pandas as pd

import main as m


def _siapkan(tmp_path, n=14, hari=400, train=300):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-02", periods=hari)
    tick = [f"S{i}.JK" for i in range(n)]
    rets = pd.DataFrame(rng.normal(0.0004, 0.012, (hari, n)), index=idx, columns=tick)
    data = tmp_path / "data"
    data.mkdir()
    rets.to_csv(data / "log_return.csv")
    tr = rets.iloc[:train]
    np.savez(data / "mu_cov_train.npz", mu=tr.mean().values * 252,
             cov=tr.cov().values * 252, tickers=np.array(tick))
    return str(data), str(tmp_path / "hasil"), str(idx[train - 1].date())


def test_pipeline_end_to_end(tmp_path):
    data, hasil, train_end = _siapkan(tmp_path)
    out = m.jalankan(data, hasil, n_runs=3, pop_size=40, generations=15,
                     train_end=train_end)
    assert set(out["tabel"].index) == {"GA-terbaik", "GA-median", "SLSQP", "1/N"}
    for w in out["bobot"].values():
        assert abs(w.sum() - 1) < 1e-6 and w.max() <= 0.10 + 1e-9
    for f in ["bobot.csv", "metrik_uji.csv", "ga_semua_run.csv",
              "1_konvergensi.png", "2_sebaran_sharpe.png", "3_bobot.png",
              "4_kumulatif.png", "5_metrik.png", "6_drawdown.png"]:
        assert os.path.getsize(os.path.join(hasil, f)) > 0


def test_mu_cov_dihitung_dari_train_end_bukan_dari_npz(tmp_path):
    # npz dibuat untuk train=300, tetapi train_end yang diminta hanya 250 hari:
    # mu/cov harus mengikuti train_end (tanpa kebocoran data uji).
    data, hasil, _ = _siapkan(tmp_path, train=300)
    rets = pd.read_csv(os.path.join(data, "log_return.csv"), index_col=0, parse_dates=True)
    train_end = str(rets.index[249].date())
    out = m.jalankan(data, hasil, n_runs=2, pop_size=20, generations=3,
                     train_end=train_end)
    assert np.allclose(out["mu"], rets.iloc[:250].mean().values * 252)
    assert np.allclose(out["cov"], rets.iloc[:250].cov().values * 252)


def test_ringkasan_sebaran_sharpe_uji(tmp_path, capsys):
    data, hasil, train_end = _siapkan(tmp_path)
    out = m.jalankan(data, hasil, n_runs=4, pop_size=20, generations=3,
                     train_end=train_end)
    r = out["ringkasan"]
    assert set(r) >= {"sharpe_uji_ga_mean", "sharpe_uji_ga_std",
                      "sharpe_latih_ga_terbaik", "sharpe_latih_slsqp"}
    assert np.isfinite(r["sharpe_uji_ga_std"])
    assert "Sebaran Sharpe uji GA" in capsys.readouterr().out
