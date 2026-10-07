"""Pipeline: data -> baseline -> GA (30 run) -> evaluasi uji -> visual.

Jalankan dari root proyek:  python3 main.py            (pakai data/ yang ada)
                            python3 main.py --unduh    (unduh ulang data)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import config  # noqa: E402
import evaluasi as ev  # noqa: E402
import visualisasi as vz  # noqa: E402
from baseline import equal_weight, max_sharpe_slsqp  # noqa: E402
from GeneticAlgorithm import pilih_median, pilih_terbaik, run_ga_batch  # noqa: E402
from portofolio import sharpe_ratio  # noqa: E402


def jalankan(data_dir=config.DATA_DIR, hasil_dir=config.HASIL_DIR,
             n_runs=config.N_RUNS, pop_size=config.POP_SIZE,
             generations=config.GENERATIONS, train_end=config.TRAIN_END):
    os.makedirs(hasil_dir, exist_ok=True)
    rets = pd.read_csv(os.path.join(data_dir, "log_return.csv"),
                       index_col=0, parse_dates=True)
    z = np.load(os.path.join(data_dir, "mu_cov_train.npz"), allow_pickle=True)
    mu, cov, tickers = z["mu"], z["cov"], list(z["tickers"])
    rets = rets[tickers]
    test = rets.loc[rets.index > pd.Timestamp(train_end)]
    if test.empty:
        raise ValueError("Data uji kosong; periksa TRAIN_END dan data/")
    cap, rf = config.MAX_WEIGHT, config.RISK_FREE

    # Tahap 6 (lebih dulu): baseline
    w_qp = max_sharpe_slsqp(mu, cov, rf, cap)
    w_eq = equal_weight(len(mu), cap)
    s_qp = sharpe_ratio(w_qp, mu, cov, rf)
    print(f"Baseline latih: SLSQP Sharpe={s_qp:.3f} | "
          f"1/N Sharpe={sharpe_ratio(w_eq, mu, cov, rf):.3f}")

    # Tahap 5: GA, 30 run dengan seed berbeda
    results = run_ga_batch(mu, cov, cap, rf, seeds=range(n_runs),
                           pop_size=pop_size, generations=generations)
    best, med = pilih_terbaik(results), pilih_median(results)
    fit = np.array([r.fitness for r in results])
    print(f"GA latih ({n_runs} run): terbaik={fit.max():.3f} "
          f"median={np.median(fit):.3f} std={fit.std():.4f}")

    # Tahap 7: evaluasi data uji
    bobot = {"GA-terbaik": best.weights, "GA-median": med.weights,
             "SLSQP": w_qp, "1/N": w_eq}
    tabel = ev.bandingkan(test, bobot, rf)
    print("\nMetrik data uji:\n", tabel.round(4))

    pd.DataFrame(bobot, index=tickers).to_csv(os.path.join(hasil_dir, "bobot.csv"))
    tabel.to_csv(os.path.join(hasil_dir, "metrik_uji.csv"))
    pd.DataFrame({
        "seed": [r.seed for r in results],
        "sharpe_latih": fit,
        "sharpe_uji": [ev.hitung_metrik(test, r.weights, rf)["sharpe"] for r in results],
    }).to_csv(os.path.join(hasil_dir, "ga_semua_run.csv"), index=False)

    # Tahap 8: enam visual
    p = lambda f: os.path.join(hasil_dir, f)  # noqa: E731
    kum = {n: ev.nilai_kumulatif(test, w)
           for n, w in {"GA": best.weights, "SLSQP": w_qp, "1/N": w_eq}.items()}
    tabel_viz = tabel.rename(index={"GA-terbaik": "GA"}).loc[["GA", "SLSQP", "1/N"]]
    vz.plot_konvergensi(results, p("1_konvergensi.png"))
    vz.plot_sebaran_sharpe(results, s_qp, p("2_sebaran_sharpe.png"))
    vz.plot_bobot(tickers, best.weights, w_qp, p("3_bobot.png"))
    vz.plot_kumulatif(kum, p("4_kumulatif.png"))
    vz.plot_metrik(tabel_viz, p("5_metrik.png"))
    vz.plot_drawdown({n: ev.drawdown_series(s) for n, s in kum.items()},
                     p("6_drawdown.png"))
    print(f"\nHasil tersimpan di {hasil_dir}")
    return {"tabel": tabel, "bobot": bobot, "ga_results": results}


def main():
    ada = all(os.path.exists(os.path.join(config.DATA_DIR, f))
              for f in ("log_return.csv", "mu_cov_train.npz"))
    if "--unduh" in sys.argv or not ada:
        import scrappingdata
        scrappingdata.main()
    jalankan()


if __name__ == "__main__":
    main()
