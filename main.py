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
from scrappingdata import hitung_mu_cov, pisah_latih_uji  # noqa: E402


def jalankan(data_dir=config.DATA_DIR, hasil_dir=config.HASIL_DIR,
             n_runs=config.N_RUNS, pop_size=config.POP_SIZE,
             generations=config.GENERATIONS, train_end=config.TRAIN_END):
    os.makedirs(hasil_dir, exist_ok=True)
    rets = pd.read_csv(os.path.join(data_dir, "log_return.csv"),
                       index_col=0, parse_dates=True)
    tickers = list(rets.columns)
    # mu/Sigma dihitung ulang dari data latih sesuai train_end (bukan dari
    # mu_cov_train.npz) supaya data uji tidak bocor bila TRAIN_END diubah.
    train, test = pisah_latih_uji(rets, train_end)
    mu, cov = hitung_mu_cov(train)
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
    if fit.max() <= s_qp + 1e-6:
        print("Catatan: GA mencapai optimum SLSQP di data latih, tidak melampauinya.")

    # Tahap 7: evaluasi data uji
    bobot = {"GA-terbaik": best.weights, "GA-median": med.weights,
             "SLSQP": w_qp, "1/N": w_eq}
    tabel = ev.bandingkan(test, bobot, rf)
    print("\nMetrik data uji:\n", tabel.round(4))

    sharpe_uji = np.array([ev.hitung_metrik(test, r.weights, rf)["sharpe"]
                           for r in results])
    ringkasan = {"sharpe_uji_ga_mean": float(sharpe_uji.mean()),
                 "sharpe_uji_ga_std": float(sharpe_uji.std(ddof=1)) if n_runs > 1 else 0.0,
                 "sharpe_latih_ga_terbaik": float(fit.max()),
                 "sharpe_latih_slsqp": float(s_qp)}
    print(f"Sebaran Sharpe uji GA ({n_runs} run): "
          f"{ringkasan['sharpe_uji_ga_mean']:.3f} +/- {ringkasan['sharpe_uji_ga_std']:.3f} "
          f"(rentang {sharpe_uji.min():.3f}..{sharpe_uji.max():.3f}). Selisih GA vs "
          f"SLSQP di bawah sebaran antar-run tidak bermakna; data uji hanya satu lintasan.")

    pd.DataFrame(bobot, index=tickers).to_csv(os.path.join(hasil_dir, "bobot.csv"))
    tabel.to_csv(os.path.join(hasil_dir, "metrik_uji.csv"))
    pd.DataFrame({
        "seed": [r.seed for r in results],
        "sharpe_latih": fit,
        "sharpe_uji": sharpe_uji,
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
    return {"tabel": tabel, "bobot": bobot, "ga_results": results,
            "mu": mu, "cov": cov, "ringkasan": ringkasan}


def main():
    ada = all(os.path.exists(os.path.join(config.DATA_DIR, f))
              for f in ("log_return.csv", "mu_cov_train.npz"))
    if "--unduh" in sys.argv or not ada:
        import scrappingdata
        scrappingdata.main()
    jalankan()


if __name__ == "__main__":
    main()
