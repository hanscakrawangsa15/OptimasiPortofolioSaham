"""Enam visual utama untuk laporan."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito; ketiganya lolos validate_palette (light): CVD dE 11, kontras >= 3:1
WARNA = {"GA": "#0072B2", "SLSQP": "#D55E00", "1/N": "#009E73"}
_DEF = list(WARNA.values())


def _warna(nama, i):
    return WARNA.get(nama, _DEF[i % len(_DEF)])


def _simpan(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_konvergensi(results, path):
    best = np.array([r.hist_best for r in results])
    mean = np.array([r.hist_mean for r in results])
    gen = np.arange(best.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(gen, best.min(0), best.max(0), color=WARNA["GA"], alpha=0.2,
                    label="Rentang terbaik (semua run)")
    ax.plot(gen, best.mean(0), color=WARNA["GA"], lw=2, label="Rata-rata terbaik")
    ax.plot(gen, mean.mean(0), color=WARNA["SLSQP"], lw=1.5, ls="--",
            label="Rata-rata populasi")
    ax.set(xlabel="Generasi", ylabel="Sharpe ratio (latih)",
           title=f"Konvergensi GA ({len(results)} run)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _simpan(fig, path)


def plot_sebaran_sharpe(results, sharpe_slsqp, path):
    fit = [r.fitness for r in results]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(fit, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=WARNA["GA"], alpha=0.4))
    ax.scatter(np.ones(len(fit)), fit, color=WARNA["GA"], s=18, zorder=3)
    ax.axhline(sharpe_slsqp, color=WARNA["SLSQP"], ls="--", label="SLSQP")
    ax.set(xticks=[1], xticklabels=["GA"], ylabel="Sharpe ratio (latih)",
           title=f"Sebaran hasil {len(fit)} run GA")
    ax.legend()
    ax.grid(alpha=0.3)
    return _simpan(fig, path)


def plot_bobot(tickers, w_ga, w_qp, path):
    nama = [t.replace(".JK", "") for t in tickers]
    urut = np.argsort(np.maximum(w_ga, w_qp))
    y = np.arange(len(nama))
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(nama))))
    ax.barh(y - 0.2, np.asarray(w_ga)[urut] * 100, 0.4, color=WARNA["GA"], label="GA")
    ax.barh(y + 0.2, np.asarray(w_qp)[urut] * 100, 0.4, color=WARNA["SLSQP"], label="SLSQP")
    ax.set_yticks(y, [nama[i] for i in urut])
    ax.set(xlabel="Bobot (%)", title="Alokasi portofolio GA vs SLSQP")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    return _simpan(fig, path)


def plot_kumulatif(kum, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (nama, s) in enumerate(kum.items()):
        ax.plot(s.index, s.values, label=nama, color=_warna(nama, i), lw=2)
    ax.set(ylabel="Nilai portofolio (awal = 1)", title="Kinerja kumulatif data uji")
    ax.legend()
    ax.grid(alpha=0.3)
    return _simpan(fig, path)


def plot_metrik(tabel, path):
    kolom = [("return_tahunan", "Return tahunan", 100), ("volatilitas", "Volatilitas", 100),
             ("sharpe", "Sharpe", 1), ("max_drawdown", "Max drawdown", 100)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    for ax, (k, judul, skala) in zip(axes, kolom):
        nilai = tabel[k].values * skala
        ax.bar(tabel.index, nilai,
               color=[_warna(n, i) for i, n in enumerate(tabel.index)])
        ax.set_title(judul + (" (%)" if skala == 100 else ""))
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(axis="y", alpha=0.3)
    return _simpan(fig, path)


def plot_drawdown(dd, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (nama, s) in enumerate(dd.items()):
        ax.plot(s.index, s.values * 100, label=nama, color=_warna(nama, i), lw=1.8)
    ax.set(ylabel="Drawdown (%)", title="Drawdown data uji")
    ax.legend()
    ax.grid(alpha=0.3)
    return _simpan(fig, path)
