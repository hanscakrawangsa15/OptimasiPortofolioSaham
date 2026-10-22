"""Grafik matplotlib untuk GUI (GA #0072B2, PSO #D55E00)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WARNA = {"GA": "#0072B2", "PSO": "#D55E00"}


def warna(metode):
    return WARNA["GA"] if metode.startswith("GA") else WARNA["PSO"]


def _riwayat(runs):
    x = np.array([h[0] for h in runs[0].history])
    Y = np.array([[h[1] for h in r.history] for r in runs], float)
    return x, Y


def plot_konvergensi(hasil, R_opt=None):
    """Median R terbaik (garis) dan rentang kuartil (pita) terhadap jumlah evaluasi."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, runs in hasil.items():
        x, Y = _riwayat(runs)
        med, q1, q3 = (np.nanpercentile(Y, q, axis=0) for q in (50, 25, 75))
        ax.plot(x, med, color=warna(m), label=m)
        ax.fill_between(x, q1, q3, color=warna(m), alpha=0.2)
    if R_opt is not None:
        ax.axhline(R_opt, color="k", ls="--", lw=1, label="optimum sejati (brute force)")
    ax.set(xlabel="jumlah evaluasi", ylabel="R terbaik yang layak", title="Konvergensi")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


def plot_risiko_return(hasil, sigma_max):
    """Solusi akhir tiap seed pada bidang risiko-return."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, runs in hasil.items():
        ax.scatter([r.sigma for r in runs], [r.R for r in runs], color=warna(m), alpha=0.7, label=m)
    ax.axvline(sigma_max, color="k", ls="--", lw=1, label="batas risiko")
    ax.set(xlabel="risiko σ", ylabel="return ekspektasi R", title="Solusi akhir tiap seed")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


def plot_bobot(P, lots, judul=""):
    """Bar horizontal bobot tiap saham terpilih plus kas."""
    nilai = lots * P.harga_lot / P.D
    idx = np.where(lots > 0)[0]
    idx = idx[np.argsort(nilai[idx])]
    nama = [P.tickers[i] for i in idx] + ["KAS"]
    val = list(nilai[idx]) + [1.0 - nilai.sum()]
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * len(nama))))
    ax.barh(nama, val, color=["#009E73"] * len(idx) + ["#999999"])
    for i, v in enumerate(val):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=8)
    ax.set(xlabel="bobot dari modal", title=judul or "Bobot portofolio")
    ax.set_xlim(0, max(val) * 1.15 + 0.02)
    fig.tight_layout()
    return fig


def plot_box(hasil, R_opt=None):
    """Sebaran R akhir antar seed (run tak layak dilewati)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    nama = list(hasil)
    data = [[r.R for r in hasil[m] if r.feasible] for m in nama]
    bp = ax.boxplot(data, tick_labels=nama, patch_artist=True)
    for patch, m in zip(bp["boxes"], nama):
        patch.set_facecolor(warna(m)); patch.set_alpha(0.5)
    if R_opt is not None:
        ax.axhline(R_opt, color="k", ls="--", lw=1)
    ax.set(ylabel="R akhir", title="Sebaran antar seed")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_proses(trace, P, judul):
    """Proses satu run: sebaran populasi (risiko-return) pada awal, tengah, akhir, plus R terbaik per iterasi."""
    G = len(trace["R"])
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, g in zip(axs[:3], [0, G // 2, G - 1]):
        lay = trace["viol"][g] <= 0
        ax.scatter(trace["sigma"][g][~lay], trace["R"][g][~lay], s=12, color="#999999", label="melanggar")
        ax.scatter(trace["sigma"][g][lay], trace["R"][g][lay], s=12, color="#009E73", label="layak")
        ax.axvline(P.sigma_max, color="k", ls="--", lw=1)
        ax.set(title=f"iterasi {g}", xlabel="σ", ylabel="R"); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7)
    terbaik = np.array([np.nanmax(np.where(v <= 0, R, np.nan)) if (v <= 0).any() else np.nan
                        for R, v in zip(trace["R"], trace["viol"])])
    axs[3].plot(np.fmax.accumulate(np.nan_to_num(terbaik, nan=-np.inf)), color="#0072B2")
    axs[3].set(title="R terbaik s.d. iterasi ini", xlabel="iterasi"); axs[3].grid(alpha=0.3)
    fig.suptitle(judul)
    fig.tight_layout()
    return fig


def plot_uji(nilai, D):
    """Nilai portofolio harian pada data uji."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for nama, v in nilai.items():
        if nama.startswith("GA") or nama.startswith("PSO"):
            ax.plot(v, color=warna(nama), label=nama)
        else:
            ax.plot(v, color="#666666", ls="--" if "1/N" in nama else ":", label=nama)
    ax.axhline(D, color="k", lw=0.5)
    ax.set(xlabel="hari uji", ylabel="nilai portofolio (Rp)", title="Beli-dan-tahan pada data uji")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
