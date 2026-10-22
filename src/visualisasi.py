"""Grafik matplotlib untuk GUI (GA #0072B2, PSO #D55E00)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WARNA = {"GA": "#0072B2", "PSO": "#D55E00", "SA": "#CC79A7", "DE": "#009E73"}


def warna(metode):
    return WARNA.get(metode.split()[0], "#666666")


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
    ax.barh(nama, val, color=["#E69F00"] * len(idx) + ["#999999"])
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


def plot_konvergensi_profil(kurva, sigma_max):
    """Kurva konvergensi per profil risiko (satu panel tiap profil): median dan rentang kuartil tiap metode.

    kurva = {profil: {metode: hasil runner.konvergensi}}; sigma_max = {profil: batas risiko}.
    """
    profil = list(kurva)
    fig, axs = plt.subplots(1, len(profil), figsize=(5 * len(profil), 3.8), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, p in zip(axs, profil):
        for m, k in kurva[p].items():
            ax.plot(k["evaluasi"], k["median"], color=warna(m), label=m)
            ax.fill_between(k["evaluasi"], k["q1"], k["q3"], color=warna(m), alpha=0.2)
            if not np.isnan(k["capai_99"]):
                ax.axvline(k["capai_99"], color=warna(m), ls=":", lw=1)
        ax.set(title=f"{p} (σ_max = {sigma_max[p]:.3f})", xlabel="jumlah evaluasi")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("R terbaik yang layak")
    axs[0].legend(fontsize=8)
    fig.suptitle("Konvergensi menurut profil risiko (garis putus-putus: 99% perbaikan tercapai)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_proses(trace, P, judul):
    """Proses satu run: sebaran populasi (risiko-return) pada awal, tengah, akhir, plus R terbaik per iterasi."""
    G = len(trace["R"])
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, g in zip(axs[:3], [0, G // 2, G - 1]):
        lay = trace["viol"][g] <= 0
        ax.scatter(trace["sigma"][g][~lay], trace["R"][g][~lay], s=12, color="#999999", label="melanggar")
        ax.scatter(trace["sigma"][g][lay], trace["R"][g][lay], s=12, color="#2CA02C", label="layak")
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


def plot_proses_sa(trace, P, judul):
    """Proses satu run SA: lintasan solusi saat ini (risiko-return), R saat ini, dan jadwal temperatur."""
    R = np.asarray(trace["R"]).ravel()
    sg = np.asarray(trace["sigma"]).ravel()
    T = np.asarray(trace["temperature"]).ravel()
    langkah = np.arange(len(R))
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.4))
    sc = axs[0].scatter(sg, R, c=langkah, cmap="viridis", s=4)
    axs[0].axvline(P.sigma_max, color="k", ls="--", lw=1)
    axs[0].set(title="lintasan solusi saat ini", xlabel="σ", ylabel="R"); axs[0].grid(alpha=0.3)
    fig.colorbar(sc, ax=axs[0], label="langkah")
    axs[1].plot(langkah, R, color=WARNA["SA"], lw=0.6)
    axs[1].set(title="R solusi saat ini", xlabel="langkah"); axs[1].grid(alpha=0.3)
    axs[2].semilogy(langkah, T, color="#444444")
    axs[2].set(title="jadwal temperatur", xlabel="langkah"); axs[2].grid(alpha=0.3)
    fig.suptitle(judul)
    fig.tight_layout()
    return fig


def plot_uji(nilai, D):
    """Nilai portofolio harian pada data uji."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for nama, v in nilai.items():
        if nama.split()[0] in WARNA:
            ax.plot(v, color=warna(nama), label=nama)
        else:
            ax.plot(v, color="#666666", ls="--" if "1/N" in nama else ":", label=nama)
    ax.axhline(D, color="k", lw=0.5)
    ax.set(xlabel="hari uji", ylabel="nilai portofolio (Rp)", title="Beli-dan-tahan pada data uji")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
