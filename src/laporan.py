"""Bahan Bab 4 laporan: tabel dan gambar menurut outline (4.1 sampai 4.6), dari hasil Fase 1 dan Fase 2.

Sumber utama: hasil/csv/fase2_hasil.csv (3.600 run, parameter beku) dan hasil/csv/fase1_*_grid.csv. Tiga bagian butuh
percobaan tambahan yang kecil (seed 0..29, parameter beku, kasus tetap), disimpan sebagai CSV:
  - profil risiko 1,5x / 2,0x / 2,5x dengan riwayat konvergensi (4.x.2 dan 4.6.3),
  - ablasi operator GA: tanpa SBX, tanpa mutasi Gaussian, tanpa mutasi tukar (4.2.2).
Jalankan:  python3 src/laporan.py   ->  hasil/tabel/*.csv, hasil/gambar/*.png, docs/hasil_bab4.md
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

if __package__ is None:                                   # dijalankan sebagai skrip
    for _p in (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))):
        if _p not in sys.path:
            sys.path.insert(0, _p)

import config
import stats
from algorithms.brute_force_k2 import brute_force_k2
from data import NAMA_DATA, buat_problem, muat_jendela
from evaluasi import nilai_kas, nilai_uji, rp
from parameter import kw_terpilih
from penalaan import MODUL
from runner import PROFIL_RISIKO
from visualisasi import WARNA, warna

ALGO = ("GA", "PSO", "SA", "DE")
CSV = os.path.join(config.HASIL_DIR, "csv")
GAMBAR = os.path.join(config.HASIL_DIR, "gambar")
TABEL = os.path.join(config.HASIL_DIR, "tabel")
DOC = os.path.join(config.ROOT, "docs", "hasil_bab4.md")
F2 = os.path.join(CSV, "fase2_hasil.csv")
F_PROFIL = os.path.join(CSV, "bab4_profil_run.csv")
F_KONV = os.path.join(CSV, "bab4_profil_konvergensi.csv")
F_ABLASI = os.path.join(CSV, "bab4_ablasi_ga.csv")
DATA_UTAMA = "kasus_tetap"

ABLASI = {"Lengkap (parameter terpilih)": {}, "Tanpa pindah silang (SBX)": {"p_sbx": 0.0},
          "Tanpa mutasi Gaussian": {"laju_mut": 0.0}, "Tanpa mutasi tukar kunci": {"p_swap": 0.0},
          "Tanpa kedua mutasi": {"laju_mut": 0.0, "p_swap": 0.0}}


@lru_cache(maxsize=None)
def _problem(data, K, mult=1.5):
    return buat_problem(muat_jendela(data), K, mult=mult)


def idn(x, nd=2):
    """Angka berformat Indonesia: koma desimal."""
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}".replace(".", ",")


def rpf(x):
    return rp(x)


# ----------------------------------------------------------------------------- percobaan tambahan
def _run_profil(a):
    algo, K, profil, mult, seed = a
    P = _problem(DATA_UTAMA, K, mult)
    r = MODUL[algo].run(P, seed=seed, **kw_terpilih(algo))
    ring = dict(algoritma=algo, K=K, profil=profil, mult=mult, seed=seed, sigma_max=P.sigma_max, R=r.R, sigma=r.sigma,
                layak=bool(r.feasible), eval_terbaik=r.n_eval_best, detik=r.seconds)
    return ring, [(algo, K, profil, seed, n, v) for n, v in r.history]


def _run_ablasi(a):
    nama, K, seed = a
    P = _problem(DATA_UTAMA, K)
    r = MODUL["GA"].run(P, seed=seed, **{**kw_terpilih("GA"), **ABLASI[nama]})
    return dict(varian=nama, K=K, seed=seed, R=r.R, sigma=r.sigma, layak=bool(r.feasible),
                eval_terbaik=r.n_eval_best, detik=r.seconds)


def _peta(fungsi, tugas, proses):
    if proses and proses > 1:
        with ProcessPoolExecutor(max_workers=proses) as ex:
            return list(ex.map(fungsi, tugas, chunksize=max(1, len(tugas) // (proses * 16))))
    return [fungsi(t) for t in tugas]


def siapkan_percobaan(proses=None, paksa=False, cetak=print):
    """Jalankan percobaan tambahan bila CSV-nya belum ada."""
    os.makedirs(CSV, exist_ok=True)
    if paksa or not (os.path.exists(F_PROFIL) and os.path.exists(F_KONV)):
        t0 = time.perf_counter()
        tugas = [(a, K, p, m, s) for a in ALGO for K in config.K_LIST for p, m in PROFIL_RISIKO.items()
                 for s in config.SEEDS_FASE2]
        hasil = _peta(_run_profil, tugas, proses)
        pd.DataFrame([h[0] for h in hasil]).to_csv(F_PROFIL, index=False)
        pd.DataFrame([x for h in hasil for x in h[1]],
                     columns=["algoritma", "K", "profil", "seed", "n_eval", "R"]).to_csv(F_KONV, index=False)
        cetak(f"profil risiko: {len(tugas)} run dalam {time.perf_counter() - t0:.1f} dtk")
    if paksa or not os.path.exists(F_ABLASI):
        t0 = time.perf_counter()
        tugas = [(n, K, s) for n in ABLASI for K in (7, 10) for s in config.SEEDS_FASE2]
        pd.DataFrame(_peta(_run_ablasi, tugas, proses)).to_csv(F_ABLASI, index=False)
        cetak(f"ablasi GA: {len(tugas)} run dalam {time.perf_counter() - t0:.1f} dtk")


# ----------------------------------------------------------------------------- 4.1 validasi constraint
def _lots(s):
    return np.array([int(v) for v in str(s).split(";")], dtype=np.int64)


def periksa_run(lots, P):
    """Pemeriksaan kendala keras satu portofolio (semua harus True)."""
    biaya = lots * P.harga_lot
    w = biaya / P.D
    return {"lot bulat >= 0": bool((lots >= 0).all()), "saham <= K": bool((lots > 0).sum() <= P.K),
            "bobot <= cap": bool((w <= P.cap + 1e-9).all()), "dana <= modal": bool(biaya.sum() <= P.D + 1e-6),
            "kas >= 0": bool(P.D - biaya.sum() >= -1e-6)}


def tabel_validasi(df):
    """Per algoritma dan K: jumlah run, run layak (risiko <= sigma_max), dan jumlah pelanggaran kendala keras."""
    baris = []
    for (a, K), g in df.groupby(["algoritma", "K"]):
        pel = 0
        for _, r in g.iterrows():
            pel += int(not all(periksa_run(_lots(r["lots"]), _problem(r["data"], int(r["K"]))).values()))
        baris.append({"Algoritma": a, "K": K, "Run": len(g), "Layak risiko (%)": 100 * g["layak"].mean(),
                      "Pelanggaran": pel})
    return pd.DataFrame(baris)


# ----------------------------------------------------------------------------- 4.x.1 alokasi dana
def _terbaik(g):
    g = g[g["layak"]]
    return g.sort_values(["R", "seed"], ascending=[False, True]).iloc[0]


def terbaik_per_k(df, algo, data=DATA_UTAMA):
    baris = []
    for K, g in df[(df.algoritma == algo) & (df.data == data)].groupby("K"):
        r = _terbaik(g)
        P = _problem(data, int(K))
        dana = float((_lots(r["lots"]) * P.harga_lot).sum())
        baris.append({"K": int(K), "Seed": int(r["seed"]), "Jumlah saham": int(r["jumlah_saham"]),
                      "Dana saham (Rp)": dana, "Sisa kas (Rp)": P.D - dana, "Expected return (%/tahun)": 100 * r["R"],
                      "Risiko (%/tahun)": 100 * r["sigma"]})
    return pd.DataFrame(baris)


def rincian_portofolio(baris):
    """Rincian saham satu portofolio: lot, biaya, bobot, kontribusi return; plus total dan kas."""
    P = _problem(baris["data"], int(baris["K"]))
    lots = _lots(baris["lots"])
    idx = np.where(lots > 0)[0]
    idx = idx[np.argsort(-(lots[idx] * P.harga_lot[idx]))]
    biaya = lots[idx] * P.harga_lot[idx]
    rinci = pd.DataFrame({"Saham": [P.tickers[i] for i in idx], "Jumlah lot": lots[idx], "Biaya pembelian (Rp)": biaya,
                          "Bobot (%)": 100 * biaya / P.D, "Kontribusi return (pp)": 100 * (biaya / P.D) * P.mu[idx]})
    dana = float(biaya.sum())
    return rinci, dict(dana_saham=dana, sisa_kas=P.D - dana, kontribusi_kas=(P.D - dana) / P.D * P.rf)


# ----------------------------------------------------------------------------- 4.x.2 konsistensi
def tabel_kinerja30(df, algo, data=DATA_UTAMA):
    d = df[(df.algoritma == algo) & (df.data == data)]
    ac = stats.acuan_masalah(df[df.data == data])
    r = stats.ringkasan_latih(df[(df.algoritma == algo) & (df.data == data)], ac)
    out = d.groupby("K").agg(**{"Median R (%/tahun)": ("R", lambda s: 100 * s.median()),
                                "Simpangan baku (pp)": ("R", lambda s: 100 * s.std()),
                                "Kelayakan (%)": ("layak", lambda s: 100 * s.mean()),
                                "Median waktu (dtk)": ("detik", "median")}).reset_index()
    out["Keberhasilan gap <= 1% (%)"] = 100 * r.set_index("K").loc[out["K"], "sukses"].to_numpy()
    return out


def tabel_brute(df, algo):
    baris = []
    for d in NAMA_DATA:
        g = df[(df.algoritma == algo) & (df.data == d) & (df.K == 2)]
        rs = brute_force_k2(_problem(d, 2))["kas bebas"][0]
        baris.append({"Periode data": d, "Optimum brute force (%)": 100 * rs, "R terbaik (%)": 100 * g["R"].max(),
                      "Median R (%)": 100 * g["R"].median(),
                      "Keberhasilan gap <= 1% (%)": 100 * float((g["R"] >= 0.99 * rs).mean()),
                      "Menemukan optimum (% run)": 100 * float((g["R"] >= rs - 1e-9).mean())})
    return pd.DataFrame(baris)


# ----------------------------------------------------------------------------- 4.x.3 data uji
def tabel_uji(df, algo, data=DATA_UTAMA):
    baris = []
    for K, g in df[(df.algoritma == algo) & (df.data == data)].groupby("K"):
        r = _terbaik(g)
        baris.append({"K": int(K), "Return uji (%/tahun)": 100 * r["return_uji_tahunan"],
                      "Volatilitas (%/tahun)": 100 * r["vol_uji"], "Max drawdown (%)": 100 * r["mdd_uji"],
                      "Nilai akhir (Rp)": r["nilai_akhir_uji"]})
    return pd.DataFrame(baris)


def tabel_walkforward(df, algo):
    d = df[(df.algoritma == algo) & (df.data != DATA_UTAMA)]
    t = d.groupby(["K", "data"])["return_uji_tahunan"].median().unstack("data") * 100
    t.columns = [c.replace("jendela_", "Jendela ") for c in t.columns]
    return t.reset_index()


# ----------------------------------------------------------------------------- 4.6 lintas algoritma
def tabel_lintas_kinerja(df, data=DATA_UTAMA, rf=config.RISK_FREE):
    baris = []
    for K in config.K_LIST:
        for a in ALGO:
            g = df[(df.algoritma == a) & (df.data == data) & (df.K == K)]
            r = _terbaik(g)
            baris.append({"K": K, "Algoritma": a, "R terbaik (%)": 100 * r["R"], "Median R (%)": 100 * g["R"].median(),
                          "Risiko (%)": 100 * r["sigma"], "Sharpe latih": (r["R"] - rf) / r["sigma"],
                          "Return uji (%)": 100 * r["return_uji_tahunan"], "Volatilitas uji (%)": 100 * r["vol_uji"],
                          "Sharpe uji": r["sharpe_uji"]})
    return pd.DataFrame(baris)


def tabel_metrik_latih(df, data=DATA_UTAMA):
    """Metrik data latih lengkap (Tabel 3.14): R, kelayakan, selisih ke R acuan, keberhasilan, evaluasi, waktu."""
    d = df[df.data == data]
    r = stats.ringkasan_latih(d, stats.acuan_masalah(d))
    out = pd.DataFrame({"K": r["K"], "Algoritma": r["algoritma"], "Mean R (%)": 100 * r["mean"],
                        "Median R (%)": 100 * r["median"], "Std (pp)": 100 * r["std"], "Min R (%)": 100 * r["min"],
                        "Maks R (%)": 100 * r["max"], "Kelayakan (%)": 100 * r["layak"],
                        "Selisih ke R acuan (pp)": 100 * r["selisih"],
                        "Keberhasilan gap <= 1% (%)": 100 * r["sukses"],
                        "Median evaluasi hingga terbaik": r["eval_terbaik"], "Median waktu (dtk)": r["detik"]})
    out["_u"] = out["Algoritma"].map({a: i for i, a in enumerate(ALGO)})
    return out.sort_values(["K", "_u"]).drop(columns="_u").reset_index(drop=True)


def tabel_metrik_uji(df, data=DATA_UTAMA):
    """Metrik data uji lengkap (Tabel 3.14): median antar 30 run per algoritma dan K."""
    d = df[df.data == data]
    g = d.groupby(["K", "algoritma"]).agg(
        **{"Return uji (%/tahun)": ("return_uji_tahunan", lambda s: 100 * s.median()),
           "Volatilitas uji (%)": ("vol_uji", lambda s: 100 * s.median()), "Sharpe uji": ("sharpe_uji", "median"),
           "CVaR 95% harian (%)": ("cvar95_uji", lambda s: 100 * s.median()),
           "Max drawdown (%)": ("mdd_uji", lambda s: 100 * s.median()),
           "Shrinkage (pp)": ("shrinkage", lambda s: 100 * s.median()),
           "Volatilitas uji <= sigma_max (%)": ("risiko_uji_memenuhi", lambda s: 100 * s.mean())}).reset_index()
    g = g.rename(columns={"algoritma": "Algoritma"})
    g["_u"] = g["Algoritma"].map({a: i for i, a in enumerate(ALGO)})
    return g.sort_values(["K", "_u"]).drop(columns="_u").reset_index(drop=True)


def tabel_lintas_komputasi(df):
    ac = stats.acuan_masalah(df)
    rg = stats.ringkasan_latih(df, ac)
    out = df.groupby("algoritma").agg(**{"Median waktu (dtk)": ("detik", "median"),
                                         "Median evaluasi hingga terbaik": ("eval_terbaik", "median"),
                                         "Terbaik sebelum 50% anggaran (%)": ("eval_terbaik",
                                                                              lambda s: 100 * (s <= 0.5 * config.BUDGET_EVAL).mean())})
    out["Std R rata-rata antar seed (pp)"] = 100 * rg.groupby("algoritma")["std"].mean()
    out["Keberhasilan gap <= 1% (%)"] = 100 * rg.groupby("algoritma")["sukses"].mean()
    out["Peringkat rata-rata"] = stats.peringkat_rata(stats.median_per_masalah(df))
    return out.loc[list(ALGO)].reset_index().rename(columns={"algoritma": "Algoritma"})


def tabel_profil(prof):
    g = prof.groupby(["profil", "K", "algoritma"]).agg(R=("R", "median"), sigma=("sigma", "median"),
                                                       sigma_max=("sigma_max", "first")).reset_index()
    g["Median R (%)"] = 100 * g["R"]
    g["Median risiko (%)"] = 100 * g["sigma"]
    g["Batas (%)"] = 100 * g["sigma_max"]
    g["Mengikat"] = g["sigma"] >= g["sigma_max"] - 2e-3
    return g[["profil", "K", "algoritma", "Median R (%)", "Median risiko (%)", "Batas (%)", "Mengikat"]]


# ----------------------------------------------------------------------------- gambar
def _simpan(fig, nama):
    os.makedirs(GAMBAR, exist_ok=True)
    fig.savefig(os.path.join(GAMBAR, nama), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return nama


def gambar_bobot(df, algo):
    """Gambar 4.x.1: komposisi bobot saham dan kas portofolio terbaik tiap K."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, K in enumerate(config.K_LIST):
        r = _terbaik(df[(df.algoritma == algo) & (df.data == DATA_UTAMA) & (df.K == K)])
        rinci, tot = rincian_portofolio(r)
        bawah = 0.0
        for _, s in rinci.iterrows():
            ax.bar(i, s["Bobot (%)"], bottom=bawah, color=plt.cm.tab20(sum(map(ord, s["Saham"])) % 20), edgecolor="white", width=0.6)
            if s["Bobot (%)"] >= 6:
                ax.text(i, bawah + s["Bobot (%)"] / 2, s["Saham"], ha="center", va="center", fontsize=7)
            bawah += s["Bobot (%)"]
        kas = 100 * tot["sisa_kas"] / _problem(DATA_UTAMA, K).D
        ax.bar(i, kas, bottom=bawah, color="#cccccc", edgecolor="white", width=0.6)
        ax.text(i, bawah + kas / 2, f"Kas {idn(kas, 1)}%", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(config.K_LIST)), [f"K = {k}" for k in config.K_LIST])
    ax.set(ylabel="Bobot terhadap modal (%)", title=f"Komposisi portofolio terbaik {algo} (kasus tetap)")
    ax.set_ylim(0, 100)
    return _simpan(fig, f"bobot_{algo.lower()}.png")


def gambar_konvergensi(konv, algo):
    """Gambar 4.x.2: median dan kuartil R terbaik terhadap evaluasi, satu kurva per K (profil Konservatif = 1,5x)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    d = konv[(konv.algoritma == algo) & (konv.profil == "Konservatif")]
    for K, c in zip(config.K_LIST, plt.cm.viridis(np.linspace(0, 0.9, len(config.K_LIST)))):
        p = d[d.K == K].pivot(index="n_eval", columns="seed", values="R")
        ax.plot(p.index, p.median(axis=1), color=c, label=f"K = {K}")
        ax.fill_between(p.index, p.quantile(0.25, axis=1), p.quantile(0.75, axis=1), color=c, alpha=0.15)
    ax.set(xlabel="Jumlah evaluasi", ylabel="Expected return terbaik (layak)", title=f"Konvergensi {algo} (kasus tetap)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return _simpan(fig, f"konvergensi_{algo.lower()}.png")


def gambar_boxplot(df, algo):
    """Gambar 4.x.3: sebaran R dari 30 percobaan per K."""
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [100 * df[(df.algoritma == algo) & (df.data == DATA_UTAMA) & (df.K == K)]["R"] for K in config.K_LIST]
    bp = ax.boxplot(data, tick_labels=[str(k) for k in config.K_LIST], patch_artist=True)
    for b in bp["boxes"]:
        b.set_facecolor(warna(algo))
        b.set_alpha(0.5)
    ax.set(xlabel="Maksimum jumlah saham (K)", ylabel="Expected return latih (%/tahun)",
           title=f"Sebaran hasil {algo} pada 30 percobaan")
    ax.grid(alpha=0.3)
    return _simpan(fig, f"boxplot_{algo.lower()}.png")


def _sebar_label(y, jarak):
    """Geser posisi y label (urut naik) agar saling berjarak minimal `jarak`; mengembalikan posisi baru."""
    urut = np.argsort(y)
    baru = np.array(y, float)
    for a, b in zip(urut[:-1], urut[1:]):
        baru[b] = max(baru[b], baru[a] + jarak)
    geser = (np.mean(y) - np.mean(baru))                       # pusatkan kembali kelompok label
    return baru + geser


def gambar_nilai_uji(df, algo):
    """Perkembangan nilai portofolio terbaik tiap K pada data uji (beli-dan-tahan), dengan nilai akhir tiap garis."""
    w = muat_jendela(DATA_UTAMA)
    T = len(w["return_uji"])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    akhir, warna_k = [], []
    for K, c in zip(config.K_LIST, plt.cm.viridis(np.linspace(0, 0.9, len(config.K_LIST)))):
        r = _terbaik(df[(df.algoritma == algo) & (df.data == DATA_UTAMA) & (df.K == K)])
        P = _problem(DATA_UTAMA, K)
        v = nilai_uji(P, _lots(r["lots"]), w["return_uji"])
        ax.plot(range(1, T + 1), v / 1e6, color=c, label=f"K = {K}")
        akhir.append(float(v[-1]))
        warna_k.append(c)
    P = _problem(DATA_UTAMA, 5)
    kas = nilai_kas(P.D, P.rf, T)
    ax.plot(range(1, T + 1), kas / 1e6, "k--", label="Seluruh kas")
    ax.axhline(P.D / 1e6, color="gray", ls=":")
    nilai_label = akhir + [float(kas[-1])]
    teks = [f"K = {K}: {rp(v)}" for K, v in zip(config.K_LIST, akhir)] + [f"Kas: {rp(kas[-1])}"]
    warna_t = warna_k + ["black"]
    y_lab = _sebar_label([v / 1e6 for v in nilai_label], 0.085)
    for yv, yl, tk, c in zip(nilai_label, y_lab, teks, warna_t):
        ax.annotate(tk, xy=(T, yv / 1e6), xytext=(T + 6, yl), color=c, fontsize=8, va="center",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6, shrinkA=0, shrinkB=2), annotation_clip=False)
    ax.set_xlim(0, T + 55)
    ax.set(xlabel="Hari bursa setelah awal uji", ylabel="Nilai portofolio (juta Rp)",
           title=f"Nilai portofolio {algo} pada data uji (1 Okt 2025 - 30 Sep 2026)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    return _simpan(fig, f"nilai_uji_{algo.lower()}.png")


def gambar_ga_operator(ablasi):
    """Gambar 4.2.2: sebaran R GA tanpa operator tertentu."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, K in zip(axs, (7, 10)):
        data = [100 * ablasi[(ablasi.varian == v) & (ablasi.K == K)]["R"] for v in ABLASI]
        bp = ax.boxplot(data, tick_labels=[v.replace(" (parameter terpilih)", "").replace("Tanpa ", "tanpa\n") for v in ABLASI],
                        patch_artist=True)
        for b in bp["boxes"]:
            b.set_facecolor(WARNA["GA"])
            b.set_alpha(0.5)
        ax.set(title=f"K = {K}", ylabel="Expected return latih (%/tahun)")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.3)
    return _simpan(fig, "ga_operator.png")


def gambar_ga_populasi(df):
    """Keragaman populasi GA: jumlah himpunan saham berbeda dan median R populasi tiap generasi (seed terbaik K=7)."""
    r = _terbaik(df[(df.algoritma == "GA") & (df.data == DATA_UTAMA) & (df.K == 7)])
    P = _problem(DATA_UTAMA, 7)
    t = MODUL["GA"].run(P, seed=int(r["seed"]), trace=True, **kw_terpilih("GA")).trace
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    axs[0].plot(t["n_unik"], color=WARNA["GA"])
    axs[0].set(title="Himpunan saham berbeda dalam populasi", xlabel="Generasi", ylabel="Jumlah himpunan")
    lay = np.where(t["viol"] <= 0, t["R"], np.nan)
    axs[1].plot(np.nanmedian(lay, axis=1), color=WARNA["GA"], label="median populasi")
    axs[1].plot(np.nanmax(lay, axis=1), color="k", ls="--", label="terbaik")
    axs[1].set(title="Return ekspektasi populasi layak", xlabel="Generasi")
    axs[1].legend(fontsize=8)
    for a in axs:
        a.grid(alpha=0.3)
    return _simpan(fig, "ga_populasi.png")


def gambar_sa(df):
    """Gambar 4.4.2: jadwal suhu, lintasan R, dan pengaruh rasio suhu akhir (Fase 1)."""
    r = _terbaik(df[(df.algoritma == "SA") & (df.data == DATA_UTAMA) & (df.K == 7)])
    P = _problem(DATA_UTAMA, 7)
    t = MODUL["SA"].run(P, seed=int(r["seed"]), trace=True, **kw_terpilih("SA")).trace
    g = pd.read_csv(os.path.join(CSV, "fase1_sa_grid.csv"))
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    axs[0].semilogy(t["temperature"], color=WARNA["SA"])
    axs[0].set(title="Jadwal suhu (pendinginan geometrik)", xlabel="Langkah", ylabel="Suhu relatif T")
    axs[1].plot(np.asarray(t["R"]).ravel(), color=WARNA["SA"], lw=0.5)
    axs[1].set(title="R solusi saat ini", xlabel="Langkah", ylabel="Expected return")
    rasio = sorted(g["T_end_ratio"].unique())
    bp = axs[2].boxplot([100 * g[g.T_end_ratio == x]["R"] for x in rasio], tick_labels=[f"{x:g}" for x in rasio], patch_artist=True)
    for b in bp["boxes"]:
        b.set_facecolor(WARNA["SA"])
        b.set_alpha(0.5)
    axs[2].set(title="Pengaruh rasio suhu akhir (Fase 1)", xlabel="T_akhir / T0", ylabel="Expected return (%/tahun)")
    for a in axs:
        a.grid(alpha=0.3)
    return _simpan(fig, "sa_pendinginan.png")


def gambar_de_sensitivitas():
    """Peta panas median R menurut F dan CR (Fase 1)."""
    g = pd.read_csv(os.path.join(CSV, "fase1_de_grid.csv"))
    p = (100 * g.groupby(["F", "CR"])["R"].median()).unstack("CR")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    im = ax.imshow(p.values, cmap="Greens", aspect="auto")
    ax.set_xticks(range(p.shape[1]), [f"{c:g}" for c in p.columns])
    ax.set_yticks(range(p.shape[0]), [f"{f:g}" for f in p.index])
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if not np.isnan(p.values[i, j]):
                ax.text(j, i, idn(p.values[i, j], 2), ha="center", va="center", fontsize=8)
    ax.set(xlabel="Peluang pindah silang CR", ylabel="Faktor mutasi F", title="Median R (%/tahun) menurut F dan CR (jendela 1, K = 7)")
    plt.colorbar(im, ax=ax)
    return _simpan(fig, "de_sensitivitas.png")


def gambar_lintas_boxplot(df):
    fig, ax = plt.subplots(figsize=(11, 4.4))
    lebar = 0.2
    for k, a in enumerate(ALGO):
        data = [100 * df[(df.algoritma == a) & (df.data == DATA_UTAMA) & (df.K == K)]["R"] for K in config.K_LIST]
        bp = ax.boxplot(data, positions=np.arange(len(config.K_LIST)) + (k - 1.5) * lebar, widths=lebar * 0.9,
                        patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(warna(a))
            b.set_alpha(0.6)
        ax.plot([], [], color=warna(a), lw=8, alpha=0.6, label=a)
    ax.set_xticks(range(len(config.K_LIST)), [f"K = {k}" for k in config.K_LIST])
    ax.set(ylabel="Expected return latih (%/tahun)", title="Sebaran hasil 30 percobaan per algoritma (kasus tetap)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    return _simpan(fig, "lintas_boxplot.png")


def gambar_lintas_konvergensi(konv):
    fig, axs = plt.subplots(1, len(config.K_LIST), figsize=(4 * len(config.K_LIST), 3.6), sharex=True)
    d = konv[konv.profil == "Konservatif"]
    for ax, K in zip(axs, config.K_LIST):
        for a in ALGO:
            p = d[(d.algoritma == a) & (d.K == K)].pivot(index="n_eval", columns="seed", values="R")
            ax.plot(p.index, 100 * p.median(axis=1), color=warna(a), label=a)
            ax.fill_between(p.index, 100 * p.quantile(0.25, axis=1), 100 * p.quantile(0.75, axis=1), color=warna(a), alpha=0.12)
        ax.set(title=f"K = {K}", xlabel="Evaluasi")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Expected return terbaik (%/tahun)")
    axs[0].legend(fontsize=8)
    return _simpan(fig, "lintas_konvergensi.png")


def gambar_lintas_uji(df):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, kol, jud in ((axs[0], "return_uji_tahunan", "Return uji setara tahunan (%)"), (axs[1], "sharpe_uji", "Sharpe uji")):
        for k, a in enumerate(ALGO):
            data = [(100 if kol.startswith("return") else 1) * df[(df.algoritma == a) & (df.K == K)][kol] for K in config.K_LIST]
            bp = ax.boxplot(data, positions=np.arange(len(config.K_LIST)) + (k - 1.5) * 0.2, widths=0.18,
                            patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set_facecolor(warna(a))
                b.set_alpha(0.6)
            ax.plot([], [], color=warna(a), lw=8, alpha=0.6, label=a)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(range(len(config.K_LIST)), [f"K = {k}" for k in config.K_LIST])
        ax.set(title=jud + " (30 masalah)")
        ax.grid(axis="y", alpha=0.3)
    axs[0].legend(ncol=4, fontsize=8)
    return _simpan(fig, "lintas_uji.png")


def gambar_profil(konv, profil):
    """Gambar 4.6.3.x: konvergensi gabungan keempat algoritma pada satu profil risiko."""
    fig, axs = plt.subplots(1, len(config.K_LIST), figsize=(4 * len(config.K_LIST), 3.6), sharex=True)
    d = konv[konv.profil == profil]
    for ax, K in zip(axs, config.K_LIST):
        for a in ALGO:
            p = d[(d.algoritma == a) & (d.K == K)].pivot(index="n_eval", columns="seed", values="R")
            ax.plot(p.index, 100 * p.median(axis=1), color=warna(a), label=a)
            ax.fill_between(p.index, 100 * p.quantile(0.25, axis=1), 100 * p.quantile(0.75, axis=1), color=warna(a), alpha=0.12)
        ax.set(title=f"K = {K}", xlabel="Evaluasi")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("Expected return terbaik (%/tahun)")
    axs[0].legend(fontsize=8)
    fig.suptitle(f"Profil {profil} ({PROFIL_RISIKO[profil]}x volatilitas 1/N)", y=1.02)
    return _simpan(fig, f"profil_{profil.lower()}.png")


# ----------------------------------------------------------------------------- dokumen
def _md(df, fmt=None, nd=2):
    fmt = fmt or {}
    kol = list(df.columns)
    baris = ["| " + " | ".join(str(c) for c in kol) + " |", "|" + "|".join("---" for _ in kol) + "|"]
    for _, r in df.astype(object).iterrows():
        sel = []
        for c in kol:
            v = r[c]
            if c in fmt:
                sel.append(fmt[c](v))
            elif isinstance(v, (float, np.floating)):
                sel.append(idn(float(v), nd))
            elif isinstance(v, (bool, np.bool_)):
                sel.append("ya" if v else "tidak")
            else:
                sel.append(str(v))
        baris.append("| " + " | ".join(sel) + " |")
    return "\n".join(baris)


def _simpan_tabel(df, nama):
    os.makedirs(TABEL, exist_ok=True)
    df.to_csv(os.path.join(TABEL, nama + ".csv"), index=False)


def bangun(proses=None, paksa=False, cetak=print):
    """Buat semua tabel, gambar, dan docs/hasil_bab4.md."""
    siapkan_percobaan(proses, paksa, cetak)
    df = pd.read_csv(F2)
    prof, konv, abl = pd.read_csv(F_PROFIL), pd.read_csv(F_KONV), pd.read_csv(F_ABLASI)
    out = ["# Bahan Bab 4 (dibuat otomatis oleh src/laporan.py)", "",
           "Semua angka dari `hasil/csv/fase2_hasil.csv` (kasus tetap bila tidak disebut lain), parameter beku Fase 1, seed 0-29.",
           "Gambar ada di `hasil/gambar/`, tabel CSV di `hasil/tabel/`.", ""]
    rupiah = {"Dana saham (Rp)": rpf, "Sisa kas (Rp)": rpf, "Biaya pembelian (Rp)": rpf, "Nilai akhir (Rp)": rpf}

    # 4.1
    v = tabel_validasi(df)
    _simpan_tabel(v, "4_1_validasi_constraint")
    ringkas = v.groupby("Algoritma").agg(Run=("Run", "sum"), **{"Layak risiko (%)": ("Layak risiko (%)", "mean")},
                                         Pelanggaran=("Pelanggaran", "sum")).loc[list(ALGO)].reset_index()
    out += ["## 4.1 Validasi Constraint", "",
            "Pemeriksaan kendala keras pada seluruh run Fase 2 (lot bulat >= 0, saham <= K, bobot <= 20% bila K >= 7, dana <= modal, kas >= 0) dan kelayakan risiko (sigma <= sigma_max).",
            "", "Tabel 4.1 Hasil validasi constraint per algoritma", "", _md(ringkas, {"Run": lambda x: f"{int(x):,}".replace(",", "."),
                                                                                        "Pelanggaran": lambda x: str(int(x)),
                                                                                        "Layak risiko (%)": lambda x: idn(x, 1)}), ""]

    nomor = {"GA": "4.2", "PSO": "4.3", "SA": "4.4", "DE": "4.5"}
    # outline: GA 4.2.1-4.2.2, PSO 4.3.1-4.3.3, SA 4.4.1-4.4.2, DE 4.5.1-4.5.3; subbab lain bertanda "(tambahan)"
    sec_konv = {"GA": "4.2.3 (tambahan)", "PSO": "4.3.2", "SA": "4.4.3 (tambahan)", "DE": "4.5.2"}
    sec_uji = {"GA": "4.2.4 (tambahan)", "PSO": "4.3.3", "SA": "4.4.4 (tambahan)", "DE": "4.5.3"}
    for a in ALGO:
        n = nomor[a]
        _t, _g = iter(range(1, 99)), iter(range(1, 99))
        T = lambda: f"{n}.{next(_t)}"           # noqa: E731  nomor tabel berurutan per algoritma
        G = lambda: f"{n}.{next(_g)}"           # noqa: E731  nomor gambar berurutan per algoritma
        b = terbaik_per_k(df, a)
        _simpan_tabel(b, f"{n}_1_hasil_terbaik_{a.lower()}")
        k_best = int(b.loc[b["Expected return (%/tahun)"].idxmax(), "K"])
        r_best = _terbaik(df[(df.algoritma == a) & (df.data == DATA_UTAMA) & (df.K == k_best)])
        rinci, tot = rincian_portofolio(r_best)
        _simpan_tabel(rinci, f"{n}_1_rincian_{a.lower()}")
        P = _problem(DATA_UTAMA, k_best)
        gambar_bobot(df, a)
        out += [f"## {n} Hasil {a}", "", f"### {n}.1 Optimasi alokasi dana portofolio", "",
                f"Tabel {T()} Hasil optimasi terbaik {a} pada setiap K (batas risiko {idn(100 * P.sigma_max, 2)}%/tahun)", "",
                _md(b, {**rupiah, "K": str, "Seed": str, "Jumlah saham": str}), "",
                f"Portofolio dengan expected return tertinggi: K = {k_best} (seed {int(r_best['seed'])}), {idn(100 * r_best['R'], 2)}%/tahun.", "",
                f"Tabel {T()} Rincian alokasi dana portofolio terbaik {a} (K = {k_best})", "",
                _md(rinci, {**rupiah, "Jumlah lot": str}), "",
                f"Total saham {rpf(tot['dana_saham'])} ({idn(100 * tot['dana_saham'] / P.D, 2)}%); sisa kas {rpf(tot['sisa_kas'])} ({idn(100 * tot['sisa_kas'] / P.D, 2)}%).", "",
                f"![Gambar](../hasil/gambar/bobot_{a.lower()}.png)", f"Gambar {G()} Komposisi bobot saham dan kas {a} pada setiap K", ""]

        if a == "GA":
            ga_tab = abl.groupby(["K", "varian"])["R"].agg(["median", "std", "max"]).reset_index()
            ga_tab = ga_tab.assign(**{"Median R (%)": 100 * ga_tab["median"], "Std (pp)": 100 * ga_tab["std"],
                                      "R terbaik (%)": 100 * ga_tab["max"]}).rename(columns={"varian": "Varian"})
            ga_tab = ga_tab[["K", "Varian", "Median R (%)", "Std (pp)", "R terbaik (%)"]]
            ga_tab["Varian"] = ga_tab["Varian"].astype(str)
            urut = {v: i for i, v in enumerate(ABLASI)}
            ga_tab = ga_tab.sort_values(["K", "Varian"], key=lambda s: s.map(urut) if s.name == "Varian" else s)
            _simpan_tabel(ga_tab, "4_2_2_ablasi_ga")
            gambar_ga_operator(abl)
            gambar_ga_populasi(df)
            out += ["### 4.2.2 Analisis karakteristik pencarian (crossover dan mutasi) pada GA", "",
                    "Ablasi operator pada kasus tetap, parameter terpilih, 30 seed: tiap varian mematikan satu operator.", "",
                    f"Tabel {T()} Pengaruh operator GA terhadap expected return", "", _md(ga_tab, {"K": str}), "",
                    "![Gambar](../hasil/gambar/ga_operator.png)", f"Gambar {G()} Sebaran R GA tanpa operator tertentu", "",
                    "![Gambar](../hasil/gambar/ga_populasi.png)", f"Gambar {G()} Keragaman dan kualitas populasi GA tiap generasi (K = 7, seed terbaik)", ""]

        if a == "SA":                           # outline SA: 4.4.2 = cooling schedule, tepat setelah 4.4.1
            gambar_sa(df)
            g1 = pd.read_csv(os.path.join(CSV, "fase1_sa_grid.csv"))
            pend = (100 * g1.groupby(["target_acceptance", "T_end_ratio"])["R"].median()).unstack("T_end_ratio")
            pend.columns = [f"T_akhir/T0 = {c:g}" for c in pend.columns]
            pend = pend.reset_index().rename(columns={"target_acceptance": "Peluang penerimaan awal"})
            _simpan_tabel(pend, "4_4_2_pendinginan_sa")
            out += ["### 4.4.2 Analisis efisiensi penurunan suhu (cooling schedule) terhadap konvergensi", "",
                    "![Gambar](../hasil/gambar/sa_pendinginan.png)", f"Gambar {G()} Jadwal suhu, lintasan R, dan pengaruh rasio suhu akhir", "",
                    f"Tabel {T()} Median R (%/tahun) pada penalaan SA (jendela 1, K = 7, 30 seed per sel)", "", _md(pend), ""]

        k30 = tabel_kinerja30(df, a)
        _simpan_tabel(k30, f"{n}_2_kinerja_30_percobaan_{a.lower()}")
        brute = tabel_brute(df, a)
        _simpan_tabel(brute, f"{n}_2_brute_force_{a.lower()}")
        gambar_konvergensi(konv, a)
        gambar_boxplot(df, a)
        out += [f"### {sec_konv[a]} Analisis konvergensi dan konsistensi hasil", "",
                f"![Gambar](../hasil/gambar/konvergensi_{a.lower()}.png)", f"Gambar {G()} Konvergensi {a} pada setiap K", "",
                f"Tabel {T()} Kinerja {a} pada 30 percobaan (kasus tetap)", "", _md(k30, {"K": str}), "",
                f"![Gambar](../hasil/gambar/boxplot_{a.lower()}.png)", f"Gambar {G()} Sebaran expected return {a} pada 30 percobaan", "",
                f"Tabel {T()} Verifikasi {a} terhadap brute force pada K = 2 (semua periode data)", "", _md(brute), ""]

        u = tabel_uji(df, a)
        wf = tabel_walkforward(df, a)
        _simpan_tabel(u, f"{n}_3_data_uji_{a.lower()}")
        _simpan_tabel(wf, f"{n}_3_walkforward_{a.lower()}")
        gambar_nilai_uji(df, a)
        out += [f"### {sec_uji[a]} Evaluasi kinerja portofolio pada data uji", "",
                f"Tabel {T()} Hasil evaluasi portofolio {a} pada data uji (1 Okt 2025 - 30 Sep 2026, beli-dan-tahan)", "",
                _md(u, {**rupiah, "K": str}), "",
                f"![Gambar](../hasil/gambar/nilai_uji_{a.lower()}.png)", f"Gambar {G()} Perkembangan nilai portofolio {a} pada data uji", "",
                f"Tabel {T()} Median return uji setara tahunan (%) pada walk-forward jendela 1-5", "", _md(wf, {"K": str}), ""]

        if a == "DE":
            gambar_de_sensitivitas()
            out += ["### 4.5.4 Sensitivitas parameter F dan CR (tambahan)", "",
                    "![Gambar](../hasil/gambar/de_sensitivitas.png)", f"Gambar {G()} Median R menurut F dan CR pada penalaan DE", ""]

    # 4.6
    lk = tabel_lintas_kinerja(df)
    lc = tabel_lintas_komputasi(df)
    ml, mu = tabel_metrik_latih(df), tabel_metrik_uji(df)
    _simpan_tabel(ml, "4_6_1a_metrik_latih")
    _simpan_tabel(mu, "4_6_1b_metrik_uji")
    w = stats.wilcoxon_holm(stats.median_per_masalah(df))
    _simpan_tabel(lk, "4_6_1_lintas_kinerja")
    _simpan_tabel(lc, "4_6_2_lintas_komputasi")
    _simpan_tabel(w, "4_6_wilcoxon_holm")
    gambar_lintas_boxplot(df)
    gambar_lintas_konvergensi(konv)
    gambar_lintas_uji(df)
    out += ["## 4.6 Komparasi dan diskusi lintas algoritma", "", "### 4.6.1 Perbandingan kinerja portofolio optimal", "",
            "Tabel 4.6.1 Expected return, risiko, dan Sharpe ratio portofolio terbaik per algoritma (kasus tetap)", "",
            _md(lk, {"K": str}, nd=2), "",
            "Sharpe latih = (R - r_f) / sigma; Sharpe uji = (return setara tahunan uji - r_f) / volatilitas uji.", "",
            "Tabel 4.6.1a Metrik data latih lengkap (kasus tetap, 30 seed per sel)", "", _md(ml, {"K": str}), "",
            "Tabel 4.6.1b Metrik data uji lengkap (median 30 run per sel; kasus tetap)", "", _md(mu, {"K": str}), "",
            "![Gambar 4.6.1](../hasil/gambar/lintas_boxplot.png)", "Gambar 4.6.1 Sebaran expected return empat algoritma pada setiap K", "",
            "![Gambar 4.6.2](../hasil/gambar/lintas_uji.png)", "Gambar 4.6.2 Return dan Sharpe uji pada 30 masalah", "",
            "### 4.6.2 Perbandingan kecepatan komputasi dan stabilitas konvergensi", "",
            "Tabel 4.6.2 Komputasi dan stabilitas (seluruh 30 masalah)", "", _md(lc), "",
            "Tabel 4.6.3 Uji Wilcoxon berpasangan atas 30 masalah dengan koreksi Holm", "",
            _md(w, {"p": lambda x: idn(x, 4), "p_holm": lambda x: idn(x, 4), "median_selisih": lambda x: idn(100 * x, 3) + " pp",
                    "menang_A": str, "menang_B": str, "seri": str}), "",
            "![Gambar 4.6.3](../hasil/gambar/lintas_konvergensi.png)", "Gambar 4.6.3 Konvergensi gabungan pada setiap K (profil 1,5x)", ""]

    pr = tabel_profil(prof)
    _simpan_tabel(pr, "4_6_3_profil_risiko")
    out += ["### 4.6.3 Analisis konvergensi gabungan", "",
            "Profil risiko menentukan batas sigma_max: Konservatif 1,5x, Moderat 2,0x, Agresif 2,5x volatilitas portofolio 1/N (kasus tetap, 30 seed, parameter beku).", ""]
    for i, p in enumerate(PROFIL_RISIKO, 1):
        gambar_profil(konv, p)
        tp = pr[pr.profil == p].drop(columns="profil")
        tp = tp.pivot_table(index="K", columns="algoritma", values="Median R (%)").reset_index()
        out += [f"#### 4.6.3.{i} Skenario profil risiko {p}", "", f"![Gambar 4.6.4.{i}](../hasil/gambar/profil_{p.lower()}.png)",
                f"Gambar 4.6.4.{i} Konvergensi gabungan profil {p}", "",
                f"Tabel 4.6.4.{i} Median expected return (%/tahun) profil {p}", "", _md(tp, {"K": str}), ""]
    bind = pr.groupby(["profil", "K"])["Mengikat"].mean().unstack("K")
    bind = (100 * bind).reset_index()
    out += ["Tabel 4.6.5 Persentase algoritma yang risikonya menempel pada batas (mengikat) per profil dan K", "",
            _md(bind, {c: (lambda x: idn(x, 0)) for c in bind.columns if c != "profil"}), "",
            "Bila batas tidak mengikat, menaikkan batas risiko tidak menaikkan return, sehingga profil yang berbeda dapat memberi hasil identik.", ""]
    os.makedirs(os.path.dirname(DOC), exist_ok=True)
    with open(DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    cetak(f"selesai: {DOC}")


if __name__ == "__main__":
    bangun(proses=os.cpu_count())
