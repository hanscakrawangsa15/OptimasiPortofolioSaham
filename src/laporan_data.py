"""Deteksi anomali data dan penulisan data/laporan_data.md.

Fungsi di sini hanya MELAPORKAN; tidak ada data yang dihapus atau diperbaiki.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# Perkiraan awal jumlah saham dengan 1 lot <= 20% D (harga adjusted), urut D_LEVELS
PERKIRAAN_CAP20 = [39, 40, 42, 42, 42, 42]


def deteksi_anomali(raw, adj, ambang=0.35, selisih=0.10):
    """Hari dengan |perubahan harga raw| > ambang yang TIDAK tercermin di Adj Close.

    "Tidak tercermin" = |ret_raw - ret_adj| > selisih. Indikasi pemecahan saham
    atau reverse split yang hanya ada di salah satu seri. Hanya dilaporkan.
    """
    rr, ra = raw.pct_change(), adj.pct_change()
    mask = (rr.abs() > ambang) & ((rr - ra).abs() > selisih)
    baris = [(t, tk, rr.at[t, tk], ra.at[t, tk])
             for tk in raw.columns for t in rr.index[mask[tk].values]]
    return pd.DataFrame(baris, columns=["tanggal", "ticker", "ret_raw", "ret_adj"])


def deteksi_outlier_return(ret, ambang=0.35):
    """Return harian sederhana dengan |r| > ambang. Tidak di-winsorize, hanya dilaporkan."""
    baris = [(t, tk, ret.at[t, tk])
             for tk in ret.columns for t in ret.index[(ret[tk].abs() > ambang).values]]
    return pd.DataFrame(baris, columns=["tanggal", "ticker", "return"])


def split_setelah_beli(jendela, splits):
    """Split yang terjadi SESUDAH tanggal beli tiap jendela.

    Close Yahoo sudah disesuaikan split, jadi harga_beli jendela tersebut tampak
    `rasio` kali lebih murah daripada harga transaksi sebenarnya saat itu.
    """
    baris = [(w["id"], w["tanggal_latih_akhir"], r.ticker, r.tanggal, r.rasio)
             for w in jendela for r in splits.itertuples()
             if r.tanggal > w["tanggal_latih_akhir"]]
    return pd.DataFrame(baris, columns=["jendela", "tanggal_beli", "ticker",
                                        "tanggal_split", "rasio"])


def tabel_kelayakan_lot(harga_raw_terakhir, d_levels=config.D_LEVELS,
                        lot=config.LOT_SIZE, cap=config.MAX_WEIGHT, k_maks=10):
    """Per D: jumlah saham dengan 1 lot <= cap*D dan 1 lot <= D/k_maks."""
    harga_lot = harga_raw_terakhir.values * lot
    return pd.DataFrame({
        "D (juta Rp)": [d / 1e6 for d in d_levels],
        f"lot <= {cap:.0%} D": [int((harga_lot <= cap * d).sum()) for d in d_levels],
        f"lot <= D/{k_maks}": [int((harga_lot <= d / k_maks).sum()) for d in d_levels],
    })


def _md(df, floatfmt="{:.4f}"):
    kol = [str(c) for c in df.columns]
    out = ["| " + " | ".join(kol) + " |", "|" + "---|" * len(kol)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(
            floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
            for v in r.values) + " |")
    return "\n".join(out)


def tulis_laporan(path, meta, dibuang, info, jendela, kasus, raw, anomali,
                  outlier, splits):
    L = []
    a = L.append
    a("# Laporan data\n")
    a("## Ringkasan data\n")
    a(f"- Rentang tanggal harga: {meta['tanggal_awal']} s.d. {meta['tanggal_akhir']}")
    a(f"- Jumlah ticker dipakai: {meta['n_ticker']} (dari {len(config.TICKERS)} kandidat)")
    a(f"- Jumlah hari harga: {meta['n_hari']}; jumlah hari return sederhana: {meta['n_hari_return']}")
    a(f"- Hari tanpa data sama sekali (hari libur, dibuang): {info['n_hari_kosong_semua']}")
    a(f"- Hari dibuang di awal (ticker lain belum punya data / lubang awal): {info['n_hari_dibuang_awal']}")
    a(f"- Hari dibuang di tengah/akhir (lubang > 3 hari setelah ffill): {info['n_hari_dibuang_tengah']}")
    if info["hari_dibuang_tengah"]:
        a(f"  - tanggal: {', '.join(info['hari_dibuang_tengah'])}. Return sesudah tanggal ini "
          f"mencakup beberapa hari bursa tetapi dihitung sebagai satu hari.")
    if info.get("ticker_penyebab_awal"):
        a(f"  - ticker yang baru mulai pada tanggal awal data: "
          f"{', '.join(info['ticker_penyebab_awal'])}")
    a("\n### Ticker dibuang\n")
    if dibuang:
        a(_md(pd.DataFrame({"ticker": list(dibuang), "alasan": list(dibuang.values())})))
    else:
        a("Tidak ada.")

    a("\n## Jendela walk-forward\n")
    a(f"Latih {config.WINDOW_TRAIN} hari, uji {config.WINDOW_TEST} hari, geser "
      f"{config.WINDOW_STEP} hari. Jumlah jendela: {len(jendela)}.\n")
    kolom_sig = [f"sigma_max {m}x" for m in config.SIGMA_MULT]
    baris = []
    for w in jendela + [kasus]:
        baris.append([w["id"], f"{w['tanggal_latih_awal']} .. {w['tanggal_latih_akhir']}",
                      f"{w['tanggal_uji_awal']} .. {w['tanggal_uji_akhir']}",
                      w["n_latih"], w["n_uji"], w["vol_benchmark"]]
                     + [w["sigma_max"][m] for m in config.SIGMA_MULT]
                     + [w["eig_min"]])
    tab = pd.DataFrame(baris, columns=["jendela", "latih", "uji", "n latih", "n uji",
                                       "vol 1/N"] + kolom_sig + ["eig min cov"])
    tab["eig min cov"] = tab["eig min cov"].map(lambda v: f"{v:.3e}")
    a(_md(tab))
    a("\nBaris `tetap` = kasus tetap (latih s.d. TRAIN_END, uji = seluruh sisa data). "
      "Volatilitas dan sigma_max tahunan (return sederhana).")

    a("\n### Volatilitas tahunan saham tunggal (data latih)\n")
    vb = []
    for w in jendela + [kasus]:
        v = np.sqrt(np.diag(w["cov"]))
        vb.append([w["id"], v.min(), float(np.median(v)), v.max()])
    a(_md(pd.DataFrame(vb, columns=["jendela", "min", "median", "maks"])))

    a("\n## Kelayakan lot\n")
    harga_akhir = raw.iloc[-1]
    a(f"Harga raw terakhir ({raw.index[-1].date()}), 1 lot = {config.LOT_SIZE} lembar. "
      f"Jumlah saham (dari {raw.shape[1]}) yang harga 1 lot-nya muat dalam batas.\n")
    kel = tabel_kelayakan_lot(harga_akhir)
    kel["perkiraan awal (cap 20%)"] = PERKIRAAN_CAP20[:len(kel)]
    kel["selisih"] = kel.iloc[:, 1] - kel["perkiraan awal (cap 20%)"]
    a(_md(kel, "{:g}"))
    if (kel["selisih"] == 0).all():
        a("\nIdentik dengan perkiraan awal. Wajar: pada hari terakhir, Adj Close sama dengan "
          "Close (penyesuaian dividen hanya mengubah harga masa lalu), sehingga harga raw "
          "dan adjusted terakhir sama.")
    else:
        a("\nAda selisih terhadap perkiraan awal; periksa kolom selisih dan ticker yang "
          "harga 1 lot-nya dekat batas.")

    a("\n## Anomali dan pencilan (hanya dilaporkan, tidak ada yang dihapus)\n")
    a(f"### Perubahan harga raw > 35% yang tidak tercermin di Adj Close: {len(anomali)} kasus\n")
    a(_md(anomali.assign(tanggal=anomali["tanggal"].astype(str))) if len(anomali) else "Tidak ada.")
    a(f"\n### Return harian sederhana |r| > 35%: {len(outlier)} kasus\n")
    a(_md(outlier.assign(tanggal=outlier["tanggal"].astype(str))) if len(outlier) else "Tidak ada.")
    a("\n### Pemecahan saham tercatat Yahoo dalam periode unduhan\n")
    a(_md(splits) if len(splits) else "Tidak ada.")
    sb = split_setelah_beli(jendela + [kasus], splits)
    a("\n### Split sesudah tanggal beli (harga_beli tampak terlalu murah)\n")
    a(_md(sb, "{:g}") if len(sb) else "Tidak ada.")
    a("\nCatatan: Close dari Yahoo sudah disesuaikan untuk stock split (hanya dividen yang "
      "tidak). Harga raw di sini adalah harga tanpa penyesuaian DIVIDEN. Untuk saham yang "
      "split setelah tanggal jendela, harga beli historis tampak lebih rendah dari harga "
      "transaksi sebenarnya saat itu.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
