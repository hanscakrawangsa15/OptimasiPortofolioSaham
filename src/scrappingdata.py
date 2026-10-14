"""Menyiapkan data untuk optimasi portofolio LQ45 (walk-forward).

Jalankan dari root proyek:  python3 src/scrappingdata.py

Keluaran di data/:
  harga_close_raw.csv   Close tanpa penyesuaian dividen (dipakai untuk lot)
  harga_adjclose.csv    Adj Close (dipakai untuk return)
  return_sederhana.csv  pct_change Adj Close (BUKAN log-return)
  ticker_dibuang.txt, stock_splits.csv, metadata.json, laporan_data.md
  windows/              jendela_{k}.npz, kasus_tetap.npz, ringkasan_jendela.csv
"""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
import laporan_data  # noqa: E402
import walkforward  # noqa: E402


def unduh_mentah(tickers, start, end):
    """Unduh Adj Close, Close (tanpa penyesuaian dividen), dan Stock Splits."""
    import yfinance as yf
    dl = yf.download(tickers, start=start, end=end, auto_adjust=False,
                     actions=True, progress=False)
    ambil = lambda k: dl[k].reindex(columns=tickers)  # noqa: E731
    return ambil("Adj Close"), ambil("Close"), ambil("Stock Splits")


def bersihkan_data(adj, raw, max_missing):
    """Samakan kolom dan tanggal Adj Close dan Close; tidak ada yang dibuang diam-diam.

    - Hari tanpa data sama sekali (libur) dibuang dan dihitung.
    - Ticker kosong > max_missing (dari Adj Close) atau gagal unduh dibuang + alasan.
    - ffill maksimal 3 hari, lalu baris yang masih NaN dibuang dan dilaporkan.
    Mengembalikan (adj, raw, dibuang: {ticker: alasan}, info).
    """
    raw = raw.reindex(index=adj.index, columns=adj.columns)
    semua_kosong = adj.isna().all(axis=1) & raw.isna().all(axis=1)
    adj, raw = adj[~semua_kosong], raw[~semua_kosong]

    kosong = adj.isna().mean()
    dibuang = {}
    for t in adj.columns:
        if adj[t].isna().all():
            dibuang[t] = "unduhan gagal / tidak ada data sama sekali"
        elif kosong[t] > max_missing:
            dibuang[t] = f"data kosong {kosong[t]:.1%} > {max_missing:.0%} (dihitung dari Adj Close)"
    kolom = [c for c in adj.columns if c not in dibuang]

    adj_f = adj[kolom].ffill(limit=3)
    raw_f = raw[kolom].ffill(limit=3)
    ok = adj_f.notna().all(axis=1) & raw_f.notna().all(axis=1)
    if not ok.any():
        raise ValueError("Tidak ada hari dengan data lengkap setelah pembersihan")
    awal_ok = ok.idxmax()
    pertama = adj[kolom].apply(lambda s: s.first_valid_index())
    penyebab_awal = sorted(pertama[pertama == pertama.max()].index)
    hari_buang = ok.index[~ok]
    tgl = lambda idx: [str(pd.Timestamp(t).date()) for t in idx]  # noqa: E731
    info = {
        "n_hari_kosong_semua": int(semua_kosong.sum()),
        "n_hari_dibuang_awal": int((hari_buang < awal_ok).sum()),
        "hari_dibuang_tengah": tgl(hari_buang[hari_buang > awal_ok]),
        "ticker_penyebab_awal": penyebab_awal,
    }
    info["n_hari_dibuang_tengah"] = len(info["hari_dibuang_tengah"])
    return adj_f[ok], raw_f[ok], dibuang, info


def ekstrak_split(splits):
    s = splits.fillna(0.0)
    baris = [(str(pd.Timestamp(t).date()), tk, float(v))
             for tk in s.columns for t, v in s[tk][s[tk] != 0].items()]
    return pd.DataFrame(baris, columns=["tanggal", "ticker", "rasio"])


def hitung_return_sederhana(adj):
    return adj.pct_change().dropna()


def main():
    import yfinance as yf
    os.makedirs(config.DATA_DIR, exist_ok=True)
    win_dir = os.path.join(config.DATA_DIR, "windows")
    p = lambda f: os.path.join(config.DATA_DIR, f)  # noqa: E731

    adj_m, raw_m, spl_m = unduh_mentah(config.TICKERS, config.START, config.END)
    adj, raw, dibuang, info = bersihkan_data(adj_m, raw_m, config.MAX_MISSING)
    if dibuang:
        print(f"PERINGATAN: {len(dibuang)} ticker dibuang:")
        for t, alasan in dibuang.items():
            print(f"  {t}: {alasan}")
    if adj.shape[1] < max(config.K_LIST):
        sys.exit(f"Saham tersisa {adj.shape[1]} < K maksimum {max(config.K_LIST)}.")
    if info["n_hari_dibuang_tengah"]:
        print(f"PERINGATAN: {info['n_hari_dibuang_tengah']} hari dibuang di tengah data "
              f"(lubang > 3 hari): {info['hari_dibuang_tengah']}")

    ret = hitung_return_sederhana(adj)
    adj.to_csv(p("harga_adjclose.csv"))
    raw.to_csv(p("harga_close_raw.csv"))
    ret.to_csv(p("return_sederhana.csv"))
    with open(p("ticker_dibuang.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(f"{t}\t{a}" for t, a in dibuang.items())
                if dibuang else "(tidak ada ticker yang dibuang)")
        f.write("\n")
    splits = ekstrak_split(spl_m[list(adj.columns)])
    splits.to_csv(p("stock_splits.csv"), index=False)

    meta = {
        "waktu_unduh": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "versi_yfinance": yf.__version__,
        "tanggal_awal": str(adj.index[0].date()),
        "tanggal_akhir": str(adj.index[-1].date()),
        "permintaan_start": config.START, "permintaan_end": config.END,
        "n_ticker": int(adj.shape[1]), "n_hari": int(len(adj)),
        "n_hari_return": int(len(ret)), "n_ticker_dibuang": len(dibuang),
        **info,
    }
    with open(p("metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    jendela = walkforward.buat_jendela(ret, raw)
    kasus = walkforward.kasus_tetap(ret, raw, config.TRAIN_END)
    walkforward.simpan_jendela(jendela, kasus, win_dir)
    for w in jendela + [kasus]:
        if w["eig_min"] < -1e-10:
            print(f"PERINGATAN: cov jendela {w['id']} tidak semidefinit positif "
                  f"(eig min {w['eig_min']:.3e}); TIDAK diperbaiki.")

    anomali = laporan_data.deteksi_anomali(raw, adj)
    outlier = laporan_data.deteksi_outlier_return(ret)
    laporan_data.tulis_laporan(p("laporan_data.md"), meta, dibuang, info, jendela,
                               kasus, raw, anomali, outlier, splits)
    print(f"Ticker dipakai: {adj.shape[1]} | hari: {len(adj)} "
          f"({meta['tanggal_awal']} .. {meta['tanggal_akhir']}) | return: {len(ret)}")
    print(f"Jendela walk-forward: {len(jendela)} | anomali split: {len(anomali)} | "
          f"outlier return: {len(outlier)}")
    print("Selesai. Lihat data/laporan_data.md")


if __name__ == "__main__":
    main()
