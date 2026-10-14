# Laporan data

## Ringkasan data

- Rentang tanggal harga: 2021-11-22 s.d. 2026-09-30
- Jumlah ticker dipakai: 42 (dari 46 kandidat)
- Jumlah hari harga: 1163; jumlah hari return sederhana: 1162
- Hari tanpa data sama sekali (hari libur, dibuang): 0
- Hari dibuang di awal (ticker lain belum punya data / lubang awal): 35
- Hari dibuang di tengah/akhir (lubang > 3 hari setelah ffill): 0
  - ticker yang baru mulai pada tanggal awal data: MTEL.JK

### Ticker dibuang

| ticker | alasan |
|---|---|
| AMMN.JK | data kosong 35.7% > 5% (dihitung dari Adj Close) |
| GOTO.JK | data kosong 10.9% > 5% (dihitung dari Adj Close) |
| MBMA.JK | data kosong 31.9% > 5% (dihitung dari Adj Close) |
| PGEO.JK | data kosong 29.0% > 5% (dihitung dari Adj Close) |

## Jendela walk-forward

Latih 504 hari, uji 126 hari, geser 126 hari. Jumlah jendela: 5.

| jendela | latih | uji | n latih | n uji | vol 1/N | sigma_max 1.5x | sigma_max 2.0x | sigma_max 2.5x | eig min cov |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2021-11-23 .. 2023-12-14 | 2023-12-15 .. 2024-07-09 | 504 | 126 | 0.1365 | 0.2048 | 0.2731 | 0.3413 | 1.981e-02 |
| 2 | 2022-06-07 .. 2024-07-09 | 2024-07-10 .. 2025-01-09 | 504 | 126 | 0.1318 | 0.1976 | 0.2635 | 0.3294 | 2.074e-02 |
| 3 | 2022-12-01 .. 2025-01-09 | 2025-01-10 .. 2025-07-31 | 504 | 126 | 0.1308 | 0.1962 | 0.2616 | 0.3270 | 2.054e-02 |
| 4 | 2023-06-14 .. 2025-07-31 | 2025-08-01 .. 2026-02-03 | 504 | 126 | 0.1739 | 0.2609 | 0.3478 | 0.4348 | 2.191e-02 |
| 5 | 2023-12-15 .. 2026-02-03 | 2026-02-04 .. 2026-08-20 | 504 | 126 | 0.1811 | 0.2717 | 0.3622 | 0.4528 | 2.170e-02 |
| tetap | 2021-11-23 .. 2025-09-30 | 2025-10-01 .. 2026-09-30 | 923 | 239 | 0.1593 | 0.2390 | 0.3186 | 0.3983 | 2.545e-02 |

Baris `tetap` = kasus tetap (latih s.d. TRAIN_END, uji = seluruh sisa data). Volatilitas dan sigma_max tahunan (return sederhana).

### Volatilitas tahunan saham tunggal (data latih)

| jendela | min | median | maks |
|---|---|---|---|
| 1 | 0.2000 | 0.3587 | 0.7353 |
| 2 | 0.1886 | 0.3312 | 0.7132 |
| 3 | 0.2005 | 0.3217 | 0.6542 |
| 4 | 0.2292 | 0.3573 | 0.6942 |
| 5 | 0.2012 | 0.3921 | 0.6613 |
| tetap | 0.2262 | 0.3599 | 0.6464 |

## Kelayakan lot

Harga raw terakhir (2026-09-30), 1 lot = 100 lembar. Jumlah saham (dari 42) yang harga 1 lot-nya muat dalam batas.

| D (juta Rp) | lot <= 20% D | lot <= D/10 | perkiraan awal (cap 20%) | selisih |
|---|---|---|---|---|
| 5 | 39 | 35 | 39 | 0 |
| 10 | 40 | 39 | 40 | 0 |
| 20 | 42 | 40 | 42 | 0 |
| 35 | 42 | 42 | 42 | 0 |
| 50 | 42 | 42 | 42 | 0 |
| 100 | 42 | 42 | 42 | 0 |

Identik dengan perkiraan awal. Wajar: pada hari terakhir, Adj Close sama dengan Close (penyesuaian dividen hanya mengubah harga masa lalu), sehingga harga raw dan adjusted terakhir sama.

## Anomali dan pencilan (hanya dilaporkan, tidak ada yang dihapus)

### Perubahan harga raw > 35% yang tidak tercermin di Adj Close: 0 kasus

Tidak ada.

### Return harian sederhana |r| > 35%: 0 kasus

Tidak ada.

### Pemecahan saham tercatat Yahoo dalam periode unduhan

| tanggal | ticker | rasio |
|---|---|---|
| 2022-01-12 | AKRA.JK | 5.0000 |
| 2021-10-13 | BBCA.JK | 5.0000 |
| 2023-10-06 | BBNI.JK | 2.0000 |
| 2023-04-04 | BMRI.JK | 2.0000 |
| 2022-12-20 | BRPT.JK | 1.0042 |
| 2024-06-27 | BRPT.JK | 1.0016 |
| 2022-06-02 | HRUM.JK | 5.0000 |
| 2024-10-14 | ISAT.JK | 4.0000 |

### Split sesudah tanggal beli (harga_beli tampak terlalu murah)

| jendela | tanggal_beli | ticker | tanggal_split | rasio |
|---|---|---|---|---|
| 1 | 2023-12-14 | BRPT.JK | 2024-06-27 | 1.0016 |
| 1 | 2023-12-14 | ISAT.JK | 2024-10-14 | 4 |
| 2 | 2024-07-09 | ISAT.JK | 2024-10-14 | 4 |

Catatan: Close dari Yahoo sudah disesuaikan untuk stock split (hanya dividen yang tidak). Harga raw di sini adalah harga tanpa penyesuaian DIVIDEN. Untuk saham yang split setelah tanggal jendela, harga beli historis tampak lebih rendah dari harga transaksi sebenarnya saat itu.
