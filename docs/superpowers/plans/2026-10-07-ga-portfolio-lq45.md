# Optimasi Portofolio LQ45 dengan GA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membangun pipeline Python yang mengoptimalkan bobot portofolio saham LQ45 dengan GA real-coded (maksimasi Sharpe, batas 10% per saham), membandingkannya dengan baseline SLSQP dan 1/N pada data uji, dan menghasilkan tabel metrik serta enam visual.

**Architecture:** Modul kecil di `src/` yang diimpor sebagai top-level (`src` ditambahkan ke `sys.path` oleh `main.py` dan `tests/conftest.py`). Fungsi murni (Sharpe, repair) ada di `src/portofolio.py` dan dipakai bersama oleh GA, baseline, dan evaluasi. `main.py` menjalankan: data, baseline, GA 30 run, evaluasi, visual.

**Tech Stack:** Python 3.13, numpy, pandas, scipy (SLSQP), matplotlib, yfinance, pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-ga-portfolio-lq45-design.md`

## Global Constraints

- Bobot long-only, jumlah = 1, tiap bobot <= `MAX_WEIGHT` = 0.10 (parameter di `config.py`; GA, SLSQP, dan 1/N memakai nilai yang sama).
- mu dan Sigma untuk fitness dan baseline hanya dari data latih; data uji tidak boleh dipakai untuk memilih bobot.
- Latih: START..TRAIN_END (2021-10-01..2025-09-30); uji: sesudahnya sampai 2026-09-30.
- GA: 30 run, seed 0..29, `numpy.random.default_rng(seed)`, populasi 200, 300 generasi, rf 5%/tahun, 252 hari bursa.
- Bobot GA terbaik dipilih dari fitness latih tertinggi, tidak pernah dari metrik uji.
- Nama file memakai yang ada di `src/`: `scrappingdata.py`, `GeneticAlgorithm.py`, `baseline.py`, `evaluasi.py`, `visualisasi.py`. File tambahan: `config.py` (root), `src/portofolio.py`.
- Keluaran ke `data/` (csv/npz) dan `hasil/` (png/csv).

## Review Focus

1. Ticker IPO belakangan (mis. GOTO, AMMN) membuat `dropna()` memangkas seluruh rentang tanggal: ticker dengan data kosong > `MAX_MISSING` harus dibuang sebelum `dropna()` (Task 1).
2. Cap tidak layak (jumlah saham x cap < 1) harus melempar `ValueError` jelas di repair, SLSQP, dan 1/N, bukan menghasilkan bobot salah (Task 2, 3).
3. Vektor bobot semua nol atau varians nol tidak boleh menghasilkan NaN/inf diam-diam (Task 2).
4. `TRAIN_END` yang bukan hari bursa, atau periode uji kosong, harus tetap membagi data dengan benar atau gagal dengan pesan jelas (Task 1).
5. Hasil GA harus identik untuk seed sama dan ukuran populasi ganjil ditolak (Task 4).

---

## File Structure

| File | Tanggung jawab |
|---|---|
| `config.py` | Konstanta: ticker, tanggal, cap, rf, parameter GA, path |
| `src/portofolio.py` | `sharpe_ratio`, `repair_weights`, `repair_population` |
| `src/scrappingdata.py` | Unduh dan bersihkan data, log return, split, mu/Sigma |
| `src/baseline.py` | `equal_weight`, `max_sharpe_slsqp` |
| `src/GeneticAlgorithm.py` | `GAResult`, `run_ga`, `run_ga_batch` |
| `src/evaluasi.py` | Metrik uji, nilai kumulatif, drawdown |
| `src/visualisasi.py` | Enam fungsi gambar |
| `main.py` | Orkestrasi pipeline |
| `tests/` | `conftest.py` dan satu file tes per modul |

---

### Task 1: Config dan data pipeline

**Files:**
- Create: `config.py`, `tests/conftest.py`, `tests/test_data.py`
- Modify: `src/scrappingdata.py` (tulis ulang)

**Interfaces:**
- Produces (`config`): `TICKERS: list[str]`, `START, END, TRAIN_END: str`, `TRADING_DAYS: int`, `MAX_WEIGHT: float`, `RISK_FREE: float`, `MAX_MISSING: float`, `N_RUNS, POP_SIZE, GENERATIONS: int`, `DATA_DIR, HASIL_DIR: str`.
- Produces (`scrappingdata`): `bersihkan_harga(prices, max_missing) -> (DataFrame, list[str])`, `hitung_log_return(prices) -> DataFrame`, `pisah_latih_uji(rets, train_end) -> (train, test)`, `hitung_mu_cov(train, trading_days) -> (mu, cov)`, `unduh_harga(tickers, start, end) -> DataFrame`, `main() -> None`.

- [ ] **Step 1: Pasang pytest**

Run: `pip install --user pytest` (atau `sudo pacman -S python-pytest`)
Expected: `python3 -m pytest --version` mencetak versi.

- [ ] **Step 2: Tulis `config.py`**

```python
"""Konfigurasi proyek optimasi portofolio LQ45 dengan GA."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
HASIL_DIR = os.path.join(ROOT, "hasil")

# Kandidat konstituen LQ45. Ticker tanpa data cukup dibuang otomatis saat
# pengunduhan. Cocokkan dengan daftar resmi IDX periode yang kamu pakai.
TICKERS = [
    "ACES", "ADRO", "AKRA", "AMMN", "AMRT", "ANTM", "ARTO", "ASII", "BBCA",
    "BBNI", "BBRI", "BBTN", "BMRI", "BRIS", "BRPT", "BUKA", "CPIN", "ESSA",
    "EXCL", "GGRM", "GOTO", "HRUM", "ICBP", "INCO", "INDF", "INKP", "INTP",
    "ISAT", "ITMG", "JPFA", "KLBF", "MAPI", "MBMA", "MDKA", "MEDC", "MTEL",
    "PGAS", "PGEO", "PTBA", "SIDO", "SMGR", "SMRA", "TLKM", "TOWR", "UNTR",
    "UNVR",
]
TICKERS = [t + ".JK" for t in TICKERS]

START = "2021-10-01"
END = "2026-10-01"
TRAIN_END = "2025-09-30"   # latih: START..TRAIN_END, uji: sesudahnya
TRADING_DAYS = 252
MAX_MISSING = 0.05         # ticker dengan data kosong > 5% dibuang

MAX_WEIGHT = 0.10          # batas atas per saham (bisa 0.15)
RISK_FREE = 0.05           # per tahun

N_RUNS = 30
POP_SIZE = 200
GENERATIONS = 300
```

- [ ] **Step 3: Tulis `tests/conftest.py`**

```python
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
```

- [ ] **Step 4: Tulis tes yang gagal (`tests/test_data.py`)**

```python
import numpy as np
import pandas as pd
import pytest

import scrappingdata as sd


def _harga():
    idx = pd.bdate_range("2024-01-01", periods=100)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (100, 3)), axis=0)),
        index=idx, columns=["A.JK", "B.JK", "C.JK"])
    return df


def test_bersihkan_buang_ticker_ipo_belakangan_tanpa_memangkas_tanggal():
    df = _harga()
    df.loc[df.index[:40], "C.JK"] = np.nan          # IPO belakangan
    bersih, dibuang = sd.bersihkan_harga(df, max_missing=0.05)
    assert dibuang == ["C.JK"]
    assert list(bersih.columns) == ["A.JK", "B.JK"]
    assert len(bersih) == 100                        # tanggal tidak terpangkas


def test_bersihkan_isi_lubang_kecil():
    df = _harga()
    df.iloc[50, 0] = np.nan
    bersih, dibuang = sd.bersihkan_harga(df, max_missing=0.05)
    assert dibuang == []
    assert not bersih.isna().any().any()


def test_log_return():
    df = pd.DataFrame({"A": [100.0, 110.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    r = sd.hitung_log_return(df)
    assert r.iloc[0, 0] == pytest.approx(np.log(1.1))


def test_split_tidak_bocor_dan_train_end_bukan_hari_bursa():
    rets = sd.hitung_log_return(_harga())
    # 2024-02-03 adalah Sabtu
    train, test = sd.pisah_latih_uji(rets, "2024-02-03")
    assert train.index.max() <= pd.Timestamp("2024-02-03")
    assert test.index.min() > pd.Timestamp("2024-02-03")
    assert len(train) + len(test) == len(rets)


def test_split_uji_kosong_gagal_jelas():
    rets = sd.hitung_log_return(_harga())
    with pytest.raises(ValueError, match="uji"):
        sd.pisah_latih_uji(rets, "2030-01-01")


def test_mu_cov_tahunan():
    rets = sd.hitung_log_return(_harga())
    mu, cov = sd.hitung_mu_cov(rets, trading_days=252)
    assert mu.shape == (3,) and cov.shape == (3, 3)
    assert np.allclose(mu, rets.mean().values * 252)
    assert np.allclose(cov, rets.cov().values * 252)
```

- [ ] **Step 5: Jalankan, pastikan gagal**

Run: `python3 -m pytest tests/test_data.py -v`
Expected: FAIL (`AttributeError: module 'scrappingdata' has no attribute 'bersihkan_harga'`).

- [ ] **Step 6: Tulis ulang `src/scrappingdata.py`**

```python
"""Menyiapkan data untuk GA alokasi portofolio LQ45.

Jalankan dari root proyek:  python3 src/scrappingdata.py
Keluaran di data/: harga_adjclose.csv, log_return.csv, mu_cov_train.npz
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def unduh_harga(tickers, start, end):
    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    return raw["Close"].copy()


def bersihkan_harga(prices, max_missing):
    """Buang ticker dengan data kosong > max_missing SEBELUM dropna(),
    supaya satu ticker IPO belakangan tidak memangkas seluruh tanggal."""
    kosong = prices.isna().mean()
    dibuang = [c for c in prices.columns if kosong[c] > max_missing]
    bersih = prices.drop(columns=dibuang).ffill(limit=3).dropna()
    return bersih, dibuang


def hitung_log_return(prices):
    return np.log(prices / prices.shift(1)).dropna()


def pisah_latih_uji(rets, train_end):
    train = rets.loc[rets.index <= pd.Timestamp(train_end)]
    test = rets.loc[rets.index > pd.Timestamp(train_end)]
    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Data latih ({len(train)} hari) atau data uji ({len(test)} hari) "
            f"kosong untuk TRAIN_END={train_end}")
    return train, test


def hitung_mu_cov(train, trading_days=config.TRADING_DAYS):
    """Return dan kovarians tahunan dari data latih (input fitness GA)."""
    return train.mean().values * trading_days, train.cov().values * trading_days


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    mentah = unduh_harga(config.TICKERS, config.START, config.END)
    prices, dibuang = bersihkan_harga(mentah, config.MAX_MISSING)
    print(f"Ticker dibuang ({len(dibuang)}): {dibuang}")
    print(f"Ticker dipakai: {prices.shape[1]} | hari: {len(prices)} "
          f"({prices.index[0].date()} .. {prices.index[-1].date()})")

    min_saham = math.ceil(1 / config.MAX_WEIGHT)
    if prices.shape[1] < min_saham:
        sys.exit(f"Saham tersisa {prices.shape[1]} < {min_saham}; "
                 f"cap {config.MAX_WEIGHT} tidak layak. Tambah ticker.")

    rets = hitung_log_return(prices)
    train, test = pisah_latih_uji(rets, config.TRAIN_END)
    print(f"Latih: {len(train)} hari | Uji: {len(test)} hari")
    mu, cov = hitung_mu_cov(train)

    prices.to_csv(os.path.join(config.DATA_DIR, "harga_adjclose.csv"))
    rets.to_csv(os.path.join(config.DATA_DIR, "log_return.csv"))
    np.savez(os.path.join(config.DATA_DIR, "mu_cov_train.npz"),
             mu=mu, cov=cov, tickers=np.array(train.columns))
    print("Selesai. File tersimpan di data/.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Jalankan tes, pastikan lolos**

Run: `python3 -m pytest tests/test_data.py -v`
Expected: 6 passed.

- [ ] **Step 8: Unduh data LQ45 (butuh internet) dan periksa**

Run: `python3 src/scrappingdata.py`
Expected: daftar ticker dibuang dan jumlah ticker dipakai >= 10 (target 30-45), hari sekitar 1.200. Catat ticker yang dibuang untuk laporan.

- [ ] **Step 9: Commit**

```bash
git add config.py tests src/scrappingdata.py data
git commit -m "feat: config dan data pipeline LQ45"
```

---

### Task 2: Fungsi inti portofolio (Sharpe dan repair)

**Files:**
- Create: `src/portofolio.py`, `tests/test_portofolio.py`

**Interfaces:**
- Produces: `sharpe_ratio(w, mu, cov, rf) -> float`, `sharpe_population(pop, mu, cov, rf) -> np.ndarray` (shape P), `repair_weights(w, cap) -> np.ndarray`, `repair_population(pop, cap) -> np.ndarray`.

- [ ] **Step 1: Tulis tes yang gagal (`tests/test_portofolio.py`)**

```python
import numpy as np
import pytest

from portofolio import (repair_population, repair_weights, sharpe_population,
                        sharpe_ratio)


def _valid(w, cap):
    return (abs(w.sum() - 1) < 1e-9) and (w >= -1e-12).all() and (w <= cap + 1e-9).all()


def test_sharpe_manual():
    mu = np.array([0.10, 0.20])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    w = np.array([0.5, 0.5])
    expected = (0.15 - 0.05) / np.sqrt(0.25 * 0.04 + 0.25 * 0.09)
    assert sharpe_ratio(w, mu, cov, 0.05) == pytest.approx(expected)


def test_sharpe_population_sama_dengan_satuan():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    cov = A @ A.T / 5 + 0.01 * np.eye(5)
    mu = rng.normal(0.1, 0.05, 5)
    pop = rng.dirichlet(np.ones(5), size=7)
    got = sharpe_population(pop, mu, cov, 0.05)
    want = [sharpe_ratio(w, mu, cov, 0.05) for w in pop]
    assert np.allclose(got, want)


def test_sharpe_varians_nol_tidak_nan():
    mu = np.array([0.1, 0.1])
    cov = np.zeros((2, 2))
    val = sharpe_ratio(np.array([0.5, 0.5]), mu, cov, 0.05)
    assert np.isfinite(val)


@pytest.mark.parametrize("seed", range(20))
def test_repair_acak_selalu_valid(seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.05, 0.3, 15)          # ada negatif dan > cap
    out = repair_weights(w, 0.10)
    assert _valid(out, 0.10)


def test_repair_semua_nol_jadi_sama_rata():
    out = repair_weights(np.zeros(10), 0.10)
    assert np.allclose(out, 0.1)


def test_repair_terkonsentrasi_dipotong():
    w = np.array([1.0] + [0.0] * 19)
    out = repair_weights(w, 0.10)
    assert _valid(out, 0.10)
    assert out.max() <= 0.10 + 1e-9


def test_repair_cap_tidak_layak_error():
    with pytest.raises(ValueError, match="tidak layak"):
        repair_weights(np.ones(5) / 5, 0.10)   # 5 x 0.10 < 1


def test_repair_cap_pas_sama_rata():
    out = repair_weights(np.random.default_rng(0).random(10), 0.10)
    assert np.allclose(out, 0.1)


def test_repair_population_valid_dan_tidak_mengubah_yang_sudah_valid():
    rng = np.random.default_rng(3)
    pop = rng.normal(0.05, 0.2, (50, 20))
    out = repair_population(pop, 0.10)
    assert all(_valid(r, 0.10) for r in out)
    again = repair_population(out.copy(), 0.10)
    assert np.allclose(out, again)
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `python3 -m pytest tests/test_portofolio.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'portofolio'`).

- [ ] **Step 3: Tulis `src/portofolio.py`**

```python
"""Fungsi inti portofolio: Sharpe ratio dan operator repair batas bobot."""
import numpy as np

_EPS = 1e-12


def sharpe_ratio(w, mu, cov, rf):
    var = float(w @ cov @ w)
    vol = np.sqrt(max(var, _EPS))
    return float((w @ mu - rf) / vol)


def sharpe_population(pop, mu, cov, rf):
    ret = pop @ mu
    var = np.einsum("ij,jk,ik->i", pop, cov, pop)
    return (ret - rf) / np.sqrt(np.maximum(var, _EPS))


def repair_weights(w, cap):
    """Buat bobot valid: >= 0, jumlah 1, tiap bobot <= cap.

    Potong ke [0, cap], lalu bagikan kelebihan ke saham yang masih di bawah
    cap (sebanding bobotnya), ulangi sampai tidak ada yang melebihi cap.
    """
    n = len(w)
    if n * cap < 1 - 1e-12:
        raise ValueError(
            f"Cap {cap} tidak layak untuk {n} saham (perlu n x cap >= 1)")
    w = np.clip(np.asarray(w, dtype=float), 0.0, None)
    total = w.sum()
    w = w / total if total > 0 else np.full(n, 1.0 / n)

    for _ in range(n + 1):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = w < cap - _EPS
        if not free.any():
            break
        base = w[free] if w[free].sum() > 0 else np.ones(free.sum())
        w[free] += excess * base / base.sum()
    return np.clip(w, 0.0, cap)


def repair_population(pop, cap):
    """Repair seluruh populasi; baris yang sudah valid hanya dinormalisasi."""
    pop = np.clip(pop, 0.0, None)
    sums = pop.sum(axis=1, keepdims=True)
    zero = (sums <= 0).ravel()
    pop = np.where(sums > 0, pop / np.where(sums > 0, sums, 1), 1.0 / pop.shape[1])
    bad = (pop > cap + _EPS).any(axis=1) | zero
    for i in np.where(bad)[0]:
        pop[i] = repair_weights(pop[i], cap)
    return pop
```

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `python3 -m pytest tests/test_portofolio.py -v`
Expected: semua passed. Jika `test_repair_cap_pas_sama_rata` gagal karena toleransi, naikkan toleransi `np.allclose` ke `atol=1e-6`; jangan melonggarkan `_valid`.

- [ ] **Step 5: Commit**

```bash
git add src/portofolio.py tests/test_portofolio.py
git commit -m "feat: sharpe dan repair bobot"
```

---

### Task 3: Baseline (SLSQP max-Sharpe dan 1/N)

**Files:**
- Create: `src/baseline.py`, `tests/test_baseline.py`

**Interfaces:**
- Consumes: `portofolio.sharpe_ratio`.
- Produces: `equal_weight(n, cap) -> np.ndarray`, `max_sharpe_slsqp(mu, cov, rf, cap, n_starts=6, seed=0) -> np.ndarray`.

- [ ] **Step 1: Tulis tes yang gagal (`tests/test_baseline.py`)**

```python
import numpy as np
import pytest

from baseline import equal_weight, max_sharpe_slsqp
from portofolio import sharpe_ratio


def _masalah(n=15, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    cov = A @ A.T / n * 0.1 + 0.02 * np.eye(n)
    mu = rng.normal(0.12, 0.08, n)
    return mu, cov


def test_equal_weight():
    w = equal_weight(20, 0.10)
    assert np.allclose(w, 0.05) and w.sum() == pytest.approx(1)


def test_equal_weight_cap_tidak_layak():
    with pytest.raises(ValueError, match="tidak layak"):
        equal_weight(5, 0.10)


def test_slsqp_memenuhi_batasan():
    mu, cov = _masalah()
    w = max_sharpe_slsqp(mu, cov, 0.05, 0.10)
    assert w.sum() == pytest.approx(1, abs=1e-6)
    assert (w >= -1e-9).all() and (w <= 0.10 + 1e-9).all()


def test_slsqp_lebih_baik_dari_sama_rata_dan_acak():
    mu, cov = _masalah()
    w = max_sharpe_slsqp(mu, cov, 0.05, 0.10)
    s = sharpe_ratio(w, mu, cov, 0.05)
    assert s >= sharpe_ratio(equal_weight(15, 0.10), mu, cov, 0.05) - 1e-9
    rng = np.random.default_rng(5)
    for _ in range(200):
        r = rng.dirichlet(np.ones(15))
        r = np.minimum(r, 0.10)
        r /= r.sum()
        if r.max() <= 0.10 + 1e-9:
            assert s >= sharpe_ratio(r, mu, cov, 0.05) - 1e-6


def test_slsqp_cap_tidak_layak():
    mu, cov = _masalah(5)
    with pytest.raises(ValueError, match="tidak layak"):
        max_sharpe_slsqp(mu, cov, 0.05, 0.10)
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `python3 -m pytest tests/test_baseline.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'baseline'`).

- [ ] **Step 3: Tulis `src/baseline.py`**

```python
"""Baseline pembanding: 1/N dan max-Sharpe (SLSQP) dengan batasan sama seperti GA."""
import numpy as np
from scipy.optimize import minimize

from portofolio import repair_weights, sharpe_ratio


def _cek_cap(n, cap):
    if n * cap < 1 - 1e-12:
        raise ValueError(
            f"Cap {cap} tidak layak untuk {n} saham (perlu n x cap >= 1)")


def equal_weight(n, cap):
    _cek_cap(n, cap)
    return np.full(n, 1.0 / n)


def max_sharpe_slsqp(mu, cov, rf, cap, n_starts=6, seed=0):
    """Maksimalkan Sharpe, long-only, jumlah 1, bobot <= cap.

    Dicoba dari beberapa titik awal (1/N dan acak yang di-repair); diambil
    hasil terbaik karena Sharpe tidak konveks pada w.
    """
    n = len(mu)
    _cek_cap(n, cap)
    rng = np.random.default_rng(seed)
    starts = [np.full(n, 1.0 / n)]
    starts += [repair_weights(rng.dirichlet(np.ones(n)), cap)
               for _ in range(n_starts - 1)]

    def neg_sharpe(w):
        return -sharpe_ratio(w, mu, cov, rf)

    best_w, best_val = None, np.inf
    for x0 in starts:
        res = minimize(neg_sharpe, x0, method="SLSQP",
                       bounds=[(0.0, cap)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       options={"maxiter": 500, "ftol": 1e-12})
        if res.success or res.fun < best_val:
            w = repair_weights(res.x, cap)       # rapikan galat numerik kecil
            val = neg_sharpe(w)
            if val < best_val:
                best_w, best_val = w, val
    return best_w
```

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `python3 -m pytest tests/test_baseline.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/baseline.py tests/test_baseline.py
git commit -m "feat: baseline SLSQP dan 1/N"
```

---

### Task 4: Genetic Algorithm

**Files:**
- Create: `src/GeneticAlgorithm.py`, `tests/test_ga.py`

**Interfaces:**
- Consumes: `portofolio.sharpe_population`, `portofolio.repair_population`, `config.POP_SIZE`, `config.GENERATIONS`.
- Produces:
  - `GAResult` (dataclass): `seed: int`, `weights: np.ndarray`, `fitness: float`, `hist_best: np.ndarray` (len = generations+1), `hist_mean: np.ndarray` (len = generations+1).
  - `run_ga(mu, cov, cap, rf, seed, pop_size=config.POP_SIZE, generations=config.GENERATIONS, tournament_k=3, crossover_prob=0.9, eta_c=15.0, mutation_prob=None, mutation_sigma=0.02, elite=2) -> GAResult` (`mutation_prob=None` berarti 1/N per gen).
  - `run_ga_batch(mu, cov, cap, rf, seeds, **kwargs) -> list[GAResult]`.
  - `pilih_terbaik(results) -> GAResult` (fitness latih tertinggi), `pilih_median(results) -> GAResult`.

- [ ] **Step 1: Tulis tes yang gagal (`tests/test_ga.py`)**

```python
import numpy as np
import pytest

from baseline import max_sharpe_slsqp
from GeneticAlgorithm import (pilih_median, pilih_terbaik, run_ga,
                              run_ga_batch)
from portofolio import sharpe_ratio


def _masalah(n=15, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    cov = A @ A.T / n * 0.1 + 0.02 * np.eye(n)
    mu = rng.normal(0.12, 0.08, n)
    return mu, cov


KW = dict(pop_size=60, generations=80)


def test_hasil_valid_dan_riwayat_benar():
    mu, cov = _masalah()
    r = run_ga(mu, cov, 0.10, 0.05, seed=1, **KW)
    assert r.weights.sum() == pytest.approx(1, abs=1e-9)
    assert (r.weights >= -1e-12).all() and (r.weights <= 0.10 + 1e-9).all()
    assert len(r.hist_best) == 81 and len(r.hist_mean) == 81
    assert r.fitness == pytest.approx(sharpe_ratio(r.weights, mu, cov, 0.05))


def test_fitness_terbaik_tidak_turun_elitisme():
    mu, cov = _masalah()
    r = run_ga(mu, cov, 0.10, 0.05, seed=2, **KW)
    assert (np.diff(r.hist_best) >= -1e-12).all()
    assert r.hist_best[-1] > r.hist_best[0]


def test_reprodusibel_dengan_seed_sama():
    mu, cov = _masalah()
    a = run_ga(mu, cov, 0.10, 0.05, seed=7, **KW)
    b = run_ga(mu, cov, 0.10, 0.05, seed=7, **KW)
    c = run_ga(mu, cov, 0.10, 0.05, seed=8, **KW)
    assert np.array_equal(a.weights, b.weights)
    assert not np.array_equal(a.weights, c.weights)


def test_populasi_ganjil_ditolak():
    mu, cov = _masalah()
    with pytest.raises(ValueError, match="genap"):
        run_ga(mu, cov, 0.10, 0.05, seed=0, pop_size=51, generations=5)


def test_cap_tidak_layak_ditolak():
    mu, cov = _masalah(5)
    with pytest.raises(ValueError, match="tidak layak"):
        run_ga(mu, cov, 0.10, 0.05, seed=0, pop_size=20, generations=5)


def test_mendekati_slsqp():
    mu, cov = _masalah()
    ref = sharpe_ratio(max_sharpe_slsqp(mu, cov, 0.05, 0.10), mu, cov, 0.05)
    r = run_ga(mu, cov, 0.10, 0.05, seed=3, pop_size=200, generations=300)
    assert r.fitness >= 0.95 * ref


def test_batch_dan_pemilihan():
    mu, cov = _masalah()
    rs = run_ga_batch(mu, cov, 0.10, 0.05, seeds=range(5), pop_size=40,
                      generations=20)
    assert [r.seed for r in rs] == [0, 1, 2, 3, 4]
    best = pilih_terbaik(rs)
    assert best.fitness == max(r.fitness for r in rs)
    med = pilih_median(rs)
    urut = sorted(r.fitness for r in rs)
    assert med.fitness == urut[len(urut) // 2]
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `python3 -m pytest tests/test_ga.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'GeneticAlgorithm'`).

- [ ] **Step 3: Tulis `src/GeneticAlgorithm.py`**

```python
"""Genetic Algorithm real-coded untuk memaksimalkan Sharpe ratio portofolio.

Kromosom = vektor bobot. Operator: seleksi turnamen, SBX, mutasi Gaussian,
elitisme, dan repair (potong ke cap lalu normalisasi) setelah variasi.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from portofolio import repair_population, sharpe_population  # noqa: E402


@dataclass
class GAResult:
    seed: int
    weights: np.ndarray
    fitness: float
    hist_best: np.ndarray
    hist_mean: np.ndarray


def _seleksi_turnamen(fit, k, rng):
    p = len(fit)
    cand = rng.integers(0, p, size=(p, k))
    return cand[np.arange(p), np.argmax(fit[cand], axis=1)]


def _sbx(p1, p2, eta, rng):
    u = rng.random(p1.shape)
    beta = np.where(u <= 0.5,
                    (2 * u) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return c1, c2


def run_ga(mu, cov, cap, rf, seed, pop_size=config.POP_SIZE,
           generations=config.GENERATIONS, tournament_k=3,
           crossover_prob=0.9, eta_c=15.0, mutation_prob=None,
           mutation_sigma=0.02, elite=2):
    if pop_size % 2:
        raise ValueError("pop_size harus genap")
    n = len(mu)
    if n * cap < 1 - 1e-12:
        raise ValueError(
            f"Cap {cap} tidak layak untuk {n} saham (perlu n x cap >= 1)")
    pm = 1.0 / n if mutation_prob is None else mutation_prob
    rng = np.random.default_rng(seed)

    pop = repair_population(rng.dirichlet(np.ones(n), size=pop_size), cap)
    fit = sharpe_population(pop, mu, cov, rf)
    hist_best = [fit.max()]
    hist_mean = [fit.mean()]

    for _ in range(generations):
        elit_idx = np.argsort(fit)[-elite:]
        elit = pop[elit_idx].copy()

        parents = pop[_seleksi_turnamen(fit, tournament_k, rng)]
        p1, p2 = parents[0::2], parents[1::2]
        c1, c2 = _sbx(p1, p2, eta_c, rng)
        do_x = (rng.random(len(p1)) < crossover_prob)[:, None]
        c1 = np.where(do_x, c1, p1)
        c2 = np.where(do_x, c2, p2)
        child = np.empty_like(pop)
        child[0::2], child[1::2] = c1, c2

        mask = rng.random(child.shape) < pm
        child = child + mask * rng.normal(0.0, mutation_sigma, child.shape)
        child = repair_population(child, cap)

        child_fit = sharpe_population(child, mu, cov, rf)
        worst = np.argsort(child_fit)[:elite]
        child[worst] = elit
        child_fit[worst] = fit[elit_idx]

        pop, fit = child, child_fit
        hist_best.append(fit.max())
        hist_mean.append(fit.mean())

    i = int(np.argmax(fit))
    return GAResult(seed=seed, weights=pop[i].copy(), fitness=float(fit[i]),
                    hist_best=np.array(hist_best), hist_mean=np.array(hist_mean))


def run_ga_batch(mu, cov, cap, rf, seeds, **kwargs):
    return [run_ga(mu, cov, cap, rf, seed=s, **kwargs) for s in seeds]


def pilih_terbaik(results):
    """Run dengan fitness LATIH tertinggi (bukan berdasarkan data uji)."""
    return max(results, key=lambda r: r.fitness)


def pilih_median(results):
    urut = sorted(results, key=lambda r: r.fitness)
    return urut[len(urut) // 2]
```

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `python3 -m pytest tests/test_ga.py -v`
Expected: 7 passed. Jika `test_mendekati_slsqp` gagal, cek dulu ukuran gap (cetak `r.fitness` dan `ref`) sebelum mengubah parameter: naikkan `mutation_sigma` ke 0.03 atau `generations`; jangan melonggarkan ambang 0.95 tanpa memberi tahu pengguna.

- [ ] **Step 5: Commit**

```bash
git add src/GeneticAlgorithm.py tests/test_ga.py
git commit -m "feat: genetic algorithm real-coded SBX"
```

---

### Task 5: Evaluasi out-of-sample

**Files:**
- Create: `src/evaluasi.py`, `tests/test_evaluasi.py`

**Interfaces:**
- Produces: `portfolio_daily_returns(rets, w) -> pd.Series` (return sederhana harian, bobot konstan), `max_drawdown(cum) -> float`, `drawdown_series(cum) -> pd.Series`, `hitung_metrik(rets, w, rf, trading_days=252) -> dict` dengan kunci `return_tahunan, volatilitas, sharpe, max_drawdown`, `bandingkan(rets, bobot: dict[str, np.ndarray], rf) -> pd.DataFrame` (indeks nama strategi, kolom empat metrik), `nilai_kumulatif(rets, w) -> pd.Series` (mulai dari 1 sebelum hari pertama, yaitu `cumprod(1+r)`).

- [ ] **Step 1: Tulis tes yang gagal (`tests/test_evaluasi.py`)**

```python
import numpy as np
import pandas as pd
import pytest

import evaluasi as ev


def _rets(vals):
    idx = pd.bdate_range("2025-01-01", periods=len(vals))
    return pd.DataFrame(vals, index=idx, columns=["A", "B"])


def test_return_harian_pakai_expm1():
    rets = _rets([[np.log(1.10), np.log(1.00)]])
    r = ev.portfolio_daily_returns(rets, np.array([0.5, 0.5]))
    assert r.iloc[0] == pytest.approx(0.05)


def test_max_drawdown_manual():
    cum = pd.Series([1.0, 1.2, 0.9, 1.0, 0.6])
    assert ev.max_drawdown(cum) == pytest.approx(0.6 / 1.2 - 1)
    dd = ev.drawdown_series(cum)
    assert dd.iloc[1] == 0 and dd.min() == pytest.approx(-0.5)


def test_metrik_konstan_vol_nol_tidak_nan():
    rets = _rets(np.zeros((50, 2)))
    m = ev.hitung_metrik(rets, np.array([0.5, 0.5]), rf=0.05)
    assert m["volatilitas"] == 0
    assert np.isfinite(m["sharpe"])
    assert m["max_drawdown"] == 0


def test_metrik_dan_bandingkan():
    rng = np.random.default_rng(0)
    rets = _rets(rng.normal(0.0005, 0.01, (200, 2)))
    w = np.array([0.7, 0.3])
    m = ev.hitung_metrik(rets, w, rf=0.05)
    r = ev.portfolio_daily_returns(rets, w)
    assert m["return_tahunan"] == pytest.approx(r.mean() * 252)
    assert m["volatilitas"] == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert m["sharpe"] == pytest.approx((m["return_tahunan"] - 0.05) / m["volatilitas"])
    tabel = ev.bandingkan(rets, {"X": w, "Y": np.array([0.5, 0.5])}, rf=0.05)
    assert list(tabel.index) == ["X", "Y"]
    assert set(tabel.columns) == {"return_tahunan", "volatilitas", "sharpe", "max_drawdown"}
    assert ev.nilai_kumulatif(rets, w).iloc[-1] == pytest.approx((1 + r).prod())
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `python3 -m pytest tests/test_evaluasi.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'evaluasi'`).

- [ ] **Step 3: Tulis `src/evaluasi.py`**

```python
"""Evaluasi out-of-sample: bobot dari data latih, dinilai di data uji.

Return portofolio dihitung dari return sederhana (expm1 dari log return) dengan
bobot konstan per hari; metrik tahunan memakai 252 hari bursa.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_daily_returns(rets, w):
    return pd.Series(np.expm1(rets.values) @ np.asarray(w), index=rets.index)


def nilai_kumulatif(rets, w):
    return (1 + portfolio_daily_returns(rets, w)).cumprod()


def drawdown_series(cum):
    return cum / cum.cummax() - 1


def max_drawdown(cum):
    return float(drawdown_series(cum).min())


def hitung_metrik(rets, w, rf, trading_days=TRADING_DAYS):
    r = portfolio_daily_returns(rets, w)
    ret = float(r.mean() * trading_days)
    vol = float(r.std(ddof=1) * np.sqrt(trading_days))
    sharpe = (ret - rf) / vol if vol > 1e-12 else 0.0
    return {"return_tahunan": ret, "volatilitas": vol, "sharpe": sharpe,
            "max_drawdown": max_drawdown((1 + r).cumprod())}


def bandingkan(rets, bobot, rf):
    baris = {nama: hitung_metrik(rets, w, rf) for nama, w in bobot.items()}
    return pd.DataFrame.from_dict(baris, orient="index")
```

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `python3 -m pytest tests/test_evaluasi.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/evaluasi.py tests/test_evaluasi.py
git commit -m "feat: evaluasi out-of-sample"
```

---

### Task 6: Visualisasi (enam gambar)

**Files:**
- Create: `src/visualisasi.py`, `tests/test_visualisasi.py`

**Interfaces:**
- Consumes: `GAResult` (atribut `hist_best`, `hist_mean`, `fitness`, `weights`), pandas `Series` nilai kumulatif dan drawdown, tabel metrik dari `evaluasi.bandingkan`.
- Produces (semua menerima `path` PNG sebagai argumen terakhir dan mengembalikan `path`):
  - `plot_konvergensi(results, path)`
  - `plot_sebaran_sharpe(results, sharpe_slsqp, path)`
  - `plot_bobot(tickers, w_ga, w_qp, path)`
  - `plot_kumulatif(kum: dict[str, pd.Series], path)`
  - `plot_metrik(tabel: pd.DataFrame, path)`
  - `plot_drawdown(dd: dict[str, pd.Series], path)`

Catatan eksekutor: sebelum menulis kode gambar, panggil skill `dataviz` agar palet dan gaya konsisten. Kode di bawah memakai palet Okabe-Ito yang ramah buta warna, ganti jika skill menyarankan lain.

- [ ] **Step 1: Tulis tes yang gagal (`tests/test_visualisasi.py`)**

```python
import os

import numpy as np
import pandas as pd

import visualisasi as vz
from GeneticAlgorithm import GAResult


def _results(k=4, g=10):
    rng = np.random.default_rng(0)
    out = []
    for s in range(k):
        best = np.sort(rng.random(g + 1))
        out.append(GAResult(s, np.full(12, 1 / 12), float(best[-1]), best, best - 0.1))
    return out


def _ok(path):
    assert os.path.exists(path) and os.path.getsize(path) > 1000


def test_semua_gambar_terbuat(tmp_path):
    res = _results()
    tick = [f"T{i}.JK" for i in range(12)]
    w1, w2 = np.full(12, 1 / 12), np.r_[np.full(6, 0.1), np.full(6, 0.4 / 6)]
    idx = pd.bdate_range("2025-10-01", periods=30)
    kum = {n: pd.Series(1 + np.arange(30) * 0.001 * (i + 1), index=idx)
           for i, n in enumerate(["GA", "SLSQP", "1/N"])}
    dd = {n: s / s.cummax() - 1 for n, s in kum.items()}
    tabel = pd.DataFrame({"return_tahunan": [0.1, 0.12, 0.09],
                          "volatilitas": [0.2, 0.19, 0.21],
                          "sharpe": [0.25, 0.37, 0.19],
                          "max_drawdown": [-0.1, -0.08, -0.12]},
                         index=["GA", "SLSQP", "1/N"])
    for f, args in [
        (vz.plot_konvergensi, (res,)),
        (vz.plot_sebaran_sharpe, (res, 0.9)),
        (vz.plot_bobot, (tick, w1, w2)),
        (vz.plot_kumulatif, (kum,)),
        (vz.plot_metrik, (tabel,)),
        (vz.plot_drawdown, (dd,)),
    ]:
        p = str(tmp_path / f"{f.__name__}.png")
        assert f(*args, p) == p
        _ok(p)
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `python3 -m pytest tests/test_visualisasi.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'visualisasi'`).

- [ ] **Step 3: Tulis `src/visualisasi.py`**

```python
"""Enam visual utama untuk laporan."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito
WARNA = {"GA": "#0072B2", "SLSQP": "#D55E00", "1/N": "#009E73",
         "GA-median": "#56B4E9"}
_DEF = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


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
           title="Sebaran hasil 30 run GA")
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
```

- [ ] **Step 4: Jalankan, pastikan lolos**

Run: `python3 -m pytest tests/test_visualisasi.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/visualisasi.py tests/test_visualisasi.py
git commit -m "feat: enam visual hasil"
```

---

### Task 7: `main.py` dan jalankan end-to-end

**Files:**
- Modify: `main.py` (saat ini kosong)
- Create: `tests/test_main.py`

**Interfaces:**
- Consumes: semua modul di atas. Membaca `data/log_return.csv` dan `data/mu_cov_train.npz` (jika belum ada, memanggil `scrappingdata.main()`).
- Produces: `jalankan(data_dir, hasil_dir, n_runs, pop_size, generations) -> dict` berisi `tabel` (DataFrame metrik uji), `bobot` (dict nama -> vektor), `ga_results` (list). Menulis ke `hasil/`: `bobot.csv`, `metrik_uji.csv`, `ga_semua_run.csv`, dan enam PNG `1_konvergensi.png` .. `6_drawdown.png`.

- [ ] **Step 1: Tulis tes yang gagal (`tests/test_main.py`)**

```python
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
```

- [ ] **Step 2: Jalankan, pastikan gagal**

Run: `python3 -m pytest tests/test_main.py -v`
Expected: FAIL (`AttributeError: module 'main' has no attribute 'jalankan'`).

- [ ] **Step 3: Tulis `main.py`**

```python
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
```

- [ ] **Step 4: Jalankan tes, pastikan lolos**

Run: `python3 -m pytest tests/test_main.py -v`
Expected: 1 passed.

- [ ] **Step 5: Jalankan seluruh tes**

Run: `python3 -m pytest tests -v`
Expected: semua passed.

- [ ] **Step 6: Jalankan pipeline sungguhan dan periksa hasil**

Run: `python3 main.py`
Expected: cetak baseline latih, ringkasan 30 run GA (std kecil), tabel metrik uji, dan `hasil/` berisi 6 PNG + 3 CSV. Buka PNG dan pastikan terbaca. Laporkan apa adanya ke pengguna, termasuk bila GA tidak mengalahkan SLSQP atau 1/N di data uji (itu hasil yang sah, bukan bug).

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_main.py hasil
git commit -m "feat: pipeline main dan hasil"
```
