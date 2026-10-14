"""Konfigurasi proyek optimasi portofolio LQ45 (batas risiko, kardinalitas, lot)."""
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
TRAIN_END = "2025-09-30"   # kasus tetap (perbandingan algoritma): latih s.d. tanggal ini
TRADING_DAYS = 252
MAX_MISSING = 0.05         # ticker dengan data kosong > 5% (Adj Close) dibuang

RISK_FREE = 0.05           # per tahun

# ---- Masalah optimasi: maksimasi return ekspektasi di bawah batas risiko ----
K_LIST = [1, 2, 3, 4, 5, 7, 10]          # jumlah saham maksimum
SIGMA_MULT = [1.5, 2.0, 2.5]             # batas risiko = kelipatan vol portofolio 1/N
D_LEVELS = [5e6, 10e6, 20e6, 35e6, 50e6, 100e6]   # modal (Rp)
LOT_SIZE = 100                           # 1 lot = 100 lembar; harga beli = harga RAW
MAX_WEIGHT = 0.20                        # batas per saham ...
CAP_BERLAKU_MULAI_K = 7                  # ... hanya berlaku untuk K >= 7

# ---- Walk-forward (hari bursa) ----
WINDOW_TRAIN = 504
WINDOW_TEST = 126
WINDOW_STEP = 126

N_RUNS = 30
