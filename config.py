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
