"""Kelas Problem dan Result. SUB-PROYEK A.

Problem (frozen dataclass): mu, cov, price, D, K, sigma_max, tickers, cap, lot=100, rf=0.05.
Result: lots, R, sigma, viol, feasible, history, n_eval, seconds, seed, status.
Semua algoritma menerima Problem dan mengembalikan Result: run(problem, budget, seed) -> Result.
Spesifikasi: docs/superpowers/specs/2026-10-14-optimasi-lq45-metaheuristik-design.md, bagian 5.
"""
