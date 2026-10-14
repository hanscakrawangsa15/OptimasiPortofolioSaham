"""Pemuatan data dan skenario walk-forward. SUB-PROYEK A.

Spesifikasi: docs/superpowers/specs/2026-10-14-optimasi-lq45-metaheuristik-design.md, bagian 5.
Sementara, fungsi jendela masih ada di walkforward.py; dilebur ke sini di sub-proyek A.
"""


def muat_data(data_dir):
    """Kembalikan (return_df, harga_raw_df) dari data/ (return sederhana, harga raw)."""
    raise NotImplementedError("Sub-proyek A")


def muat_skenario(data_dir, K_list, sigma_mult, D):
    """Kembalikan list[Skenario]: 5 jendela walk-forward + kasus tetap, tiap K dan sigma_mult."""
    raise NotImplementedError("Sub-proyek A")
