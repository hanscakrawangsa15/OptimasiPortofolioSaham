"""Dekoder bersama: vektor 2N (kunci + bobot mentah) -> lot yang layak. SUB-PROYEK A.

Spesifikasi: docs/superpowers/specs/2026-10-14-optimasi-lq45-metaheuristik-design.md, bagian 5.
"""


def decode(x, problem):
    """Satu vektor x (2N) -> lots (N,) bilangan bulat."""
    raise NotImplementedError("Sub-proyek A")


def decode_batch(X, problem):
    """Matriks X (M, 2N) -> lots (M, N); harus identik dengan decode per baris."""
    raise NotImplementedError("Sub-proyek A")
