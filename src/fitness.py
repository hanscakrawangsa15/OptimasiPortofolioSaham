"""Evaluator (pintu tunggal evaluasi + hitungan anggaran) dan aturan kelayakan Deb. SUB-PROYEK A.

Spesifikasi: docs/superpowers/specs/2026-10-14-optimasi-lq45-metaheuristik-design.md, bagian 5.
"""


class Evaluator:
    """Evaluator(problem, budget); evaluate_batch(X) menghitung R, sigma, viol dan mencatat riwayat."""

    def __init__(self, problem, budget):
        raise NotImplementedError("Sub-proyek A")


def deb_better(a, b):
    """True bila solusi a lebih baik dari b menurut aturan Deb."""
    raise NotImplementedError("Sub-proyek A")
