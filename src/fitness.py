"""Objektif, aturan kelayakan Deb, dan Evaluator (pintu tunggal evaluasi)."""
import numpy as np

import config
from decoder import decode_batch
from problem import Result


def hitung(lots, w, kas, P):
    """Objektif R dan pelanggaran risiko untuk sekumpulan portofolio (batch)."""
    R = w @ P.mu + kas * P.rf
    var = np.einsum("ij,jk,ik->i", w, P.cov, w)
    sigma = np.sqrt(np.maximum(var, 0.0))
    viol = np.maximum(0.0, sigma - P.sigma_max)
    return R, sigma, viol


def deb_urut(R, viol):
    """Indeks dari terbaik ke terburuk: layak dulu (R besar), lalu tak layak (viol kecil)."""
    layak = viol <= 0.0
    return np.lexsort((-R, np.where(layak, 0.0, viol), ~layak))


def deb_rank(R, viol):
    """Peringkat tiap individu (0 = terbaik)."""
    urut = deb_urut(R, viol)
    rank = np.empty(len(urut), dtype=np.int64)
    rank[urut] = np.arange(len(urut))
    return rank


def deb_better(a, b):
    """a, b = (R, viol). True bila a lebih baik daripada b menurut aturan Deb."""
    (Ra, va), (Rb, vb) = a, b
    if va <= 0 and vb <= 0:
        return Ra > Rb
    if va <= 0:
        return True
    if vb <= 0:
        return False
    return va < vb


def deb_lebih_baik(R1, v1, R2, v2):
    """Aturan Deb versi vektor: True di mana (R1, v1) lebih baik daripada (R2, v2)."""
    l1, l2 = v1 <= 0.0, v2 <= 0.0
    return np.where(l1 & l2, R1 > R2, np.where(l1, True, np.where(l2, False, v1 < v2)))


class AnggaranHabis(Exception):
    pass


class Evaluator:
    """Dekode + objektif + risiko + anggaran + solusi terbaik + riwayat."""

    def __init__(self, P, budget=config.BUDGET_EVAL, step=config.HISTORY_STEP):
        self.P, self.budget, self.step = P, budget, step
        self.n_eval = 0
        self.n_eval_best = 0
        self.best = None                 # dict: x, lots, R, sigma, viol
        self.history = []                # [(n_eval, R terbaik yang layak atau nan)]

    def evaluate(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] != self.P.n_genes:
            raise ValueError(f"lebar kromosom {X.shape[1]} != n_genes {self.P.n_genes}")
        if self.n_eval + len(X) > self.budget:
            raise AnggaranHabis(f"{self.n_eval} + {len(X)} > anggaran {self.budget}")
        hasil_R, hasil_s, hasil_v = [], [], []
        i = 0
        while i < len(X):                                   # pecah di kelipatan `step` agar riwayat tepat
            batas = self.step - (self.n_eval % self.step)
            chunk = X[i:i + batas]
            lots, w, kas = decode_batch(chunk, self.P)
            R, s, v = hitung(lots, w, kas, self.P)
            j = deb_urut(R, v)[0]
            cand = (R[j], v[j])
            if self.best is None or deb_better(cand, (self.best["R"], self.best["viol"])):
                self.best = dict(x=chunk[j].copy(), lots=lots[j].copy(), R=R[j], sigma=s[j], viol=v[j])
                self.n_eval_best = self.n_eval + j + 1
            self.n_eval += len(chunk)
            if self.n_eval % self.step == 0:
                b = self.best
                self.history.append((self.n_eval, b["R"] if b["viol"] <= 0 else np.nan))
            hasil_R.append(R); hasil_s.append(s); hasil_v.append(v)
            i += len(chunk)
        return np.concatenate(hasil_R), np.concatenate(hasil_s), np.concatenate(hasil_v)

    def hasil(self, seed, seconds):
        b = self.best
        return Result(lots=b["lots"], R=float(b["R"]), sigma=float(b["sigma"]), viol=float(b["viol"]),
                      feasible=bool(b["viol"] <= 0), history=list(self.history), n_eval=self.n_eval,
                      seconds=seconds, seed=seed, x=b["x"], n_eval_best=self.n_eval_best)
