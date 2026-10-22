"""Dekoder kromosom 2N+1 -> lot bulat. Deterministik; hanya batas risiko yang bisa dilanggar."""
import numpy as np

_EPS = 1e-12


def split(x, N):
    """Pisahkan kromosom menjadi (kunci, bobot mentah, gen kas)."""
    x = np.asarray(x, float)
    return x[..., :N], x[..., N:2 * N], x[..., 2 * N]


def _cap_redistribute(w, pilih, cap):
    # Potong bobot di atas cap, bagikan kelebihan ke saham terpilih yang belum penuh
    # (sebanding bobotnya), ulangi sampai stabil. Kelebihan yang tak bisa dibagi menjadi kas.
    w = w.copy()
    for _ in range(w.shape[1] + 1):
        lebih = w > cap + _EPS
        if not lebih.any():
            break
        excess = np.where(lebih, w - cap, 0.0).sum(1, keepdims=True)
        w = np.where(lebih, cap, w)
        bebas = pilih & (w < cap - _EPS)
        dasar = np.where(bebas, w, 0.0)
        ds = dasar.sum(1, keepdims=True)
        dasar = np.where(ds > 0, dasar, bebas.astype(float))   # semua bobot bebas nol -> sama rata
        ds = dasar.sum(1, keepdims=True)
        w = w + np.where(ds > 0, excess * dasar / np.where(ds > 0, ds, 1.0), 0.0)
    return np.minimum(w, cap)


def bobot_target(X, P):
    """Kromosom (M, 2N+1) -> (himpunan terpilih (M,N) bool, bobot target (M,N), g0 (M,))."""
    X = np.atleast_2d(np.asarray(X, float))
    M, N = X.shape[0], P.n
    if X.shape[1] != P.n_genes:
        raise ValueError(f"lebar kromosom {X.shape[1]} != n_genes {P.n_genes}")
    keys, raw, g0 = split(X, N)
    raw = np.clip(raw, 0.0, 1.0)
    g0 = np.clip(g0, 0.0, 1.0)

    layak = P.layak_beli                                       # saham terjangkau, lalu K kunci terbesar
    urut = np.argsort(-np.where(layak, keys, -np.inf), axis=1, kind="stable")[:, :P.K]
    pilih = np.zeros((M, N), bool)
    np.put_along_axis(pilih, urut, True, axis=1)
    pilih &= layak

    r = np.where(pilih, raw, 0.0)                              # (1 - g0) dibagi sebanding bobot mentah
    s = r.sum(1, keepdims=True)
    sama = pilih / np.maximum(pilih.sum(1, keepdims=True), 1)
    frac = np.where(s > 0, r / np.where(s > 0, s, 1.0), sama)
    w = (1.0 - g0)[:, None] * frac

    if P.cap < 1.0:
        w = _cap_redistribute(w, pilih, P.cap)
    return pilih, w, g0


def decode_batch(X, P):
    """Kromosom (M, 2N+1) -> (lots (M,N) int, bobot akhir (M,N), kas (M,))."""
    pilih, w, g0 = bobot_target(X, P)
    lots = np.floor(w * P.D / P.harga_lot).astype(np.int64)
    w_akhir = lots * P.harga_lot / P.D
    kas = np.maximum(1.0 - w_akhir.sum(1), 0.0)
    return lots, w_akhir, kas


def decode(x, P):
    """Satu kromosom -> (lots (N,), bobot (N,), kas)."""
    lots, w, kas = decode_batch(np.asarray(x, float)[None, :], P)
    return lots[0], w[0], kas[0]


def init_population(P, rng, n, n_seed=0):
    """Populasi awal (n, 2N+1): acak seragam, g0 di [0; 0,5]; n_seed baris pertama disemai top-K."""
    N = P.n
    X = np.empty((n, P.n_genes))
    X[:, :N] = rng.random((n, N))
    X[:, N:2 * N] = rng.random((n, N))
    X[:, 2 * N] = rng.uniform(0.0, 0.5, n)
    skor = (P.mu - P.rf) / P.vol                               # heuristik: return lebih per unit risiko
    skor = (skor - skor.min()) / (skor.max() - skor.min() + 1e-12)
    for i in range(min(n_seed, n)):
        X[i, :N] = np.clip(skor + rng.normal(0.0, 0.05, N), 0.0, 1.0)
        X[i, N:2 * N] = rng.uniform(0.3, 1.0, N)
        X[i, 2 * N] = rng.uniform(0.0, 0.1)
    return X
