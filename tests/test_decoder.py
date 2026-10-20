"""Tes dekoder 2N+1: invarian, kesetaraan dengan dekoder referensi, gen kas, dan contoh nyata."""
import numpy as np
import pytest

from data import NAMA_DATA, buat_problem, muat_jendela
from decoder import (bobot_target, decode, decode_batch, init_population,
                     split)

K_UJI = [2, 3, 5, 7, 10]


def _w(nama):
    return muat_jendela(nama)


def _acak(P, rng, m):
    X = rng.random((m, P.n_genes))
    X[:, -1] = rng.uniform(0.0, 1.0, m)
    return X


@pytest.mark.parametrize("nama", ["kasus_tetap", "jendela_3"])
@pytest.mark.parametrize("K", K_UJI)
def test_invarian_dekoder(nama, K):
    P = buat_problem(_w(nama), K)
    rng = np.random.default_rng(12345)
    lots, w, kas = decode_batch(_acak(P, rng, 2000), P)
    assert lots.dtype.kind == "i" and (lots >= 0).all()
    assert ((lots > 0).sum(1) <= K).all()
    assert (w <= P.cap + 1e-9).all()
    assert ((lots * P.harga_lot).sum(1) <= P.D * (1 + 1e-12)).all()
    assert (kas >= 0).all() and np.allclose(w.sum(1) + kas, 1.0)
    assert (lots[:, ~P.layak_beli] == 0).all()


def _referensi(x, P):
    """Dekoder ditulis ulang dengan loop biasa (independen dari versi vektor)."""
    N = P.n
    keys, raw, g0 = x[:N], np.clip(x[N:2 * N], 0, 1), min(max(x[2 * N], 0.0), 1.0)
    kandidat = sorted((i for i in range(N) if P.harga_lot[i] <= P.D * P.cap), key=lambda i: (-keys[i], i))
    S = kandidat[:P.K]
    w = np.zeros(N)
    if S:
        tot = sum(raw[i] for i in S)
        for i in S:
            w[i] = (1 - g0) * (raw[i] / tot if tot > 0 else 1.0 / len(S))
    if P.cap < 1.0:
        for _ in range(N + 1):
            lebih = [i for i in S if w[i] > P.cap + 1e-12]
            if not lebih:
                break
            excess = sum(w[i] - P.cap for i in lebih)
            for i in lebih:
                w[i] = P.cap
            bebas = [i for i in S if w[i] < P.cap - 1e-12]
            if not bebas:
                continue
            dasar = [w[i] for i in bebas]
            if sum(dasar) <= 0:
                dasar = [1.0] * len(bebas)
            for i, b in zip(bebas, dasar):
                w[i] += excess * b / sum(dasar)
    return np.array([int(np.floor(w[i] * P.D / P.harga_lot[i])) for i in range(N)])


@pytest.mark.parametrize("K", K_UJI)
def test_dekoder_vektor_sama_dengan_referensi(K):
    P = buat_problem(_w("jendela_3"), K)
    X = _acak(P, np.random.default_rng(7), 300)
    lots_v, _, _ = decode_batch(X, P)
    for x, lv in zip(X, lots_v):
        assert (_referensi(x, P) == lv).all()
    assert (decode(X[0], P)[0] == lots_v[0]).all()


def test_gen_kas_satu_semua_kas():
    P = buat_problem(_w("kasus_tetap"), 2)
    x = np.zeros(P.n_genes)
    x[-1] = 1.0
    lots, w, kas = decode(x, P)
    assert lots.sum() == 0 and kas == pytest.approx(1.0)


def test_satu_saham_sisa_kurang_dari_satu_lot():
    P = buat_problem(_w("kasus_tetap"), 1)
    rng = np.random.default_rng(3)
    for _ in range(100):
        x = rng.random(P.n_genes)
        x[-1] = 0.0
        lots, _, kas = decode(x, P)
        assert (lots > 0).sum() == 1
        assert kas < P.harga_lot[int(np.argmax(lots))] / P.D + 1e-12


def test_contoh_bris_isat_jendela_3():
    P = buat_problem(_w("jendela_3"), 2)
    ib, isa = P.tickers.index("BRIS"), P.tickers.index("ISAT")
    x = np.zeros(P.n_genes)
    x[ib] = x[isa] = 1.0
    x[P.n + ib], x[P.n + isa] = 0.55, 0.45
    x[-1] = 0.30
    lots, w, kas = decode(x, P)
    assert (lots[ib], lots[isa]) == (7, 6)
    assert kas == pytest.approx(0.335, abs=1e-3)


def test_lebar_kromosom_salah_ditolak():
    P = buat_problem(_w("kasus_tetap"), 5)
    with pytest.raises(ValueError, match="lebar"):
        decode_batch(np.zeros((2, P.n_genes - 1)), P)


def test_split_dan_init_population():
    P = buat_problem(_w("kasus_tetap"), 7)
    pop = init_population(P, np.random.default_rng(0), 40)
    assert pop.shape == (40, P.n_genes) and pop.min() >= 0 and pop.max() <= 1
    keys, raw, g0 = split(pop, P.n)
    assert keys.shape == raw.shape == (40, P.n) and g0.shape == (40,)
    assert g0.max() <= 0.5 + 1e-12                       # g0 awal di [0; 0,5]
    seeded = init_population(P, np.random.default_rng(0), 40, n_seed=10)
    assert (seeded[:10, -1] <= 0.1 + 1e-12).all()        # individu semai: g0 rendah


def test_bobot_target_konsisten_dengan_dekoder():
    P = buat_problem(_w("kasus_tetap"), 7)
    X = _acak(P, np.random.default_rng(1), 50)
    pilih, w, g0 = bobot_target(X, P)
    assert (pilih.sum(1) <= P.K).all() and (w <= P.cap + 1e-9).all()
    assert (w.sum(1) <= 1 - g0 + 1e-9).all()


def test_semua_data_tersedia():
    for nama in NAMA_DATA:
        d = _w(nama)
        assert len(d["tickers"]) == 42 and d["return_uji"].shape[1] == 42
