"""Evaluasi satu portofolio: rincian kas, penguraian return/risiko, dan uji pada data uji."""
import numpy as np

import config

from decoder import bobot_target, decode_batch


def rp(x):
    """Format Rupiah Indonesia, mis. Rp 1.676.000."""
    return "Rp " + f"{x:,.0f}".replace(",", ".")


def rincian_kas(x, P):
    """Rincian dana (Rp) satu kromosom: dana terpakai, sisa kas akhir, dan tiga komponennya."""
    x = np.asarray(x, float)
    pilih, w, g0 = bobot_target(x[None, :], P)
    lots, _, _ = decode_batch(x[None, :], P)
    terpakai = float((lots[0] * P.harga_lot).sum())
    kas_akhir = P.D - terpakai
    kas_sengaja = float(g0[0] * P.D)
    kas_cap = max(0.0, float(P.D * (1.0 - g0[0] - w[0].sum())))
    kas_bulat = kas_akhir - kas_sengaja - kas_cap                 # sisanya = efek pembulatan lot
    assert kas_bulat >= -1e-6, "sisa pembulatan tidak boleh negatif"
    return dict(modal=P.D, terpakai=terpakai, kas_akhir=kas_akhir, kas_sengaja=kas_sengaja,
                kas_cap=kas_cap, kas_pembulatan=max(kas_bulat, 0.0))


def analisis(P, lots):
    """Penguraian return dan risiko satu portofolio (pecahan dari modal)."""
    nilai = lots * P.harga_lot
    w = nilai / P.D
    kas = 1.0 - w.sum()
    sigma = float(np.sqrt(w @ P.cov @ w))
    return dict(w=w, kas=kas, sigma=sigma,
                R_saham=w * P.mu,                              # kontribusi return tiap saham
                R_kas=kas * P.rf,                              # kontribusi kas (bernilai r_f)
                R=float((w * P.mu).sum() + kas * P.rf),
                kontr_risiko=w * (P.cov @ w) / sigma ** 2)     # kontribusi tiap saham terhadap varians


def nilai_kas(D, rf, T):
    """Nilai harian (Rp) bila seluruh modal disimpan sebagai kas yang tumbuh r_f."""
    return D * (1.0 + rf) ** (np.arange(1, T + 1) / 252)


def nilai_1n(D, r_uji):
    """Nilai harian (Rp) portofolio sama rata atas seluruh saham, beli-dan-tahan."""
    return (D / r_uji.shape[1]) * np.cumprod(1.0 + r_uji, axis=0).sum(1)


def nilai_uji(P, lots, r_uji):
    """Nilai portofolio harian (Rp): tiap saham tumbuh sesuai return hariannya, kas tumbuh r_f."""
    pos = lots * P.harga_lot
    saham = np.cumprod(1.0 + r_uji, axis=0) @ pos
    return saham + (P.D - pos.sum()) * (1.0 + P.rf) ** (np.arange(1, len(r_uji) + 1) / 252)


def metrik_uji(nilai, D, rf=config.RISK_FREE):
    """Metrik data uji. sharpe = (return setara tahun - rf) / volatilitas; cvar95 = rata-rata 5% return harian terburuk."""
    T = len(nilai)
    awal = np.concatenate([[D], nilai])
    harian = np.diff(awal) / awal[:-1]
    total = nilai[-1] / D - 1
    puncak = np.maximum.accumulate(awal)[1:]
    vol = harian.std(ddof=1) * np.sqrt(252)
    setara = total * 252 / T
    n_ekor = max(1, int(np.ceil(0.05 * len(harian))))
    return dict(total=total, setara_tahun=setara, vol=vol, mdd=float((nilai / puncak - 1).min()),
                sharpe=float((setara - rf) / vol) if vol > 1e-9 else float("nan"),
                cvar95=float(np.sort(harian)[:n_ekor].mean()))


def komponen_akhir(P, lots, r_uji):
    """Nilai akhir (Rp) yang berada di saham dan di kas pada akhir periode uji."""
    pos = lots * P.harga_lot
    T = len(r_uji)
    return (float(np.cumprod(1.0 + r_uji, axis=0)[-1] @ pos),
            float((P.D - pos.sum()) * (1.0 + P.rf) ** (T / 252)))
