"""Optimum sejati untuk K <= 2 dengan enumerasi penuh (acuan untuk menilai GA dan PSO)."""
import numpy as np


def brute_force_k2(P):
    # Optimum sejati untuk K <= 2: semua saham tunggal dan pasangan, semua kombinasi lot (n_i >= 1).
    # Mengembalikan dict {aturan_kas: (R, {saham: lot}, kas)}.
    mu, cov, c, D, rf = P.mu, P.cov, P.harga_lot, P.D, P.rf
    N = P.n
    terbaik = {"kas bebas": (-9.0, None, None), "kas <= 5%": (-9.0, None, None),
               "kas hanya dari pembulatan": (-9.0, None, None)}

    def catat(nama, R, lot, kas):
        if R > terbaik[nama][0]:
            terbaik[nama] = (float(R), lot, float(kas))

    def aturan(layak, kas, biaya_min):
        return {"kas bebas": layak, "kas <= 5%": layak & (kas <= 0.05),
                "kas hanya dari pembulatan": layak & (kas * D < biaya_min)}

    for i in range(N):
        n1 = np.arange(1, int(D // c[i]) + 1)
        w1 = n1 * c[i] / D
        kas = 1 - w1
        sigma = w1 * np.sqrt(cov[i, i])
        R = w1 * mu[i] + kas * rf
        for nama, m in aturan(sigma <= P.sigma_max, kas, c[i]).items():
            if m.any():
                k = np.argmax(np.where(m, R, -9)); catat(nama, R[k], {P.tickers[i]: int(n1[k])}, kas[k])
        for j in range(i + 1, N):
            a = np.arange(1, int(D // c[i]) + 1)[:, None]
            b = np.arange(1, int(D // c[j]) + 1)[None, :]
            ok = a * c[i] + b * c[j] <= D
            w1, w2 = a * c[i] / D, b * c[j] / D
            kas = 1 - w1 - w2
            var = w1 ** 2 * cov[i, i] + w2 ** 2 * cov[j, j] + 2 * w1 * w2 * cov[i, j]
            R = w1 * mu[i] + w2 * mu[j] + kas * rf
            layak = ok & (np.sqrt(var) <= P.sigma_max)
            for nama, m in aturan(layak, kas, c[i] + c[j]).items():
                if m.any():
                    idx = np.unravel_index(np.argmax(np.where(m, R, -9)), R.shape)
                    catat(nama, R[idx], {P.tickers[i]: int(a[idx[0], 0]), P.tickers[j]: int(b[0, idx[1]])}, kas[idx])
    return terbaik
