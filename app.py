"""GUI Streamlit: hasil GA dan PSO untuk optimasi portofolio LQ45.

Jalankan:  streamlit run app.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import streamlit as st

import config
import runner
import visualisasi as viz
from algorithms.brute_force_k2 import brute_force_k2
from data import NAMA_DATA, buat_problem, muat_jendela
from evaluasi import (analisis, komponen_akhir, metrik_uji, nilai_1n, nilai_kas,
                      nilai_uji, rincian_kas, rp)

st.set_page_config(page_title="Optimasi Portofolio LQ45", layout="wide")


@st.cache_data(show_spinner=False)
def hitung_semua(nama_data, K, D, mult, metode, n_seed):
    """Jalankan tiap metode pada n_seed seed; ulangi run terbaik dengan trace (hasil identik)."""
    P = buat_problem(muat_jendela(nama_data), K, D=D, mult=mult)
    hasil, terbaik, trace = {}, {}, {}
    for m in metode:
        runs = runner.jalankan(m, P, range(n_seed))
        b = runner.terbaik(runs)
        hasil[m], terbaik[m] = runs, b
        trace[m] = runner.jalankan(m, P, [b.seed], trace=True)[0].trace
    R_bf = brute_force_k2(P)["kas bebas"][0] if K == 2 else None
    return hasil, terbaik, trace, R_bf


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Pengaturan")
nama_data = st.sidebar.selectbox("Data (jendela walk-forward)", NAMA_DATA)
K = st.sidebar.selectbox("Maksimum jumlah saham (K)", config.K_LIST, index=3)
D = st.sidebar.number_input("Modal (Rp)", min_value=1_000_000.0, max_value=1_000_000_000.0,
                            value=float(config.D_UTAMA), step=1_000_000.0)
mult = st.sidebar.selectbox("Batas risiko (× volatilitas 1/N)", config.SIGMA_MULT, index=0)
algo = st.sidebar.multiselect("Algoritma", ["GA", "PSO", "SA"], default=["GA", "PSO", "SA"])
mode_pso = st.sidebar.radio("Parameter PSO", ["terpilih (hasil penalaan Fase 1)", "awal (PDF)"])
n_seed = st.sidebar.slider("Jumlah seed", 1, 30, 10)
jalankan = st.sidebar.button("Jalankan", type="primary")

st.title("Optimasi Alokasi Dana Saham LQ45")
st.caption("Maksimasi return ekspektasi dengan batas risiko σ_max, maksimum K saham, dan lot 100 lembar.")

def _nama_metode(a):
    if a == "PSO":
        return "PSO terpilih" if mode_pso.startswith("terpilih") else "PSO awal"
    return a


metode = tuple(_nama_metode(a) for a in algo)
if jalankan:
    st.session_state["param"] = (nama_data, K, D, mult, metode, n_seed)
param = st.session_state.get("param")
if param is None or not param[4]:
    st.info("Atur parameter di sisi kiri, pilih minimal satu algoritma, lalu klik **Jalankan**.")
    st.stop()

nama_data, K, D, mult, metode, n_seed = param
with st.spinner("Menjalankan algoritma..."):
    hasil, terbaik, trace, R_bf = hitung_semua(*param)
w = muat_jendela(nama_data)
P = buat_problem(w, K, D=D, mult=mult)
r_uji = w["return_uji"]
st.write(f"**{nama_data}** · latih {w['tanggal_latih_awal']} s.d. {w['tanggal_latih_akhir']} · "
         f"uji {w['tanggal_uji_awal']} s.d. {w['tanggal_uji_akhir']} · K = {K} · σ_max = {P.sigma_max:.4f}")

tabs = st.tabs(["Ringkasan", "Portofolio", "Proses", "GA vs PSO", "Data uji", "Hasil penelitian", "Tentang"])

# ------------------------------------------------------------------ Ringkasan
with tabs[0]:
    for m in metode:
        b = terbaik[m]
        A = analisis(P, b.lots)
        st.subheader(f"{m} (seed terbaik {b.seed})")
        c = st.columns(5)
        c[0].metric("Return ekspektasi / tahun", f"{b.R:.2%}")
        c[1].metric("Setara Rp / tahun", rp(b.R * P.D))
        c[2].metric("Nilai akhir (1 tahun)", rp(P.D * (1 + b.R)))
        c[3].metric("Sisa kas", rp(P.D - (b.lots * P.harga_lot).sum()), f"{A['kas']:.1%} dari modal")
        c[4].metric("Risiko σ", f"{b.sigma:.4f}", f"batas {P.sigma_max:.4f}", delta_color="off")
        st.caption(f"{'Layak' if b.feasible else 'MELANGGAR batas risiko'} · {b.n_eval:,} evaluasi · "
                   f"{b.seconds:.2f} detik · solusi ditemukan pada evaluasi ke-{b.n_eval_best:,}")

# ------------------------------------------------------------------ Portofolio
with tabs[1]:
    kolom = st.columns(len(metode))
    for col, m in zip(kolom, metode):
        b = terbaik[m]
        with col:
            st.subheader(m)
            idx = np.where(b.lots > 0)[0]
            idx = idx[np.argsort(-b.lots[idx] * P.harga_lot[idx])]
            st.dataframe(pd.DataFrame({
                "Saham": [P.tickers[i] for i in idx], "Lot": b.lots[idx], "Lembar": b.lots[idx] * P.lot,
                "Harga/lembar": [rp(P.price[i]) for i in idx],
                "Nilai posisi": [rp(b.lots[i] * P.harga_lot[i]) for i in idx],
                "Bobot": [f"{b.lots[i] * P.harga_lot[i] / P.D:.1%}" for i in idx]}), hide_index=True)
            r = rincian_kas(b.x, P)
            st.table(pd.DataFrame({"Rupiah": {
                "Modal": rp(r["modal"]), "Dana terpakai": rp(r["terpakai"]), "Sisa kas akhir": rp(r["kas_akhir"]),
                "– kas sengaja (g0 × D)": rp(r["kas_sengaja"]), "– kelebihan cap / tak teralokasi": rp(r["kas_cap"]),
                "– sisa pembulatan lot": rp(r["kas_pembulatan"])}}))
            st.pyplot(viz.plot_bobot(P, b.lots, m))

# ------------------------------------------------------------------ Proses
with tabs[2]:
    st.pyplot(viz.plot_konvergensi(hasil, R_bf))
    st.pyplot(viz.plot_risiko_return(hasil, P.sigma_max))
    for m in metode:
        judul = f"{m}: proses pada seed terbaik ({terbaik[m].seed})"
        st.pyplot(viz.plot_proses_sa(trace[m], P, judul) if m == "SA" else viz.plot_proses(trace[m], P, judul))

# ------------------------------------------------------------------ GA vs PSO
with tabs[3]:
    ref = max(r.R for rs in hasil.values() for r in rs if r.feasible)
    R_acuan = R_bf if R_bf is not None else ref
    st.caption("Acuan sukses: " + (f"optimum sejati brute force (R* = {R_bf:.4f})" if R_bf is not None
                                  else f"R terbaik yang pernah ditemukan semua metode ({ref:.4f}); optimum sejati hanya diketahui untuk K = 2"))
    tab = pd.DataFrame({m: runner.ringkas(hasil[m], R_acuan) for m in metode}).T
    tab.columns = ["Median R", "Std R", "Min R", "Maks R", "Pangsa layak", "Pangsa sukses", "Detik/run", "Evaluasi ke-terbaik (median)"]
    st.dataframe(tab.style.format("{:.4f}"))
    st.pyplot(viz.plot_box(hasil, R_bf))
    if len(metode) >= 2:
        for i in range(len(metode)):
            for j in range(i + 1, len(metode)):
                a_, b_ = metode[i], metode[j]
                c = runner.bandingkan(hasil[a_], hasil[b_])
                kata = {"A": a_, "B": b_}.get(c["lebih_baik"], c["lebih_baik"])
                st.write(f"Mann–Whitney {a_} vs {b_}: p = {c['p']:.4f} → **{kata}**"
                         + (" lebih baik." if c["lebih_baik"] != "tidak berbeda nyata" else "."))
        if n_seed < 10:
            st.warning("Jumlah seed kecil; uji statistik kurang andal (disarankan ≥ 10, idealnya 30).")

# ------------------------------------------------------------------ Data uji
with tabs[4]:
    V = {m: nilai_uji(P, terbaik[m].lots, r_uji) for m in metode}
    V["1/N (semua saham)"] = nilai_1n(P.D, r_uji)
    V["kas saja"] = nilai_kas(P.D, P.rf, len(r_uji))
    st.pyplot(viz.plot_uji(V, P.D))
    baris = {}
    for nama, v in V.items():
        mt = metrik_uji(v, P.D)
        baris[nama] = {"Nilai akhir": rp(v[-1]), "Total return": f"{mt['total']:.2%}",
                       "Setara per tahun": f"{mt['setara_tahun']:.2%}", "Volatilitas": f"{mt['vol']:.2%}",
                       "Maks. drawdown": f"{mt['mdd']:.2%}"}
    for m in metode:
        s, k = komponen_akhir(P, terbaik[m].lots, r_uji)
        baris[m]["Di saham"], baris[m]["Di kas"] = rp(s), rp(k)
    st.dataframe(pd.DataFrame(baris).T)
    st.caption("Data uji tidak dipakai saat optimasi; hasilnya jauh bisa berbeda dari ekspektasi in-sample.")

# ------------------------------------------------------------------ Hasil penelitian
with tabs[5]:
    st.subheader("Penalaan parameter PSO (Fase 1)")
    f_grid = os.path.join(config.HASIL_DIR, "csv", "pso_fase1_grid.csv")
    f_beku = os.path.join(config.HASIL_DIR, "csv", "pso_parameter_beku.json")
    if not (os.path.exists(f_grid) and os.path.exists(f_beku)):
        st.info("Berkas hasil penalaan belum ada di hasil/csv/.")
    else:
        beku = json.load(open(f_beku, encoding="utf-8"))
        grid = pd.read_csv(f_grid)
        st.write(f"Data penalaan **{beku['data_penalaan']}**, K = {beku['K']}, swarm {beku['swarm']}, "
                 f"{beku['iterasi']} iterasi ({beku['anggaran']:,} evaluasi), seed {beku['seed_penalaan'][0]}–{beku['seed_penalaan'][1]}.")
        c = st.columns(5)
        c[0].metric("w", beku["w"]); c[1].metric("c₁ = c₂", beku["c1"]); c[2].metric("v_maks", beku["v_maks"])
        c[3].metric("Median R penalaan", f"{beku['median_R_penalaan']:.4f}")
        c[4].metric("Konfigurasi dicoba", grid[["tahap", "w", "c", "v_maks"]].drop_duplicates().shape[0])
        st.caption("Aturan pemilihan: " + beku["aturan"])
        ringkas_grid = (grid.groupby(["w", "c", "v_maks"])["R"].agg(["median", "std", "count"])
                        .sort_values("median", ascending=False).reset_index())
        ringkas_grid.columns = ["w", "c", "v_maks", "Median R", "Std R", "Jumlah run"]
        st.write("Konfigurasi terbaik (median R tertinggi):")
        st.dataframe(ringkas_grid.head(15), hide_index=True)
        with st.expander("Seluruh hasil per run"):
            st.dataframe(grid, hide_index=True)

    st.subheader("Hasil Simulated Annealing (30 seed)")
    f_sa = os.path.join(config.HASIL_DIR, "hasil_sa_30_seed.csv")
    if not os.path.exists(f_sa):
        st.info("Berkas hasil/hasil_sa_30_seed.csv belum ada.")
    else:
        sa_df = pd.read_csv(f_sa)
        st.write("Data kasus_tetap, K = 7, anggaran 20.000 evaluasi per run (dari notebooks/SA_Portofolio_LQ45.ipynb).")
        c = st.columns(4)
        c[0].metric("Median R", f"{sa_df['Return'].median():.4f}")
        c[1].metric("Maks R", f"{sa_df['Return'].max():.4f}")
        c[2].metric("Std R", f"{sa_df['Return'].std():.4f}")
        c[3].metric("Run layak", f"{int(sa_df['Feasible'].sum())} / {len(sa_df)}")
        with st.expander("Seluruh hasil per seed"):
            st.dataframe(sa_df, hide_index=True)

# ------------------------------------------------------------------ Tentang
with tabs[6]:
    st.markdown(f"""
**Masalah.** Maksimalkan R = Σ wᵢμᵢ + kas·r_f dengan σ = √(wᵀΣw) ≤ σ_max, paling banyak K saham,
bobot ≤ {config.MAX_WEIGHT:.0%} per saham bila K ≥ {config.CAP_BERLAKU_MULAI_K}, pembelian dalam lot {config.LOT_SIZE} lembar
(harga raw), kas bernilai r_f = {config.RISK_FREE:.0%}. μ dan Σ diestimasi dari data latih (return sederhana, ×252).

**Kromosom 2N+1.** N kunci prioritas (memilih K saham), N bobot mentah, dan 1 gen kas; dekoder deterministik
menghasilkan lot bulat. Hanya batas risiko yang bisa dilanggar → aturan kelayakan Deb.

**Anggaran.** Tiap run {config.BUDGET_EVAL:,} evaluasi (GA: populasi 100; PSO: swarm 40; SA: satu solusi dengan kalibrasi temperatur awal, jadwal pendinginan geometrik, dan penerimaan Metropolis di bawah aturan Deb). SA lebih lambat per run (sekitar 4 detik), jadi gunakan jumlah seed kecil untuk percobaan cepat.
PSO "terpilih" memakai w = 1,0, c₁ = c₂ = 1,0, v_maks = 1,0 hasil penalaan Fase 1 pada jendela_1; "awal" memakai w = 0,7, c = 1,5, v_maks = 0,2 dari rencana.
""")
