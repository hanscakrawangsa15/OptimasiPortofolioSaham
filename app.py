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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config
import runner
import stats
import visualisasi as viz
from algorithms import de as de_algo
from algorithms.brute_force_k2 import brute_force_k2
from data import NAMA_DATA, buat_problem, muat_jendela
from parameter import muat_beku
from evaluasi import (analisis, komponen_akhir, metrik_uji, nilai_1n, nilai_kas,
                      nilai_uji, rincian_kas, rp)

st.set_page_config(page_title="Optimasi Portofolio LQ45", layout="wide")

f2_berkas = os.path.join(config.HASIL_DIR, "csv", "fase2_hasil.csv")


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


@st.cache_data(show_spinner=False)
def hitung_profil(nama_data, K, D, metode, n_seed):
    """Jalankan tiap metode pada tiap profil risiko; mengembalikan {profil: {metode: [Result]}}."""
    w = muat_jendela(nama_data)
    return {p: {m: runner.jalankan(m, buat_problem(w, K, D=D, mult=mult), range(n_seed)) for m in metode}
            for p, mult in runner.PROFIL_RISIKO.items()}


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Pengaturan")
nama_data = st.sidebar.selectbox("Data (jendela walk-forward)", NAMA_DATA)
K = st.sidebar.selectbox("Maksimum jumlah saham (K)", config.K_LIST, index=3)
D = st.sidebar.number_input("Modal (Rp)", min_value=1_000_000.0, max_value=1_000_000_000.0,
                            value=float(config.D_UTAMA), step=1_000_000.0)
mult = st.sidebar.selectbox("Batas risiko (× volatilitas 1/N)", config.SIGMA_MULT, index=0)
algo = st.sidebar.multiselect("Algoritma", ["GA", "PSO", "SA", "DE"], default=["GA", "PSO", "SA", "DE"])
mode_param = st.sidebar.radio("Parameter semua algoritma", ["terpilih (hasil penalaan Fase 1)", "awal (PDF)"])
n_seed = st.sidebar.slider("Jumlah seed", 1, 30, 10)
jalankan = st.sidebar.button("Jalankan", type="primary")

st.title("Optimasi Alokasi Dana Saham LQ45")
st.caption("Maksimasi return ekspektasi dengan batas risiko σ_max, maksimum K saham, dan lot 100 lembar.")

metode = tuple(runner.nama_metode(a, mode_param.startswith("terpilih")) for a in algo)
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

tabs = st.tabs(["Ringkasan", "Portofolio", "Proses", "Perbandingan", "Data uji", "Profil risiko", "Hasil penelitian", "Fase 2", "Tentang"])

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

# ------------------------------------------------------------------ Profil risiko
with tabs[5]:
    st.subheader("Konvergensi menurut profil risiko")
    st.markdown("Profil risiko menentukan batas risiko σ_max sebagai kelipatan volatilitas portofolio 1/N: "
                + ", ".join(f"**{n}** = {m}×" for n, m in runner.PROFIL_RISIKO.items())
                + ". Data, K, modal, algoritma, dan jumlah seed mengikuti pengaturan di sisi kiri.")
    n_run = len(runner.PROFIL_RISIKO) * len(metode) * n_seed
    st.caption(f"Menjalankan {n_run} run ({len(runner.PROFIL_RISIKO)} profil × {len(metode)} algoritma × {n_seed} seed); "
               "SA sekitar 4 detik per run.")
    if st.button("Hitung analisis profil risiko"):
        st.session_state["profil"] = (nama_data, K, D, metode, n_seed)
    if st.session_state.get("profil") != (nama_data, K, D, metode, n_seed):
        st.info("Klik tombol di atas untuk menghitung (hasil untuk pengaturan ini belum ada).")
    else:
        with st.spinner("Menjalankan semua profil risiko..."):
            profil_runs = hitung_profil(nama_data, K, D, metode, n_seed)
        kurva = {p: {m: runner.konvergensi(rs) for m, rs in per.items()} for p, per in profil_runs.items()}
        smax = {p: w["sigma_max"][runner.PROFIL_RISIKO[p]] for p in profil_runs}
        st.pyplot(viz.plot_konvergensi_profil(kurva, smax))
        baris = []
        for p, per in profil_runs.items():
            for m, rs in per.items():
                k, ring = kurva[p][m], runner.ringkas(rs)
                baris.append({"Profil": p, "Algoritma": m, "σ_max": smax[p], "Median R akhir": k["R_akhir"],
                              "Setara Rp/tahun": rp(k["R_akhir"] * D) if not np.isnan(k["R_akhir"]) else "-",
                              "Pangsa layak": ring["layak"], "Evaluasi layak pertama": k["layak_pertama"],
                              "Evaluasi capai 90%": k["capai_90"], "Evaluasi capai 99%": k["capai_99"]})
        tab = pd.DataFrame(baris)
        st.dataframe(tab.style.format({"σ_max": "{:.4f}", "Median R akhir": "{:.4f}", "Pangsa layak": "{:.0%}"}),
                     hide_index=True)
        st.caption("Capai X% = evaluasi pertama saat median R terbaik mencapai X% dari total perbaikannya "
                   "(dihitung dari titik layak pertama sampai akhir anggaran). Makin kecil, makin cepat konvergen.")
        beda = {p: {m: kurva[p][m]["R_akhir"] for m in metode} for p in profil_runs}
        st.markdown("**Pengaruh profil terhadap return akhir (median):** "
                    + "; ".join(f"{m}: " + " → ".join(f"{p} {beda[p][m]:.4f}" for p in profil_runs) for m in metode))

# ------------------------------------------------------------------ Hasil penelitian
with tabs[6]:
    st.subheader("Fase 1: penalaan parameter keempat algoritma")
    beku_semua = muat_beku()
    if not beku_semua:
        st.info("Parameter beku belum ada (jalankan python3 src/penalaan.py).")
    else:
        n_run = next(iter(beku_semua.values()))["run_per_titik"]
        st.write(f"Protokol sama untuk GA, PSO, SA, dan DE: data latih {config.TUNING_DATA}, K = {config.TUNING_K}, "
                 f"σ_max = {config.TUNING_MULT}×, {n_run} run per titik grid (seed {config.TUNING_SEEDS[0]}–{config.TUNING_SEEDS[-1]}), "
                 f"{config.BUDGET_EVAL:,} evaluasi per run, semai top-K {config.SEED_FRAKSI:.0%} populasi (SA: 1 individu).")
        baris = []
        for a, b in beku_semua.items():
            baris.append({"Algoritma": a, "Parameter terpilih": ", ".join(f"{k} = {v}" for k, v in b["grid_parameter"].items()),
                          "Parameter awal (PDF)": ", ".join(f"{k} = {v}" for k, v in b["parameter_awal"].items()),
                          "Median R terpilih": b["median_R_penalaan"], "Median R awal": b["median_R_awal"],
                          "p (Mann–Whitney)": b["p_mannwhitney_vs_awal"], "Titik grid": b["titik_diuji"]})
        st.dataframe(pd.DataFrame(baris).style.format({"Median R terpilih": "{:.4f}", "Median R awal": "{:.4f}", "p (Mann–Whitney)": "{:.3g}"}),
                     hide_index=True)
        st.caption("p < 0,05 berarti parameter terpilih berbeda nyata dari parameter awal pada seed penalaan; "
                   "bila tidak, penalaan tidak memberi perbaikan yang dapat dibuktikan.")
        pilih_a = st.selectbox("Rincian grid untuk", list(beku_semua), key="fase1_algo")
        f_g = os.path.join(config.HASIL_DIR, "csv", f"fase1_{pilih_a.lower()}_grid.csv")
        if os.path.exists(f_g):
            g_ = pd.read_csv(f_g)
            kol_p = list(beku_semua[pilih_a]["parameter_awal"])
            kol_p = [("c" if k == "c1" else k) for k in kol_p if k != "c2"]
            r_ = g_.assign(R_eff=g_["R"].where(g_["layak"], -1.0)).groupby(kol_p)["R_eff"].agg(["median", "std", "count"]).reset_index()
            r_.columns = kol_p + ["Median R", "Std R", "Jumlah run"]
            st.dataframe(r_.sort_values("Median R", ascending=False).head(15), hide_index=True)

    st.subheader("Penalaan PSO lama (notebook PSO Fase 1, tanpa semai)")
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

    st.subheader("Hasil Differential Evolution (notebook DE)")
    csv_de = {n: os.path.join(config.HASIL_DIR, "csv", f"de_{n}.csv")
              for n in ("kasus_tetap_30seed", "ringkasan_alokasi", "rincian_saham_lot", "rekap_walkforward")}
    if not all(os.path.exists(f) for f in csv_de.values()):
        st.info("Berkas hasil DE belum ada di hasil/csv/ (jalankan notebooks/DE_Portofolio_LQ45.ipynb).")
    else:
        d_tetap = pd.read_csv(csv_de["kasus_tetap_30seed"])
        d_alok = pd.read_csv(csv_de["ringkasan_alokasi"])
        d_rinci = pd.read_csv(csv_de["rincian_saham_lot"])
        d_wf = pd.read_csv(csv_de["rekap_walkforward"])
        prm = de_algo.DE_PARAM
        st.write(f"DE/rand/1/bin: populasi {prm['pop']}, F = {prm['F']}, CR = {prm['CR']}, anggaran {config.BUDGET_EVAL:,} evaluasi. "
                 "Parameter ini nilai tetap dari notebook (tidak ditala seperti PSO Fase 1).")

        st.markdown("**Kasus tetap: 30 seed per K**")
        per_k = d_tetap.groupby("K").agg(**{
            "Median R latih": ("return_latih", "median"), "Std R latih": ("return_latih", "std"),
            "Run layak": ("valid_latih", "mean"), "Median kas": ("kas", "median"),
            "Median detik/run": ("waktu_detik", "median"), "Median return uji (setara/thn)": ("return_uji", "median")})
        st.dataframe(per_k.style.format({"Median R latih": "{:.4f}", "Std R latih": "{:.4f}", "Run layak": "{:.0%}",
                                         "Median kas": "{:.1%}", "Median detik/run": "{:.2f}",
                                         "Median return uji (setara/thn)": "{:.2%}"}))
        bf = brute_force_k2(buat_problem(muat_jendela("kasus_tetap"), 2))["kas bebas"][0]
        r2 = d_tetap.loc[d_tetap.K == 2, "return_latih"].max()
        st.caption(f"Acuan K = 2: optimum brute force R* = {bf:.4f}; R terbaik DE (30 seed) = {r2:.4f}, selisih {100 * (bf - r2):.3f} poin persentase.")

        st.markdown("**Alokasi terbaik per K (dipilih dari return latih)**")
        kolom_alok = ["K Maksimal", "Seed Terbaik", "Saham Dibeli (Emiten)", "Dana Saham (Rp)", "Sisa Kas (Rp)", "Kas (% Modal)",
                      "Expected Return Latih (%/Tahun)", "Risiko Latih (%/Tahun)", "Batas Risiko (%/Tahun)", "Status"]
        st.dataframe(d_alok[kolom_alok].style.format({
            "Dana Saham (Rp)": lambda v: rp(v), "Sisa Kas (Rp)": lambda v: rp(v), "Kas (% Modal)": "{:.1%}",
            "Expected Return Latih (%/Tahun)": "{:.2%}", "Risiko Latih (%/Tahun)": "{:.2%}", "Batas Risiko (%/Tahun)": "{:.2%}"}),
            hide_index=True)
        pilih_k = st.selectbox("Rincian saham dan lot untuk K =", sorted(d_rinci["K Maksimal"].unique()), key="de_k")
        rinci = d_rinci[d_rinci["K Maksimal"] == pilih_k]
        st.dataframe(rinci.drop(columns=["K Maksimal"]).style.format({
            "Harga per Lot (Rp)": lambda v: rp(v), "Biaya Beli (Rp)": lambda v: rp(v), "Bobot (% Modal)": "{:.1%}",
            "Estimasi Return Saham (%/Tahun)": "{:.2%}", "Kontribusi Return (Poin Persentase/Tahun)": "{:.2f}"}), hide_index=True)

        st.markdown("**Walk-forward: jendela 1–3, 10 seed per K**")
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
        for nama_j, g in d_wf.groupby("jendela"):
            axs[0].plot(g["K"], g["median_return_latih"], marker="o", label=nama_j)
            axs[1].plot(g["K"], g["median_return_uji"], marker="o", label=nama_j)
        axs[0].set(title="Median R latih", xlabel="K", ylabel="return"); axs[1].set(title="Median return uji (setara/thn)", xlabel="K")
        axs[1].axhline(0, color="gray", ls="--", lw=1)
        for a in axs:
            a.grid(alpha=0.3)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        wf = d_wf[["jendela", "K", "median_return_latih", "median_return_uji", "median_vol_uji", "proporsi_memenuhi_risiko_uji", "median_kas"]]
        wf.columns = ["Jendela", "K", "Median R latih", "Median return uji", "Median volatilitas uji", "Kepatuhan risiko uji", "Median kas"]
        st.dataframe(wf.style.format({"Median R latih": "{:.4f}", "Median return uji": "{:.2%}", "Median volatilitas uji": "{:.2%}",
                                      "Kepatuhan risiko uji": "{:.0%}", "Median kas": "{:.1%}"}), hide_index=True)
        patuh = float(d_wf["proporsi_memenuhi_risiko_uji"].mean())
        st.caption(f"Semua run layak pada data latih, tetapi hanya {patuh:.0%} (rata-rata per sel) yang volatilitasnya masih di bawah σ_max "
                   "pada data uji: batas risiko dipenuhi saat optimasi, bukan jaminan pada periode berikutnya.")

# ------------------------------------------------------------------ Fase 2
with tabs[7]:
    st.subheader("Perbandingan keempat algoritma (Fase 2)")
    if not os.path.exists(f2_berkas):
        st.info("Hasil Fase 2 belum ada (jalankan python3 src/eksperimen.py).")
    else:
        f2 = pd.read_csv(f2_berkas)
        n_masalah = f2.groupby(["data", "K"]).ngroups
        st.write(f"{len(f2):,} run: {f2['algoritma'].nunique()} algoritma × {n_masalah} masalah × {f2['seed'].nunique()} seed, "
                 f"{int(f2['n_eval'].iloc[0]):,} evaluasi per run, seed sama untuk semua algoritma, parameter beku Fase 1.")
        med = stats.median_per_masalah(f2)
        st.markdown("**Peringkat rata-rata atas semua masalah** (1 = terbaik, berdasarkan median R antar seed)")
        st.dataframe(stats.peringkat_rata(med).rename("Peringkat rata-rata").to_frame().style.format("{:.2f}"))
        st.markdown("**Wilcoxon berpasangan atas masalah, koreksi Holm**")
        w_ = stats.wilcoxon_holm(med)
        st.dataframe(w_.style.format({"median_selisih": "{:+.5f}", "p": "{:.4f}", "p_holm": "{:.4f}"}), hide_index=True)
        st.markdown("**Median R per K** (rata-rata atas data)")
        st.dataframe(f2.assign(Reff=f2["R"].where(f2["layak"], -1.0)).groupby(["K", "algoritma"])["Reff"].median()
                     .unstack("algoritma").style.format("{:.4f}"))
        st.markdown("**Out-of-sample: median return uji setara tahunan dan kepatuhan risiko uji**")
        oos = f2.groupby("algoritma").agg(**{"Median return uji": ("return_uji_tahunan", "median"),
                                             "Median volatilitas uji": ("vol_uji", "median"),
                                             "Kepatuhan risiko uji": ("risiko_uji_memenuhi", "mean"),
                                             "Median detik/run": ("detik", "median")})
        st.dataframe(oos.style.format({"Median return uji": "{:.2%}", "Median volatilitas uji": "{:.2%}",
                                       "Kepatuhan risiko uji": "{:.0%}", "Median detik/run": "{:.2f}"}))
        c1_, c2_ = st.columns(2)
        pil_d = c1_.selectbox("Data", sorted(f2["data"].unique()), key="f2_data")
        pil_k = c2_.selectbox("K", sorted(f2["K"].unique()), index=3, key="f2_k")
        sub_ = f2[(f2["data"] == pil_d) & (f2["K"] == pil_k) & f2["layak"]]
        hasil_f2 = {a: [type("R", (), {"R": v, "feasible": True})() for v in g["R"]] for a, g in sub_.groupby("algoritma")}
        if hasil_f2:
            st.pyplot(viz.plot_box(hasil_f2))

# ------------------------------------------------------------------ Tentang
with tabs[8]:
    st.markdown(f"""
**Masalah.** Maksimalkan R = Σ wᵢμᵢ + kas·r_f dengan σ = √(wᵀΣw) ≤ σ_max, paling banyak K saham,
bobot ≤ {config.MAX_WEIGHT:.0%} per saham bila K ≥ {config.CAP_BERLAKU_MULAI_K}, pembelian dalam lot {config.LOT_SIZE} lembar
(harga raw), kas bernilai r_f = {config.RISK_FREE:.0%}. μ dan Σ diestimasi dari data latih (return sederhana, ×252).

**Kromosom 2N+1.** N kunci prioritas (memilih K saham), N bobot mentah, dan 1 gen kas; dekoder deterministik
menghasilkan lot bulat. Hanya batas risiko yang bisa dilanggar → aturan kelayakan Deb.

**Anggaran.** Tiap run {config.BUDGET_EVAL:,} evaluasi (GA: populasi 100; PSO: swarm 40; DE: populasi 100, mutasi rand/1, crossover binomial; SA: satu solusi dengan kalibrasi temperatur awal, jadwal pendinginan geometrik, dan penerimaan Metropolis di bawah aturan Deb). SA lebih lambat per run (sekitar 4 detik), sedangkan DE sekitar 0,4 detik, jadi gunakan jumlah seed kecil untuk percobaan cepat.
PSO "terpilih" memakai w = 1,0, c₁ = c₂ = 1,0, v_maks = 1,0 hasil penalaan Fase 1 pada jendela_1; "awal" memakai w = 0,7, c = 1,5, v_maks = 0,2 dari rencana.
""")
