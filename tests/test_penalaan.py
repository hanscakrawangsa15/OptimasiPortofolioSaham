"""Tes Fase 1: protokol penalaan sama untuk semua algoritma."""
import numpy as np
import pandas as pd
import pytest

import config
import penalaan
from parameter import kw_terpilih, muat_beku


def test_seed_penalaan_terpisah_dari_seed_fase2():
    assert len(config.TUNING_SEEDS) == 30 and len(config.SEEDS_FASE2) == 30
    assert not set(config.TUNING_SEEDS) & set(config.SEEDS_FASE2)
    assert config.TUNING_DATA == "jendela_1" and config.TUNING_K == 7 and config.TUNING_MULT == 1.5


@pytest.mark.parametrize("algo", list(penalaan.RUANG))
def test_parameter_awal_pdf_ada_dalam_grid(algo):
    """Grid harus memuat parameter awal PDF agar penalaan bisa dibandingkan dengan parameter awal."""
    nilai = {}
    for _, grid in penalaan.RUANG[algo]:
        for k, v in grid.items():
            nilai.setdefault(k, set()).update(v)
    for k, v in penalaan.AWAL[algo].items():
        assert v in nilai.get(k, set()), (algo, k, v)


def test_ke_kw_pso_c_menjadi_c1_c2():
    assert penalaan.ke_kw("PSO", dict(w=0.7, c=1.5, v_maks=0.2)) == dict(w=0.7, v_maks=0.2, c1=1.5, c2=1.5)
    assert penalaan.ke_kw("GA", dict(pop=50, laju_mut=2.0)) == dict(pop=50, laju_mut=2.0)


def test_aturan_pilih_median_lalu_std_untuk_seri():
    t = pd.DataFrame(dict(median=[0.3000, 0.3005, 0.2900, 0.3004], std=[0.0100, 0.0050, 0.0001, 0.0020]))
    assert penalaan.pilih(t).name == 3            # 0.3005 dan 0.3004 seri (<= 0,001): std 0,0020 < 0,0050
    t2 = pd.DataFrame(dict(median=[0.30, 0.32], std=[0.001, 0.05]))
    assert penalaan.pilih(t2).name == 1           # selisih > toleransi: median tertinggi menang walau std besar


def test_tala_kecil_menghasilkan_parameter_beku():
    ruang = [("A", {"pop": [50, 100], "laju_mut": [1.0, 2.0]})]
    df, beku = penalaan.tala("GA", seeds=(100, 101, 102), budget=2000, ruang=ruang, proses=1, cetak=lambda *_: None)
    assert len(df[df.tahap == "A"]) == 4 * 3 and set(df["seed"]) == {100, 101, 102}
    titik = df[["pop", "laju_mut"]].drop_duplicates()
    assert beku["titik_diuji"] == len(titik) >= 4 and len(df) == len(titik) * 3     # tahap A2 muncul bila terbaik di tepi
    assert beku["parameter"] == beku["grid_parameter"]
    assert beku["data_penalaan"] == "jendela_1" and beku["K"] == 7 and beku["run_per_titik"] == 3
    med = df.groupby(["pop", "laju_mut"])["R"].median()
    assert beku["median_R_penalaan"] == pytest.approx(med[(beku["grid_parameter"]["pop"], beku["grid_parameter"]["laju_mut"])])
    assert isinstance(beku["di_tepi_grid"], list)


def test_tala_tahap_b_memakai_terbaik_tahap_a():
    ruang = [("A", {"w": [0.7, 1.0], "c": [1.5]}), ("B", {"v_maks": [0.2, 1.0]})]
    df, beku = penalaan.tala("PSO", seeds=(100, 101), budget=2000, ruang=ruang, proses=1, cetak=lambda *_: None)
    b = df[df.tahap == "B"]
    assert b["w"].nunique() == 1 and b["c"].nunique() == 1          # tahap B hanya memvariasikan v_maks
    assert set(beku["parameter"]) == {"w", "c1", "c2", "v_maks"}


def test_kw_terpilih_kosong_bila_berkas_tidak_ada(tmp_path):
    assert kw_terpilih("GA", path=tmp_path / "tidak_ada.json") == {} and muat_beku(tmp_path / "x.json") == {}


def test_perluasan_grid_hanya_di_tepi_dan_sesuai_batas_alami():
    g = {"F": [0.3, 0.5, 0.7, 0.9], "CR": [0.5, 0.7, 0.9, 1.0]}
    besar = penalaan.perluas("DE", g, {"F": 0.3, "CR": 0.5})
    assert besar["F"][:2] == [0.1, 0.2] and besar["CR"][:2] == [0.1, 0.3]        # keduanya di tepi bawah
    assert penalaan.perluas("DE", g, {"F": 0.5, "CR": 0.7}) == {}                  # di dalam grid: tidak diperluas
    assert max(penalaan.perluas("DE", g, {"F": 0.9, "CR": 1.0})["F"]) == 1.3      # tepi atas F
    assert max(penalaan.perluas("DE", g, {"F": 0.9, "CR": 1.0})["CR"]) == 1.0           # CR tidak melebihi 1
    pso_g = {"w": [0.7, 1.0], "c": [1.0, 2.0]}
    assert max(penalaan.perluas("PSO", pso_g, {"w": 1.0, "c": 1.0})["w"]) == 1.0   # w batas alami 1,0
