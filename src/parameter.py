"""Parameter beku hasil Fase 1 (hasil/csv/fase1_parameter_beku.json) untuk GA, PSO, SA, DE."""
import json
import os

import config

BERKAS_BEKU = os.path.join(config.HASIL_DIR, "csv", "fase1_parameter_beku.json")


def muat_beku(path=None):
    """Isi berkas parameter beku sebagai dict {algoritma: {...}}; kosong bila belum ada."""
    path = path or BERKAS_BEKU
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def kw_terpilih(algoritma, path=None):
    """Argumen keyword run() hasil penalaan; dict kosong (= parameter awal PDF) bila belum ada."""
    return dict(muat_beku(path).get(algoritma, {}).get("parameter", {}))
