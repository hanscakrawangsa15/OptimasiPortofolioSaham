"""Tes GUI Streamlit dengan AppTest (tanpa browser)."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _app():
    return AppTest.from_file(APP, default_timeout=120).run()


def test_awal_menampilkan_petunjuk():
    at = _app()
    assert not at.exception and any("Jalankan" in i.value for i in at.info)
    assert len(at.tabs) == 0


def test_jalankan_dua_algoritma():
    at = _app()
    at.sidebar.selectbox[1].select(2)                    # K = 2 -> ada acuan brute force
    at.sidebar.slider[0].set_value(3)
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception
    assert len(at.tabs) == 7
    assert len(at.metric) >= 10                          # 5 metrik x 2 algoritma
    assert any("Mann" in m.value for m in at.markdown)
