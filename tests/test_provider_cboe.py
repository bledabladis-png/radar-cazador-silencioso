"""Tests del CboeProvider (F5.6-03). Sin red."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from data.providers.cboe import CboeProvider


def test_ratio_valor_valido():
    cp = CboeProvider()
    data = {"ratios": [{"name": "TOTAL PUT/CALL RATIO", "value": "0.79"}]}
    assert cp._ratio(data, "TOTAL PUT/CALL RATIO") == pytest.approx(0.79)


def test_ratio_valor_none_devuelve_none():
    cp = CboeProvider()
    data = {"ratios": [{"name": "TOTAL PUT/CALL RATIO", "value": None}]}
    assert cp._ratio(data, "TOTAL PUT/CALL RATIO") is None


def test_ratio_valor_no_numerico_devuelve_none():
    cp = CboeProvider()
    data = {"ratios": [{"name": "TOTAL PUT/CALL RATIO", "value": "N/D"}]}
    assert cp._ratio(data, "TOTAL PUT/CALL RATIO") is None


def test_ratio_ausente_devuelve_none():
    cp = CboeProvider()
    assert cp._ratio({"ratios": []}, "TOTAL PUT/CALL RATIO") is None


def test_extract_json_fallo_red_devuelve_none():
    cp = CboeProvider()
    import requests
    with patch("data.providers.cboe.requests.get",
               side_effect=requests.RequestException("timeout")):
        assert cp._extract_json() is None


def test_is_available_fallo_red_devuelve_false():
    cp = CboeProvider()
    import requests
    with patch("data.providers.cboe.requests.get",
               side_effect=requests.RequestException("nope")):
        assert cp.is_available() is False


def test_spx_spxo_nuevo_esquema_y_cache_is_available():
    cp = CboeProvider()
    fake_payload = {
        "selectedDate": "2026-11-20",
        "optionsData": {
            "ratios": [
                {"name": "TOTAL PUT/CALL RATIO", "value": "1.13"},
                {"name": "SPX + SPXW + SPXO PUT/CALL RATIO", "value": "1.41"},
            ],
            "SPX + SPXW + SPXO": [{"name": "VOLUME", "call": "1,000", "put": "1,410", "total": "2,410"}],
        },
    }
    calls = {"n": 0}
    def fake_extract():
        calls["n"] += 1
        return fake_payload
    cp._extract_json = fake_extract
    assert cp.is_available() is True
    out = cp.get_options_data()
    assert calls["n"] == 1
    assert out["spx_pcr"] == pytest.approx(1.41)
    assert out["spx_call_volume"] == pytest.approx(1000.0)
