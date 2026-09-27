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
