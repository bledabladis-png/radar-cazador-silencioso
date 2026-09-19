"""Tests unitarios del cliente OpenFIGI (sin red)."""
from __future__ import annotations

import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from src.institutional_accumulation.identity.openfigi_client import (
    VALID_ID_TYPES,
    extract_stable_identity,
    map_identifiers,
)


def _fake_response(body: bytes):
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda self, *a: None
    return resp


def test_id_type_invalido_lanza():
    with pytest.raises(ValueError):
        map_identifiers("ID_MADEUP", ["X"])


def test_lista_vacia_devuelve_dict_vacio():
    assert map_identifiers("ID_CUSIP", []) == {}
    assert map_identifiers("ID_CUSIP", [None, "", "  "]) == {}


def test_hit_con_data():
    body = b'[{"data": [{"figi": "BBG000BSJK37", "shareClassFIGI": "BBG001S5VWH2", "ticker": "T", "name": "AT&T INC", "securityType": "Common Stock", "marketSector": "Equity", "exchCode": "US"}]}]'
    with patch("urllib.request.urlopen", return_value=_fake_response(body)):
        out = map_identifiers("ID_CUSIP", ["00206R102"], api_key="fake")
    assert out["00206R102"]["ok"] is True
    assert out["00206R102"]["data"][0]["ticker"] == "T"


def test_warning_no_identifier_found():
    body = b'[{"warning": "No identifier found."}]'
    with patch("urllib.request.urlopen", return_value=_fake_response(body)):
        out = map_identifiers("ID_CUSIP", ["999999999"], api_key="fake")
    assert out["999999999"]["ok"] is False
    assert "No identifier found" in out["999999999"]["error"]


def test_error_400_no_reintenta():
    err = urllib.error.HTTPError("http://x", 400, "Bad Request", {}, MagicMock())
    err.read = lambda: b"invalid"
    with patch("urllib.request.urlopen", side_effect=err):
        out = map_identifiers("ID_CUSIP", ["X"], api_key="fake")
    assert out["X"]["ok"] is False
    assert "HTTP 400" in out["X"]["error"]


def test_extract_stable_identity_prioriza_us():
    hit = {
        "ok": True,
        "data": [
            {"figi": "F1", "shareClassFIGI": "SC1", "ticker": "X1", "exchCode": "LN"},
            {"figi": "F2", "shareClassFIGI": "SC2", "ticker": "X2", "exchCode": "US"},
        ],
    }
    out = extract_stable_identity(hit)
    assert out["figi"] == "F2"
    assert out["share_class_figi"] == "SC2"
    assert out["ticker"] == "X2"


def test_extract_stable_identity_sin_hit():
    assert extract_stable_identity(None) is None
    assert extract_stable_identity({"ok": False, "data": None, "error": "x"}) is None
    assert extract_stable_identity({"ok": True, "data": []}) is None


def test_valid_id_types_contiene_los_cuatro():
    assert set(VALID_ID_TYPES) == {"ID_CUSIP", "ID_ISIN", "ID_EXCH_SYMBOL", "TICKER"}
