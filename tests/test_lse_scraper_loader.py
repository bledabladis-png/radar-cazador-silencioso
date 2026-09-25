# -*- coding: utf-8 -*-
"""Tests del loader LSE scraper (F-IAE-LSE-INTEGRATION).

Cubre:
  - _ric_for_ticker: mapeo ticker -> RIC Refinitiv via INSTRUMENTS.
  - _to_float: conversion tolerante a fallos.
  - _extract_session_row: busqueda exacta por fecha.
  - load_lse_close_for_session: end-to-end con JSON mockeados.
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.external.lse_scraper_loader import (
    _extract_session_row,
    _ric_for_ticker,
    _to_float,
    load_lse_close_for_session,
)


# ---------------- _ric_for_ticker ----------------

def test_ric_ba_maps_baes():
    """BA.L (ticker radar) -> BAES.L (RIC Refinitiv)."""
    assert _ric_for_ticker("BA.L") == "BAES.L"


def test_ric_dge_identity():
    assert _ric_for_ticker("DGE.L") == "DGE.L"


def test_ric_azn_identity():
    assert _ric_for_ticker("AZN.L") == "AZN.L"


def test_ric_ticker_desconocido():
    assert _ric_for_ticker("XYZ.L") is None


def test_ric_ticker_usa_sin_refinitiv():
    """AAPL no esta en INSTRUMENTS con refinitiv."""
    assert _ric_for_ticker("AAPL") is None


# ---------------- _to_float ----------------

def test_to_float_str_normal():
    assert _to_float("12552.0") == 12552.0


def test_to_float_int():
    assert _to_float(42) == 42.0


def test_to_float_float():
    assert _to_float(3.14) == 3.14


def test_to_float_none():
    assert _to_float(None) is None


def test_to_float_str_invalido():
    assert _to_float("abc") is None


def test_to_float_str_vacia():
    assert _to_float("") is None


def test_to_float_str_con_espacios():
    assert _to_float("  125.5  ") == 125.5


# ---------------- _extract_session_row ----------------

def _make_json(rows):
    """Construye un dict JSON estilo scraper con las filas dadas."""
    return {
        "request": {"ric": "TEST.L"},
        "data": rows,
        "status": "OK",
        "metadata": {},
    }


def _make_row(date_str, open_="100.0", high="110.0", low="90.0", close="105.0"):
    return {
        "_DATE_END": date_str,
        "OPEN_PRC": open_,
        "HIGH_1": high,
        "LOW_1": low,
        "CLOSE_PRC": close,
    }


def test_extract_fila_existe():
    json_data = _make_json([
        _make_row("2026-09-24", close="100.0"),
        _make_row("2026-09-25", close="105.0"),
    ])
    row = _extract_session_row(json_data, date(2026, 9, 25))
    assert row is not None
    assert row["date"] == "2026-09-25"
    assert row["close"] == 105.0


def test_extract_fila_no_existe():
    """Si no hay fila de esa fecha exacta, devuelve None.

    No sustituye por max(_DATE_END). Debe ser esa fecha exacta.
    """
    json_data = _make_json([
        _make_row("2026-09-23"),
        _make_row("2026-09-24"),
    ])
    assert _extract_session_row(json_data, date(2026, 9, 25)) is None


def test_extract_data_vacia():
    json_data = _make_json([])
    assert _extract_session_row(json_data, date(2026, 9, 25)) is None


def test_extract_substatus_backend_error():
    """RIC invalido devuelve data:[] con substatus."""
    json_data = {
        "request": {"ric": "BAD.L"},
        "data": [],
        "status": "OK",
        "substatus": "BackendError",
    }
    assert _extract_session_row(json_data, date(2026, 9, 25)) is None


def test_extract_close_nan_devuelve_none():
    """Si la fila existe pero Close no es numerico, invalida."""
    json_data = _make_json([
        _make_row("2026-09-25", close=""),
    ])
    assert _extract_session_row(json_data, date(2026, 9, 25)) is None


def test_extract_ohlc_incompletos():
    """Si OHLC tienen NaN pero Close es valido, se acepta."""
    row = {
        "_DATE_END": "2026-09-25",
        "OPEN_PRC": "",
        "HIGH_1": "",
        "LOW_1": "",
        "CLOSE_PRC": "105.0",
    }
    json_data = _make_json([row])
    result = _extract_session_row(json_data, date(2026, 9, 25))
    assert result is not None
    assert result["close"] == 105.0
    assert result["open"] is None
    assert result["high"] is None
    assert result["low"] is None


def test_extract_json_no_dict():
    assert _extract_session_row(None, date(2026, 9, 25)) is None
    assert _extract_session_row("no es dict", date(2026, 9, 25)) is None


def test_extract_data_no_lista():
    json_data = {"data": "no es lista"}
    assert _extract_session_row(json_data, date(2026, 9, 25)) is None


# ---------------- load_lse_close_for_session ----------------

def _write_json(datos_dir, ric, rows):
    """Escribe un JSON estilo scraper en datos_dir."""
    datos_dir.mkdir(parents=True, exist_ok=True)
    filename = ric.replace(".", "_") + ".json"
    payload = {
        "request": {"ric": ric},
        "data": rows,
        "status": "OK",
    }
    (datos_dir / filename).write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_load_sesion_valida(tmp_path):
    _write_json(tmp_path, "AZN.L", [_make_row("2026-09-25", close="12552.0")])
    _write_json(tmp_path, "HSBA.L", [_make_row("2026-09-25", close="1512.2")])
    result = load_lse_close_for_session(
        tmp_path, date(2026, 9, 25), ["AZN.L", "HSBA.L"]
    )
    assert len(result) == 2
    assert result["AZN.L"]["close"] == 12552.0
    assert result["HSBA.L"]["close"] == 1512.2


def test_load_ticker_desconocido_se_omite(tmp_path):
    _write_json(tmp_path, "AZN.L", [_make_row("2026-09-25")])
    result = load_lse_close_for_session(
        tmp_path, date(2026, 9, 25), ["AZN.L", "DESCONOCIDO.L"]
    )
    assert "AZN.L" in result
    assert "DESCONOCIDO.L" not in result


def test_load_fecha_no_presente_se_omite(tmp_path):
    _write_json(tmp_path, "AZN.L", [_make_row("2026-09-24")])
    result = load_lse_close_for_session(
        tmp_path, date(2026, 9, 25), ["AZN.L"]
    )
    assert result == {}


def test_load_fichero_ausente_se_omite(tmp_path):
    result = load_lse_close_for_session(
        tmp_path, date(2026, 9, 25), ["AZN.L"]
    )
    assert result == {}


def test_load_json_invalido_se_omite(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "AZN_L.json").write_text("no json", encoding="utf-8")
    result = load_lse_close_for_session(
        tmp_path, date(2026, 9, 25), ["AZN.L"]
    )
    assert result == {}


def test_load_tickers_vacio(tmp_path):
    result = load_lse_close_for_session(
        tmp_path, date(2026, 9, 25), []
    )
    assert result == {}


def test_load_ba_usa_baes(tmp_path):
    """BA.L debe leer BAES_L.json, no BA_L.json."""
    # Escribimos solo BAES_L.json (RIC real)
    _write_json(tmp_path, "BAES.L", [_make_row("2026-09-25", close="1972.0")])
    # Tambien escribimos BA_L.json con datos erroneos, para verificar
    # que NO se lee
    _write_json(tmp_path, "BA.L", [_make_row("2026-09-25", close="9999.0")])
    result = load_lse_close_for_session(
        tmp_path, date(2026, 9, 25), ["BA.L"]
    )
    assert result["BA.L"]["close"] == 1972.0


def test_load_sin_conversion_unidad(tmp_path):
    """Los valores se devuelven tal cual (GBX, no /100)."""
    _write_json(tmp_path, "AZN.L", [_make_row("2026-09-25", close="12552.0")])
    result = load_lse_close_for_session(
        tmp_path, date(2026, 9, 25), ["AZN.L"]
    )
    assert result["AZN.L"]["close"] == 12552.0
    assert result["AZN.L"]["close"] != 125.52


def test_load_estructura_row_completa(tmp_path):
    _write_json(tmp_path, "AZN.L", [{
        "_DATE_END": "2026-09-25",
        "OPEN_PRC": "12400.0",
        "HIGH_1": "12614.0",
        "LOW_1": "12398.0",
        "CLOSE_PRC": "12552.0",
    }])
    result = load_lse_close_for_session(
        tmp_path, date(2026, 9, 25), ["AZN.L"]
    )
    row = result["AZN.L"]
    assert row["date"] == "2026-09-25"
    assert row["open"] == 12400.0
    assert row["high"] == 12614.0
    assert row["low"] == 12398.0
    assert row["close"] == 12552.0


def test_load_multiples_tickers_mixto(tmp_path):
    """Combinacion de ticker OK + ticker sin fecha + ticker desconocido."""
    _write_json(tmp_path, "AZN.L", [_make_row("2026-09-25")])
    _write_json(tmp_path, "HSBA.L", [_make_row("2026-09-24")])  # no fecha
    result = load_lse_close_for_session(
        tmp_path, date(2026, 9, 25),
        ["AZN.L", "HSBA.L", "DESCONOCIDO.L"]
    )
    assert "AZN.L" in result
    assert "HSBA.L" not in result
    assert "DESCONOCIDO.L" not in result
    assert len(result) == 1
