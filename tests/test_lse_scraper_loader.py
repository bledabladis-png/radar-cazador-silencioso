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

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.external.lse_scraper_loader import (
    _extract_session_row,
    _ric_for_ticker,
    _to_float,
    build_lse_close_override,
    load_lse_close_for_session,
    write_lse_provenance,
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



# ---------------- build_lse_close_override ----------------

def test_override_estructura_basica():
    loaded = {
        "AZN.L": {"date": "2026-09-25", "open": 12400.0, "high": 12614.0,
                  "low": 12398.0, "close": 12552.0},
        "HSBA.L": {"date": "2026-09-25", "open": 1515.4, "high": 1523.6,
                   "low": 1507.6, "close": 1512.2},
    }
    df = build_lse_close_override(loaded)
    assert df.shape == (1, 2)
    assert df.columns.nlevels == 2
    assert ("Close", "AZN.L") in df.columns
    assert ("Close", "HSBA.L") in df.columns


def test_override_solo_columna_close():
    """No debe haber columnas Open/High/Low/Volume: solo Close."""
    loaded = {
        "AZN.L": {"date": "2026-09-25", "open": 12400.0, "high": 12614.0,
                  "low": 12398.0, "close": 12552.0},
    }
    df = build_lse_close_override(loaded)
    levels_0 = set(c[0] for c in df.columns)
    assert levels_0 == {"Close"}


def test_override_sin_conversion_unidad():
    """Los valores quedan en GBX (sin /100)."""
    loaded = {
        "AZN.L": {"date": "2026-09-25", "open": 12400.0, "high": 12614.0,
                  "low": 12398.0, "close": 12552.0},
    }
    df = build_lse_close_override(loaded)
    assert df[("Close", "AZN.L")].iloc[0] == 12552.0


def test_override_indice_temporal_correcto():
    loaded = {
        "AZN.L": {"date": "2026-09-25", "open": 12400.0, "high": 12614.0,
                  "low": 12398.0, "close": 12552.0},
    }
    df = build_lse_close_override(loaded)
    assert len(df.index) == 1
    assert str(df.index[0].date()) == "2026-09-25"


def test_override_tickers_ordenados():
    """Las columnas se generan en orden alfabetico."""
    loaded = {
        "SHEL.L": {"date": "2026-09-25", "open": 3605.5, "high": 3633.5,
                   "low": 3580.0, "close": 3611.0},
        "AZN.L": {"date": "2026-09-25", "open": 12400.0, "high": 12614.0,
                  "low": 12398.0, "close": 12552.0},
    }
    df = build_lse_close_override(loaded)
    tickers = [c[1] for c in df.columns]
    assert tickers == ["AZN.L", "SHEL.L"]


def test_override_dict_vacio_falla():
    import pytest
    with pytest.raises(ValueError, match="dict vacio"):
        build_lse_close_override({})


def test_override_fechas_heterogeneas_falla():
    """Todas las entradas deben tener la misma fecha."""
    import pytest
    loaded = {
        "AZN.L": {"date": "2026-09-25", "open": 12400.0, "high": 12614.0,
                  "low": 12398.0, "close": 12552.0},
        "HSBA.L": {"date": "2026-09-24", "open": 1515.4, "high": 1523.6,
                   "low": 1507.6, "close": 1512.2},
    }
    with pytest.raises(ValueError, match="no homogeneas"):
        build_lse_close_override(loaded)


def test_override_dtype_index_datetime():
    import pandas as pd
    loaded = {
        "AZN.L": {"date": "2026-09-25", "open": 12400.0, "high": 12614.0,
                  "low": 12398.0, "close": 12552.0},
    }
    df = build_lse_close_override(loaded)
    assert isinstance(df.index, pd.DatetimeIndex)



# ---------------- write_lse_provenance ----------------

def _base_kwargs():
    return dict(
        target_session="2026-09-25",
        lse_expected_session="2026-09-25",
        source_repo="bledabladis-png/lse-close-scraper",
        source_ref="main",
        source_commit="abc123def456",
        tickers_from_scraper=["AZN.L", "HSBA.L"],
        tickers_from_yahoo=[],
        run_id="20260925_120000",
    )


def test_provenance_basico(tmp_path):
    out = tmp_path / "prov.json"
    payload = write_lse_provenance(out, **_base_kwargs())
    assert payload["target_session"] == "2026-09-25"
    assert payload["lse_expected_session"] == "2026-09-25"
    assert payload["source_commit"] == "abc123def456"
    assert payload["scraper_used"] is True
    assert out.exists()


def test_provenance_estructura_completa(tmp_path):
    out = tmp_path / "prov.json"
    payload = write_lse_provenance(out, **_base_kwargs())
    expected = {
        "target_session", "lse_expected_session", "source_repo",
        "source_ref", "source_commit", "tickers_from_scraper",
        "tickers_from_yahoo", "run_id", "scraper_used",
    }
    assert set(payload.keys()) == expected


def test_provenance_sin_scraper_commit_none(tmp_path):
    """Si no se uso scraper, commit puede ser None."""
    out = tmp_path / "prov.json"
    kw = _base_kwargs()
    kw["tickers_from_scraper"] = []
    kw["tickers_from_yahoo"] = ["AZN.L", "HSBA.L"]
    kw["source_commit"] = None
    payload = write_lse_provenance(out, **kw)
    assert payload["scraper_used"] is False
    assert payload["source_commit"] is None


def test_provenance_scraper_usado_commit_vacio_falla(tmp_path):
    """Dictamen D4: commit obligatorio si scraper usado."""
    out = tmp_path / "prov.json"
    kw = _base_kwargs()
    kw["source_commit"] = ""
    with pytest.raises(ValueError, match="source_commit obligatorio"):
        write_lse_provenance(out, **kw)


def test_provenance_scraper_usado_commit_none_falla(tmp_path):
    out = tmp_path / "prov.json"
    kw = _base_kwargs()
    kw["source_commit"] = None
    with pytest.raises(ValueError, match="source_commit obligatorio"):
        write_lse_provenance(out, **kw)


def test_provenance_fechas_aceptan_date(tmp_path):
    """date obj se serializa a ISO."""
    from datetime import date
    out = tmp_path / "prov.json"
    kw = _base_kwargs()
    kw["target_session"] = date(2026, 9, 25)
    kw["lse_expected_session"] = date(2026, 9, 25)
    payload = write_lse_provenance(out, **kw)
    assert payload["target_session"] == "2026-09-25"
    assert payload["lse_expected_session"] == "2026-09-25"


def test_provenance_tickers_ordenados(tmp_path):
    out = tmp_path / "prov.json"
    kw = _base_kwargs()
    kw["tickers_from_scraper"] = ["SHEL.L", "AZN.L", "HSBA.L"]
    kw["tickers_from_yahoo"] = ["ULVR.L", "BA.L"]
    payload = write_lse_provenance(out, **kw)
    assert payload["tickers_from_scraper"] == ["AZN.L", "HSBA.L", "SHEL.L"]
    assert payload["tickers_from_yahoo"] == ["BA.L", "ULVR.L"]


def test_provenance_escritura_atomica(tmp_path):
    """No debe quedar .tmp residual."""
    out = tmp_path / "prov.json"
    write_lse_provenance(out, **_base_kwargs())
    assert out.exists()
    tmps = list(tmp_path.glob("*.tmp.*"))
    assert tmps == []


def test_provenance_json_ordenado(tmp_path):
    """sort_keys=True -> claves alfabeticas en el JSON."""
    import json as _json
    out = tmp_path / "prov.json"
    write_lse_provenance(out, **_base_kwargs())
    data = _json.loads(out.read_text(encoding="utf-8"))
    keys = list(data.keys())
    assert keys == sorted(keys)


def test_provenance_round_trip(tmp_path):
    """Lo escrito en disco == lo devuelto por la funcion."""
    import json as _json
    out = tmp_path / "prov.json"
    payload = write_lse_provenance(out, **_base_kwargs())
    on_disk = _json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == payload
