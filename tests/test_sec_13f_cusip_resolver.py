# -*- coding: utf-8 -*-
"""Tests de sec_13f.identity.cusip_resolver (FA-2.2). Sin red."""
import pandas as pd
import pytest

from src.institutional_accumulation.sec_13f.identity import cusip_resolver as cr


def _mk_exc(rows):
    """rows: list of (CUSIP, ticker, from, to, source, reason, title, verified_by)."""
    df = pd.DataFrame(rows, columns=[
        "CUSIP", "ticker", "valid_from", "valid_to",
        "source", "reason", "title_of_class", "verified_by",
    ])
    df["valid_from"] = pd.to_datetime(df["valid_from"])
    df["valid_to"] = pd.to_datetime(df["valid_to"], errors="coerce")
    return df


# ---- contrato ----

def test_columnas_requeridas_constante():
    assert "CUSIP" in cr.REQUIRED_COLUMNS
    assert "title_of_class" in cr.REQUIRED_COLUMNS
    assert "verified_by" in cr.REQUIRED_COLUMNS


def test_load_exceptions_fichero_no_existe_devuelve_vacio(tmp_path):
    df = cr.load_exceptions(tmp_path / "no_existe.csv")
    assert df.empty
    assert list(df.columns) == list(cr.REQUIRED_COLUMNS)


# ---- validaciones de tabla ----

def test_load_exceptions_columnas_faltantes_lanza(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("CUSIP,ticker\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Columnas faltantes"):
        cr.load_exceptions(p)


def test_valid_from_mayor_que_valid_to_lanza(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text(
        "CUSIP,ticker,valid_from,valid_to,source,reason,title_of_class,verified_by\n"
        "X,Y,2026-12-31,2026-01-01,SEC,r,COM,manual\n", encoding="utf-8")
    with pytest.raises(ValueError, match="valid_from > valid_to"):
        cr.load_exceptions(p)


def test_source_prohibido_lanza(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text(
        "CUSIP,ticker,valid_from,valid_to,source,reason,title_of_class,verified_by\n"
        "X,Y,2026-01-01,,manual,r,COM,manual\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source no verificable"):
        cr.load_exceptions(p)


def test_verified_by_invalido_lanza(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text(
        "CUSIP,ticker,valid_from,valid_to,source,reason,title_of_class,verified_by\n"
        "X,Y,2026-01-01,,SEC,r,COM,inventado\n", encoding="utf-8")
    with pytest.raises(ValueError, match="verified_by invalido"):
        cr.load_exceptions(p)


def test_solape_lanza(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text(
        "CUSIP,ticker,valid_from,valid_to,source,reason,title_of_class,verified_by\n"
        "X,Y,2024-01-01,2025-12-31,SEC,r1,COM,manual\n"
        "X,Y,2025-06-01,2026-12-31,SEC,r2,COM,manual\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Solape"):
        cr.load_exceptions(p)


# ---- resolve_cusip ----

def test_resolve_cusip_match_exacto():
    exc = _mk_exc([("A1", "T1", "2024-01-01", "2026-12-31", "SEC", "r", "COM", "manual")])
    assert cr.resolve_cusip("A1", "2026-03-31", exc) == "T1"


def test_resolve_cusip_sin_match_devuelve_none():
    exc = _mk_exc([("A1", "T1", "2024-01-01", "2026-12-31", "SEC", "r", "COM", "manual")])
    assert cr.resolve_cusip("B2", "2026-03-31", exc) is None


def test_resolve_cusip_periodo_fuera_de_vigencia_devuelve_none():
    exc = _mk_exc([("A1", "T1", "2024-01-01", "2024-12-31", "SEC", "r", "COM", "manual")])
    assert cr.resolve_cusip("A1", "2026-03-31", exc) is None


def test_resolve_cusip_vigencia_abierta_valid_to_null():
    exc = _mk_exc([("A1", "T1", "2026-01-01", None, "SEC", "r", "COM", "manual")])
    assert cr.resolve_cusip("A1", "2030-01-01", exc) == "T1"


def test_resolve_cusip_multiples_variantes_temporales():
    exc = _mk_exc([
        ("A1", "T1", "2015-01-01", "2019-12-31", "SEC", "pre", "COM", "manual"),
        ("B2", "T1", "2020-01-01", "2024-12-31", "SEC", "post", "COM", "manual"),
        ("C3", "T1", "2025-01-01", None,       "SEC", "cur",  "COM", "manual"),
    ])
    assert cr.resolve_cusip("A1", "2016-06-30", exc) == "T1"
    assert cr.resolve_cusip("B2", "2022-06-30", exc) == "T1"
    assert cr.resolve_cusip("C3", "2026-03-31", exc) == "T1"
    assert cr.resolve_cusip("A1", "2026-03-31", exc) is None


def test_resolve_cusip_ambiguo_lanza():
    exc = _mk_exc([
        ("A1", "T1", "2024-01-01", "2026-12-31", "SEC", "r", "COM", "manual"),
        ("A1", "T2", "2025-01-01", "2026-12-31", "SEC", "r", "COM", "manual"),
    ])
    with pytest.raises(ValueError, match="Multiples tickers"):
        cr.resolve_cusip("A1", "2026-03-31", exc)


def test_resolve_cusip_empty_df_devuelve_none():
    assert cr.resolve_cusip("A1", "2026-03-31", pd.DataFrame()) is None


# ---- resolve_batch ----

def test_resolve_batch_anade_columna():
    df = pd.DataFrame({"CUSIP": ["A1", "B2", "C3"]})
    exc = _mk_exc([
        ("A1", "T1", "2024-01-01", None, "SEC", "r", "COM", "manual"),
        ("C3", "T3", "2024-01-01", None, "SEC", "r", "COM", "manual"),
    ])
    out = cr.resolve_batch(df, "2026-03-31", exc)
    assert "ticker_resolved" in out.columns
    assert out["ticker_resolved"].tolist() == ["T1", None, "T3"]


def test_resolve_batch_no_muta_original():
    df = pd.DataFrame({"CUSIP": ["A1"]})
    exc = _mk_exc([("A1", "T1", "2024-01-01", None, "SEC", "r", "COM", "manual")])
    cr.resolve_batch(df, "2026-03-31", exc)
    assert "ticker_resolved" not in df.columns


def test_resolve_batch_sin_columna_cusip_lanza():
    with pytest.raises(KeyError, match="CUSIP"):
        cr.resolve_batch(pd.DataFrame({"X": [1]}), "2026-03-31", pd.DataFrame())
