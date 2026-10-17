# -*- coding: utf-8 -*-
"""REP-AUDIT-20261017-H11/H12: distinguir fecha de observacion
(REPORT_DATE) de fecha de presentacion (FILING_DATE) en N-PORT.
"""
import pandas as pd

from src.report.flows_international import render_flujo_posicional_nport


def test_render_con_filing_date():
    df = pd.DataFrame([{
        "REPORT_DATE": pd.Timestamp("2026-03-31"),
        "FILING_DATE": pd.Timestamp("2026-05-15"),
        "REGISTRANT_NAME": "SELECT SECTOR SPDR",
        "ISSUER_NAME": "Norwegian Cruise",
        "IDENTIFIER_ISIN": "BMG667211046",
        "PREV_BALANCE": 2_543_505,
        "BALANCE": 2_429_969,
        "POSITION_CHANGE": -113_536,
        "POSITION_CHANGE_PCT": -4.4638,
    }])
    out = "".join(render_flujo_posicional_nport(df))
    assert "2026-03-31" in out
    assert "2026-05-15" in out
    assert "Fecha observacion" in out
    assert "Fecha presentacion" in out


def test_render_sin_filing_date_fallback_nd():
    df = pd.DataFrame([{
        "REPORT_DATE": pd.Timestamp("2026-03-31"),
        "REGISTRANT_NAME": "SELECT SECTOR SPDR",
        "ISSUER_NAME": "Norwegian Cruise",
        "IDENTIFIER_ISIN": "BMG667211046",
        "PREV_BALANCE": 2_543_505,
        "BALANCE": 2_429_969,
        "POSITION_CHANGE": -113_536,
        "POSITION_CHANGE_PCT": -4.4638,
    }])
    out = "".join(render_flujo_posicional_nport(df))
    assert "2026-03-31" in out
    assert "N/D" in out
    assert "Fecha presentacion" in out


def test_render_filing_date_nat():
    df = pd.DataFrame([{
        "REPORT_DATE": pd.Timestamp("2026-03-31"),
        "FILING_DATE": pd.NaT,
        "REGISTRANT_NAME": "X", "ISSUER_NAME": "Y", "IDENTIFIER_ISIN": "Z",
        "PREV_BALANCE": 100, "BALANCE": 110,
        "POSITION_CHANGE": 10, "POSITION_CHANGE_PCT": 10.0,
    }])
    out = "".join(render_flujo_posicional_nport(df))
    assert "N/D" in out


def test_nota_declara_frecuencias():
    df = pd.DataFrame([{
        "REPORT_DATE": pd.Timestamp("2026-03-31"),
        "FILING_DATE": pd.Timestamp("2026-05-15"),
        "REGISTRANT_NAME": "X", "ISSUER_NAME": "Y", "IDENTIFIER_ISIN": "Z",
        "PREV_BALANCE": 100, "BALANCE": 110,
        "POSITION_CHANGE": 10, "POSITION_CHANGE_PCT": 10.0,
    }])
    out = "".join(render_flujo_posicional_nport(df))
    assert "mensual" in out.lower()
    assert "trimestral" in out.lower()
    assert "observacion" in out.lower()
    assert "presentacion" in out.lower()


def test_data_quality_notes_declaran_4_dimensiones():
    from indicators.data_quality import SOURCES
    nport = next(s for s in SOURCES if s["source"] == "SEC N-PORT (posiciones)")
    notes = nport["notes"]
    assert "mensual" in notes.lower()
    assert "trimestral" in notes.lower()
    assert "observacion" in notes.lower() or "presentacion" in notes.lower()
