"""Tests H-69.2 - precedencia por evidencia temporal (dictamen #71).

Regla R-69.2: si hay multiples candidatos de crosswalk para un
CUSIP/period_end, y exactamente 1 tiene vigencia temporal VERIFIED,
gana ese. Los sin vigencia no generan CONFLICT contra un unico
VERIFIED. Dos o mas VERIFIED -> CONFLICT. Cero VERIFIED -> UNRESOLVED.
"""
from __future__ import annotations

import pandas as pd

from src.institutional_accumulation.sec_13f.identity import security_identity as si


def _cw(rows):
    """rows: list of (cusip, ticker, source, valid_from, valid_to).

    valid_from/valid_to se convierten a Timestamp si no son None.
    """
    def _ts(v):
        if v is None:
            return pd.NaT
        return pd.Timestamp(v).normalize()
    return pd.DataFrame([
        {"CUSIP": c, "ticker": t, "source": s,
         "valid_from": _ts(vf), "valid_to": _ts(vt)}
        for c, t, s, vf, vt in rows
    ], columns=["CUSIP", "ticker", "source", "valid_from", "valid_to"])


PERIOD = "2026-03-31"


# --- H69.2-a: 1 verified + 1 sin vigencia -> gana verified ---

def test_h692a_uno_verified_uno_sin_vigencia_gana_verified():
    df = _cw([
        ("AAA", "BRK-B", "cusip_ticker_exceptions", "2026-03-31", "2026-03-31"),
        ("AAA", "BRK.B", "etf_holdings", None, None),
    ])
    r = si.resolve_security_identity("AAA", PERIOD, crosswalk_internal_df=df)
    assert r["security_resolution_status"] == "CANONICAL"
    assert r["operational_mapping_status"] == "VERIFIED"
    assert r["canonical_security"] == "equity:BRK-B"


# --- H69.2-b: 2 verified -> CONFLICT ---

def test_h692b_dos_verified_conflict():
    df = _cw([
        ("BBB", "TICK_A", "cusip_ticker_exceptions", "2026-03-31", "2026-03-31"),
        ("BBB", "TICK_B", "cusip_ticker_exceptions", "2026-03-31", "2026-03-31"),
    ])
    r = si.resolve_security_identity("BBB", PERIOD, crosswalk_internal_df=df)
    assert r["security_resolution_status"] == "CONFLICT"
    assert r["operational_mapping_status"] == "UNRESOLVED"


# --- H69.2-c: 0 verified -> UNRESOLVED (no fabrica VERIFIED) ---

def test_h692c_cero_verified_no_fabrica_verified():
    df = _cw([
        ("CCC", "TICK_X", "etf_holdings", None, None),
        ("CCC", "TICK_Y", "etf_holdings", None, None),
    ])
    r = si.resolve_security_identity("CCC", PERIOD, crosswalk_internal_df=df)
    # 2 candidatos sin vigencia: no debe producir VERIFIED
    assert r["operational_mapping_status"] != "VERIFIED"


# --- H69.2-d: 1 candidato verified -> comportamiento preservado ---

def test_h692d_un_candidato_verified_preservado():
    df = _cw([
        ("DDD", "TICK_Z", "cusip_ticker_exceptions", "2026-03-31", "2026-03-31"),
    ])
    r = si.resolve_security_identity("DDD", PERIOD, crosswalk_internal_df=df)
    assert r["security_resolution_status"] == "CANONICAL"
    assert r["operational_mapping_status"] == "VERIFIED"


# --- H69.2-e: fuentes no implicadas preservan comportamiento ---

def test_h692e_etf_holdings_solo_no_verified():
    df = _cw([
        ("EEE", "TICK_E", "etf_holdings", None, None),
    ])
    r = si.resolve_security_identity("EEE", PERIOD, crosswalk_internal_df=df)
    assert r["security_resolution_status"] == "CANONICAL"
    assert r["operational_mapping_status"] == "TEMPORAL_UNVERIFIED"


def test_h692e_exceptions_fuera_de_vigencia_unresolved():
    df = _cw([
        ("FFF", "TICK_F", "cusip_ticker_exceptions", "2026-01-01", "2026-01-31"),
    ])
    r = si.resolve_security_identity("FFF", PERIOD, crosswalk_internal_df=df)
    assert r["operational_mapping_status"] == "UNRESOLVED"


# --- H69.2-f: caso concreto 084670702 ---

def test_h692f_brk_b_canonical_verified():
    cw = si.load_crosswalk_internal()
    r = si.resolve_security_identity("084670702", PERIOD, crosswalk_internal_df=cw)
    assert r["security_resolution_status"] == "CANONICAL"
    assert r["operational_mapping_status"] == "VERIFIED"
    assert r["canonical_security"] == "equity:BRK-B"