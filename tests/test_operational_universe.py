"""Tests §5.5 - operational_universe (dictamen #65)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.institutional_accumulation import operational_universe as ou
from src.institutional_accumulation.sec_13f.identity import sec13f_list as sl


PERIOD = "2026-03-31"


def _official_active(cusip):
    """Genera official_df minimo con CUSIP activo."""
    # Linea fixed-width 80: CUSIP(9) + option(1) + name(30) + desc(30) + status(10)
    name = "TEST ISSUER".ljust(30)
    desc = "COM".ljust(30)
    status = " " * 10
    line = cusip.ljust(9) + " " + name + desc + status
    return sl.parse_official_list_text(line)


def _official_not_in_list():
    return sl.parse_official_list_text("")


def _infotable_row(cusip, toc, ssh="1000", putcall=None, ssh_type="SH"):
    return {
        "CUSIP": cusip,
        "TITLEOFCLASS": toc,
        "SSHPRNAMT": ssh,
        "SSHPRNAMTTYPE": ssh_type,
        "PUTCALL": putcall,
    }


def _identity_ok(ticker="AAPL"):
    return {
        "observed_security_key": "cusip:037833100",
        "security_resolution_status": "CANONICAL",
        "canonical_security_kind": "CANONICAL_EQUIVALENCE",
        "canonical_security": "equity:" + ticker,
        "operational_mapping_status": "VERIFIED",
    }


def _identity_status(status_res, status_map, canon="equity:X"):
    return {
        "observed_security_key": "cusip:037833100",
        "security_resolution_status": status_res,
        "canonical_security_kind": "CANONICAL_EQUIVALENCE",
        "canonical_security": canon,
        "operational_mapping_status": status_map,
    }

# --- T-1: mapping unico + VERIFIED -> incluido ---

def test_T1_mapping_ok_incluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_ok()}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids)
    assert len(out) == 1
    assert out.iloc[0]["CUSIP"] == "037833100"
    assert out.iloc[0]["_ticker_mapped"] == True


# --- T-2..T-6: estados no-validos -> excluidos ---

def test_T2_unresolved_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_status("UNRESOLVED", "UNRESOLVED", None)}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids)
    assert len(out) == 0


def test_T3_temporal_unverified_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_status("CANONICAL", "TEMPORAL_UNVERIFIED")}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids)
    assert len(out) == 0


def test_T4_provisional_excluido():
    """OBSERVED_ONLY + UNRESOLVED (provisional)."""
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_status("OBSERVED_ONLY", "UNRESOLVED", None)}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids)
    assert len(out) == 0


def test_T5_ambiguous_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_status("AMBIGUOUS", "UNRESOLVED", None)}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids)
    assert len(out) == 0


def test_T6_conflict_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_status("CONFLICT", "CONFLICT", None)}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids)
    assert len(out) == 0

# --- T-7: NON_EQUITY + ticker valido -> excluido ---

def test_T7_non_equity_excluido():
    info = pd.DataFrame([_infotable_row("329882225", "CONVERTIBLE BOND")])
    off = _official_active("329882225")
    ids = {"329882225": _identity_ok("NIPST")}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids)
    assert len(out) == 0


# --- T-8: tipo UNRESOLVED + ticker presente -> excluido ---

def test_T8_tipo_unresolved_excluido():
    # CLASS A sigue UNRESOLVED (dictamen #62/#63) aunque el ticker exista.
    info = pd.DataFrame([_infotable_row("037833100", "CLASS A")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_ok()}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids)
    assert len(out) == 0


# --- T-9: orden de filas invariante ---

def test_T9_orden_invariante():
    rows = [
        _infotable_row("037833100", "COM"),
        _infotable_row("594918104", "COM"),
        _infotable_row("329882225", "CONVERTIBLE BOND"),
    ]
    info = pd.DataFrame(rows)
    off = sl.parse_official_list_text("\n".join([
        "037833100" + " " + "TEST".ljust(30) + "COM".ljust(30) + " " * 10,
        "594918104" + " " + "TEST2".ljust(30) + "COM".ljust(30) + " " * 10,
        "329882225" + " " + "TEST3".ljust(30) + "CVT".ljust(30) + " " * 10,
    ]))
    ids = {
        "037833100": _identity_ok("AAPL"),
        "594918104": _identity_ok("MSFT"),
        "329882225": _identity_ok("NIPST"),
    }
    out1 = ou.build_operational_universe(info, off, PERIOD, identity_results=ids)
    out2 = ou.build_operational_universe(info.iloc[::-1], off, PERIOD, identity_results=ids)
    assert len(out1) == 2
    assert set(out1["CUSIP"]) == set(out2["CUSIP"])

# --- T-10: universo vacio ---

def test_T10_universo_vacio():
    info = pd.DataFrame([], columns=["CUSIP","TITLEOFCLASS","SSHPRNAMTTYPE","PUTCALL"])
    off = _official_active("037833100")
    out = ou.build_operational_universe(info, off, PERIOD, identity_results={})
    assert len(out) == 0


# --- T-11: identity_results=None -> ValueError ---

def test_T11_identity_none():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    with pytest.raises(ValueError):
        ou.build_operational_universe(info, off, PERIOD, identity_results=None)


# --- T-12: NOT_IN_LIST -> excluido ---

def test_T12_not_in_list_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_not_in_list()
    ids = {"037833100": _identity_ok()}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids)
    assert len(out) == 0


# --- T-13: PRN o PUTCALL != NULL -> excluido ---

def test_T13a_prn_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM", ssh_type="PRN")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_ok()}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids)
    assert len(out) == 0


def test_T13b_putcall_not_null_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM", putcall="CALL")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_ok()}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids)
    assert len(out) == 0

# --- T-14: identity_results de otro periodo -> ValueError ---

def test_T14_pit_period_mismatch():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_ok()}
    with pytest.raises(ValueError):
        ou.build_operational_universe(
            info, off, PERIOD,
            identity_results=ids,
            identity_period_iso="2025-12-31",
        )


def test_T14_pit_period_match_ok():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_ok()}
    out = ou.build_operational_universe(
        info, off, PERIOD,
        identity_results=ids,
        identity_period_iso=PERIOD,
    )
    assert len(out) == 1


# --- Tests columnares requeridas ---

def test_columnas_requeridas_faltantes():
    info = pd.DataFrame([{"CUSIP": "X"}])
    off = _official_active("X")
    with pytest.raises(ValueError):
        ou.build_operational_universe(info, off, PERIOD, identity_results={})


def test_columnas_diagnosticas_presentes():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_ok()}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids)
    for c in ou.DIAGNOSTIC_COLUMNS:
        assert c in out.columns, "falta " + c