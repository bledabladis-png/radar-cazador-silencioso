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
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
    assert len(out) == 1
    assert out.iloc[0]["CUSIP"] == "037833100"
    assert out.iloc[0]["_ticker_mapped"] == True


# --- T-2..T-6: estados no-validos -> excluidos ---

def test_T2_unresolved_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_status("UNRESOLVED", "UNRESOLVED", None)}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
    assert len(out) == 0


def test_T3_temporal_unverified_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_status("CANONICAL", "TEMPORAL_UNVERIFIED")}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
    assert len(out) == 0


def test_T4_provisional_excluido():
    """OBSERVED_ONLY + UNRESOLVED (provisional)."""
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_status("OBSERVED_ONLY", "UNRESOLVED", None)}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
    assert len(out) == 0


def test_T5_ambiguous_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_status("AMBIGUOUS", "UNRESOLVED", None)}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
    assert len(out) == 0


def test_T6_conflict_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_status("CONFLICT", "CONFLICT", None)}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
    assert len(out) == 0

# --- T-7: NON_EQUITY + ticker valido -> excluido ---

def test_T7_non_equity_excluido():
    info = pd.DataFrame([_infotable_row("329882225", "CONVERTIBLE BOND")])
    off = _official_active("329882225")
    ids = {"329882225": _identity_ok("NIPST")}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
    assert len(out) == 0


# --- T-8: tipo UNRESOLVED + ticker presente -> excluido ---

def test_T8_tipo_unresolved_excluido():
    # CLASS A sigue UNRESOLVED (dictamen #62/#63) aunque el ticker exista.
    info = pd.DataFrame([_infotable_row("037833100", "CLASS A")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_ok()}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
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
    out1 = ou.build_operational_universe(info, off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
    out2 = ou.build_operational_universe(info.iloc[::-1], off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
    assert len(out1) == 2
    assert set(out1["CUSIP"]) == set(out2["CUSIP"])

# --- T-10: universo vacio ---

def test_T10_universo_vacio():
    info = pd.DataFrame([], columns=["CUSIP","TITLEOFCLASS","SSHPRNAMTTYPE","PUTCALL"])
    off = _official_active("037833100")
    out = ou.build_operational_universe(info, off, PERIOD, identity_results={}, identity_period_iso=PERIOD)
    assert len(out) == 0


# --- T-11: identity_results=None -> ValueError ---

def test_T11_identity_none():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    with pytest.raises(ValueError):
        ou.build_operational_universe(
            info, off, PERIOD,
            identity_results=None,
            identity_period_iso=PERIOD,
        )


# --- T-12: NOT_IN_LIST -> excluido ---

def test_T12_not_in_list_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_not_in_list()
    ids = {"037833100": _identity_ok()}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
    assert len(out) == 0


# --- T-13: PRN o PUTCALL != NULL -> excluido ---

def test_T13a_prn_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM", ssh_type="PRN")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_ok()}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
    assert len(out) == 0


def test_T13b_putcall_not_null_excluido():
    info = pd.DataFrame([_infotable_row("037833100", "COM", putcall="CALL")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_ok()}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
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
        ou.build_operational_universe(info, off, PERIOD, identity_results={}, identity_period_iso=PERIOD)


def test_columnas_diagnosticas_presentes():
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_ok()}
    out = ou.build_operational_universe(info, off, PERIOD, identity_results=ids, identity_period_iso=PERIOD)
    for c in ou.DIAGNOSTIC_COLUMNS:
        assert c in out.columns, "falta " + c

# --- T-14c: identity_period_iso=None -> ValueError (dictamen #66 H-66.1) ---

def test_T14c_identity_period_none_rechazado():
    """Dictamen #66: el contrato PIT es obligatorio. None -> ValueError."""
    info = pd.DataFrame([_infotable_row("037833100", "COM")])
    off = _official_active("037833100")
    ids = {"037833100": _identity_ok()}
    with pytest.raises(ValueError):
        ou.build_operational_universe(
            info, off, PERIOD,
            identity_results=ids,
            identity_period_iso=None,
        )


# --- H1-B: _apply_5_3b ---

def _official_df(rows):
    """Helper: construir official_df minimo para los tests H1-B."""
    import pandas as pd
    cusips = [r[0] for r in rows]
    descs = [r[1] for r in rows]
    return pd.DataFrame({
        "cusip": cusips,
        "option_indicator": ["*" if d == "COM" else " " for d in descs],
        "issuer_name": ["X"] * len(rows),
        "issuer_description": descs,
        "status_raw": ["   "] * len(rows),
        "status": ["ACTIVE"] * len(rows),
        "raw_line": ["x" * 80] * len(rows),
        "_line_no": list(range(1, len(rows) + 1)),
        "_order": list(range(len(rows))),
    })


def _infotable_df(cusips):
    import pandas as pd
    return pd.DataFrame({
        "CUSIP": cusips,
        "TITLEOFCLASS": ["COM"] * len(cusips),
        "SSHPRNAMTTYPE": ["SH"] * len(cusips),
        "PUTCALL": [None] * len(cusips),
        "SSHPRNAMT": ["1000"] * len(cusips),
    })


def test_apply_5_3b_excluye_option():
    from src.institutional_accumulation import operational_universe as ou
    df = _infotable_df(["CALL1"])
    official_df = _official_df([("CALL1", "CALL")])
    out, stats = ou._apply_5_3b(df, official_df)
    assert len(out) == 0
    assert stats["n_option"] == 1
    assert stats["n_equity"] == 0
    assert stats["n_unresolved"] == 0


def test_apply_5_3b_acepta_equity():
    from src.institutional_accumulation import operational_universe as ou
    df = _infotable_df(["EQ1"])
    official_df = _official_df([("EQ1", "COM")])
    out, stats = ou._apply_5_3b(df, official_df)
    assert len(out) == 1
    assert stats["n_equity"] == 1
    assert stats["n_option"] == 0
    assert stats["n_unresolved"] == 0


def test_apply_5_3b_mantiene_unresolved():
    """H1-B v2.3: UNRESOLVED (no en lista) -> se mantiene.

    Motivo: excluirlo produce falsos positivos sobre equity real
    (AMCR, LRCX) que no aparece en la Official List.
    """
    from src.institutional_accumulation import operational_universe as ou
    df = _infotable_df(["MISSING1"])
    official_df = _official_df([("OTHER", "COM")])
    out, stats = ou._apply_5_3b(df, official_df)
    assert len(out) == 1
    assert stats["n_unresolved"] == 1
    assert stats["n_equity"] == 0
    assert stats["n_option"] == 0


def test_apply_5_3b_mixto():
    """H1-B v2.3: EQUITY + UNRESOLVED se mantienen, solo OPTION se excluye."""
    from src.institutional_accumulation import operational_universe as ou
    df = _infotable_df(["EQ1", "CALL1", "MISSING1"])
    official_df = _official_df([("EQ1", "COM"), ("CALL1", "CALL")])
    out, stats = ou._apply_5_3b(df, official_df)
    assert len(out) == 2
    assert stats["n_equity"] == 1
    assert stats["n_option"] == 1
    assert stats["n_unresolved"] == 1


def test_build_operational_universe_audit_stats_en_attrs():
    """El dict audit_stats debe quedar en df.attrs tras build."""
    from src.institutional_accumulation import operational_universe as ou
    infotable = _infotable_df(["EQ1", "CALL1"])
    official_df = _official_df([("EQ1", "COM"), ("CALL1", "CALL")])
    identity_results = {
        "EQ1": {
            "security_resolution_status": "CANONICAL",
            "operational_mapping_status": "VERIFIED",
            "canonical_security": "equity:EQ1",
            "canonical_security_kind": "EQUITY",
            "observed_security_key": "cusip:EQ1",
        },
        "CALL1": {
            "security_resolution_status": "CANONICAL",
            "operational_mapping_status": "VERIFIED",
            "canonical_security": "equity:CALL1",
            "canonical_security_kind": "EQUITY",
            "observed_security_key": "cusip:CALL1",
        },
    }
    out = ou.build_operational_universe(
        infotable,
        official_df,
        "2026-03-31",
        identity_results=identity_results,
        identity_period_iso="2026-03-31",
    )
    # CALL1 debe haber sido excluido por _apply_5_3b.
    assert len(out) == 1
    stats = out.attrs.get("audit_stats")
    assert stats is not None
    assert stats["n_equity"] == 1
    assert stats["n_option"] == 1
    assert stats["n_unresolved"] == 0
    assert stats["n_dropped_by_instrument_type"] == 1


# --- Fix 2026-09-29: simetria de strip en claves de lookup ------------

def test_apply_5_3_cusip_con_espacios_encuentra_eligibilidad():
    """Bug fix 2026-09-29: _apply_5_3 usaba df['CUSIP'].astype(str)
    como clave de lookup, mientras resolve_eligibility produce claves
    str(cusip).strip(). Sin strip en el caller, CUSIP con espacios
    cae silenciosamente a NOT_IN_LIST."""
    df = pd.DataFrame({
        "CUSIP": [" 037833100 "],
        "TITLEOFCLASS": ["COM"],
        "SSHPRNAMTTYPE": ["SH"],
        "PUTCALL": [None],
    })
    # official_df con CUSIP limpio (sin espacios)
    official = pd.DataFrame({
        "cusip": ["037833100"],
        "option_indicator": [None],
        "issuer_name": ["APPLE INC"],
        "issuer_description": ["COM"],
        "status_raw": ["   "],
        "status": ["ACTIVE"],
        "raw_line": ["x" * 80],
        "_line_no": [1],
        "_order": [0],
    })
    out = ou._apply_5_3(df, official)
    # Con strip, el CUSIP con espacios se resuelve a ACTIVE (eligible).
    # Sin strip, caeria a NOT_IN_LIST y la fila se excluiria.
    assert len(out) == 1
    assert out.iloc[0]["_elig_status"] == "ACTIVE"


def test_apply_5_3b_cusip_con_espacios_encuentra_instrument_type():
    """Misma simetria que _apply_5_3 para _instrument_type."""
    df = pd.DataFrame({
        "CUSIP": [" 037833100 "],
        "TITLEOFCLASS": ["COM"],
        "SSHPRNAMTTYPE": ["SH"],
        "PUTCALL": [None],
    })
    official = pd.DataFrame({
        "cusip": ["037833100"],
        "option_indicator": [None],
        "issuer_name": ["APPLE INC"],
        "issuer_description": ["COM"],
        "status_raw": ["   "],
        "status": ["ACTIVE"],
        "raw_line": ["x" * 80],
        "_line_no": [1],
        "_order": [0],
    })
    out, stats = ou._apply_5_3b(df, official)
    # Con strip, el instrument_type es EQUITY (positivo). Sin strip,
    # seria UNKNOWN y la fila se mantendria (por la regla v2.3), pero
    # el stat n_equity no la contaria correctamente.
    assert stats["n_equity"] == 1
    assert stats["n_unresolved"] == 0


def test_apply_5_5_cusip_con_espacios_encuentra_identity():
    """Simetria con resolve_batch_identities (str(c).strip())."""
    df = pd.DataFrame({
        "CUSIP": [" 037833100 "],
        "TITLEOFCLASS": ["COM"],
        "SSHPRNAMTTYPE": ["SH"],
        "PUTCALL": [None],
    })
    identity_results = {
        "037833100": {
            "security_resolution_status": "CANONICAL",
            "operational_mapping_status": "VERIFIED",
            "canonical_security": "equity:AAPL",
            "canonical_security_kind": "CANONICAL_EQUIVALENCE",
            "observed_security_key": "cusip:037833100",
        }
    }
    out = ou._apply_5_5(df, identity_results)
    # Con strip, la fila se resuelve y se mantiene. Sin strip, _get
    # devolveria None y la fila se excluiria por mask.
    assert len(out) == 1
    assert out.iloc[0]["_canonical_security"] == "equity:AAPL"