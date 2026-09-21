"""Tests Nivel B - security_type (spec 3.3 + dictamen #61)."""
from __future__ import annotations

from src.institutional_accumulation import security_type as st


def _cls(toc):
    return st.classify_title_of_class(toc)


# --- Enums ---

def test_security_type_enum():
    assert st.ALL_SECURITY_TYPES == (
        "EQUITY", "NON_EQUITY", "ETF", "UNKNOWN",
    )


def test_status_enum():
    assert st.ALL_SECURITY_TYPE_STATUSES == (
        "RESOLVED_EQUITY", "RESOLVED_NON_EQUITY",
        "UNRESOLVED", "CONFLICT",
    )


def test_source_enum():
    assert st.ALL_SECURITY_TYPE_SOURCES == (
        "TITLEOFCLASS_13F", "OPENFIGI_METADATA",
        "INTERNAL_CURATED", "COMBINED",
    )

# --- E-1..E-6: EQUITY ---

def test_E1_COM_resolved_equity():
    assert _cls("COM") == ("EQUITY", "RESOLVED_EQUITY")


def test_E2_common_stock_resolved_equity():
    assert _cls("COMMON STOCK") == ("EQUITY", "RESOLVED_EQUITY")


def test_E3_com_cl_a_resolved_equity():
    assert _cls("COM CL A") == ("EQUITY", "RESOLVED_EQUITY")


def test_E4_cl_a_resolved_equity_segun_d62():
    """Dictamen #62: CL A autorizado como candidato de alta confianza."""
    assert _cls("CL A") == ("EQUITY", "RESOLVED_EQUITY")


def test_E5_shs_resolved_equity_segun_d62():
    assert _cls("SHS") == ("EQUITY", "RESOLVED_EQUITY")


def test_E6_sponsored_adr_unresolved():
    assert _cls("SPONSORED ADR") == ("UNKNOWN", "UNRESOLVED")

# --- E-7..E-10: NON_EQUITY ---

def test_E7_convertible_bond_non_equity():
    assert _cls("CONVERTIBLE BOND") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")


def test_E8_convertible_com_non_equity_gana():
    """NON_EQUITY tiene precedencia sobre COM."""
    assert _cls("CONVERTIBLE COM") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")


def test_E9_etf_non_equity():
    assert _cls("ETF") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")


def test_E10_mutual_funds_equities_non_equity():
    assert _cls("MUTUAL FUNDS-EQUITIES") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")

# --- E-11..E-16: ambiguedad, fallback, precedencia ---

def test_E11_notebook_inc_unresolved():
    assert _cls("NOTEBOOK INC") == ("UNKNOWN", "UNRESOLVED")


def test_E12_unknown_token_unresolved():
    assert _cls("UNKNOWN") == ("UNKNOWN", "UNRESOLVED")


def test_E14_sin_marcadores_unresolved():
    assert _cls("XYZ ABC") == ("UNKNOWN", "UNRESOLVED")


def test_E15_fuerte_gana_a_ambiguo():
    """COM NEW: COM es anchor fuerte, NEW seria ambiguo."""
    assert _cls("COM NEW") == ("EQUITY", "RESOLVED_EQUITY")


def test_E16_non_equity_gana_a_equity_substring():
    """PFD COM: PFD domina por precedencia."""
    assert _cls("PFD COM") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")

# --- Tokenizacion / fronteras ---

def test_tokenizacion_punto_guion_barra():
    for toc in ("MUTUAL-FUNDS", "MUTUAL.FUNDS", "MUTUAL/FUNDS"):
        assert _cls(toc) == ("NON_EQUITY", "RESOLVED_NON_EQUITY")


def test_w_exp_phrase():
    assert _cls("W EXP 08/03/2027") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")


def test_note_como_token_no_notebook():
    assert _cls("NOTE") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")
    assert _cls("NOTE 3.500% 6/1") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")
    assert _cls("NOTEBOOK") == ("UNKNOWN", "UNRESOLVED")


def test_warrant_token():
    assert _cls("WARRANT") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")


def test_rights_token():
    assert _cls("RIGHT 04/02/2026") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")


def test_unit_token():
    assert _cls("UNIT 01/06/2031") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")


def test_pfd_preferred():
    assert _cls("PFD") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")
    assert _cls("PFD SER B") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")
    assert _cls("PREFERRED") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")
    assert _cls("PREFERRED STOCK") == ("NON_EQUITY", "RESOLVED_NON_EQUITY")

def test_not_aprobados_no_marcan_equity():
    """Dictamen #61: CL A, SHS, STOCK, EQUITY, ADR, REIT, NEW -> UNRESOLVED."""
    for toc in ("CL C", "CLASS A", "CLASS B",
                "SHARES", "ADR", "ADS",
                "SPONSORED ADR", "SPONSORED ADS",
                "REIT", "NEW"):
        assert _cls(toc) == ("UNKNOWN", "UNRESOLVED"), toc


def test_gestores_no_son_autoridad():
    """Dictamen #61 4.3: marcas de gestor no clasifican."""
    for toc in ("ISHARES", "SPDR", "VANGUARD", "INVESCO", "SCHWAB"):
        assert _cls(toc) == ("UNKNOWN", "UNRESOLVED"), toc


def test_BDC_no_automatico():
    assert _cls("BDC") == ("UNKNOWN", "UNRESOLVED")


def test_vacios_unresolved():
    for toc in (None, "", "   ", "NAN", "<NA>", "(BLANK)"):
        assert _cls(toc) == ("UNKNOWN", "UNRESOLVED")


def test_case_insensitive():
    for toc in ("com", "Com", "COM"):
        assert _cls(toc) == ("EQUITY", "RESOLVED_EQUITY")

# --- resolve_security_type (evidence externa) ---

def test_resolve_sin_evidencia_externa():
    assert st.resolve_security_type("COM") == ("EQUITY", "RESOLVED_EQUITY")


def test_resolve_ext_anula_title_unresolved():
    # CLASS A sigue UNRESOLVED tras dictamen #62.
    r = st.resolve_security_type("CLASS A", {"security_type": "EQUITY"})
    assert r == ("EQUITY", "RESOLVED_EQUITY")


def test_resolve_ext_anula_title_unresolved_non_equity():
    r = st.resolve_security_type("CLASS A", {"security_type": "NON_EQUITY"})
    assert r == ("NON_EQUITY", "RESOLVED_NON_EQUITY")


def test_resolve_coincide():
    r = st.resolve_security_type("COM", {"security_type": "EQUITY"})
    assert r == ("EQUITY", "RESOLVED_EQUITY")


def test_resolve_E13_conflicto_real():
    """E-13: equity + bond contradicen -> CONFLICT."""
    r = st.resolve_security_type("COM", {"security_type": "NON_EQUITY"})
    assert r == ("UNKNOWN", "CONFLICT")


def test_resolve_ext_unknown_no_cambia():
    r = st.resolve_security_type("COM", {"security_type": "UNKNOWN"})
    assert r == ("EQUITY", "RESOLVED_EQUITY")


def test_resolve_ext_invalido_no_cambia():
    r = st.resolve_security_type("COM", {"security_type": "FOO"})
    assert r == ("EQUITY", "RESOLVED_EQUITY")

# --- coverage_stats ---

def test_coverage_stats_vacio():
    r = st.coverage_stats([])
    assert r["total"] == 0
    assert r["pct_unresolved"] == 0.0


def test_coverage_stats_mixto():
    records = [
        ("EQUITY", "RESOLVED_EQUITY"),
        ("EQUITY", "RESOLVED_EQUITY"),
        ("NON_EQUITY", "RESOLVED_NON_EQUITY"),
        ("UNKNOWN", "UNRESOLVED"),
        ("UNKNOWN", "CONFLICT"),
    ]
    r = st.coverage_stats(records)
    assert r["total"] == 5
    assert r["counts"]["RESOLVED_EQUITY"] == 2
    assert r["counts"]["RESOLVED_NON_EQUITY"] == 1
    assert r["counts"]["UNRESOLVED"] == 1
    assert r["counts"]["CONFLICT"] == 1
    assert r["pct_operational"] == 40.0
    assert r["pct_unresolved"] == 20.0
    assert r["pct_conflict"] == 20.0


def test_coverage_stats_admite_dicts():
    records = [
        {"security_type_status": "RESOLVED_EQUITY"},
        {"security_type_status": "UNRESOLVED"},
    ]
    r = st.coverage_stats(records)
    assert r["total"] == 2
    assert r["pct_operational"] == 50.0

# --- Dictamen #63: EQUITY/STOCK exactos ---

def test_equity_exacto():
    assert _cls("EQUITY") == ("EQUITY", "RESOLVED_EQUITY")


def test_stock_exacto():
    assert _cls("STOCK") == ("EQUITY", "RESOLVED_EQUITY")


def test_stock_como_substring_no_clasifica():
    """Dictamen #63: STOCK exacto, no substring."""
    for toc in ("STOCK FUND", "STOCK INDEX FUND", "GROWTH STOCK"):
        assert _cls(toc) == ("UNKNOWN", "UNRESOLVED"), toc


def test_equity_como_substring_no_clasifica():
    for toc in ("EQUITY FUND", "EQUITY INCOME"):
        assert _cls(toc) == ("UNKNOWN", "UNRESOLVED"), toc


# --- Dictamen #63: handoff §5.4 -> §5.5 ---

def test_handoff_resolved_equity_entra():
    assert st.is_operational_candidate("EQUITY", "RESOLVED_EQUITY") is True


def test_handoff_non_equity_excluido():
    assert st.is_operational_candidate("NON_EQUITY", "RESOLVED_NON_EQUITY") is False


def test_handoff_unresolved_excluido():
    assert st.is_operational_candidate("UNKNOWN", "UNRESOLVED") is False


def test_handoff_conflict_excluido():
    assert st.is_operational_candidate("UNKNOWN", "CONFLICT") is False


def test_handoff_valor_inesperado_fail_closed():
    """Fail-closed: cualquier valor raro -> False."""
    assert st.is_operational_candidate("FOO", "RESOLVED_EQUITY") is False
    assert st.is_operational_candidate("EQUITY", "FOO") is False
    assert st.is_operational_candidate(None, None) is False


def test_operational_universe_candidate_com():
    assert st.operational_universe_candidate("COM") is True


def test_operational_universe_candidate_convertible():
    assert st.operational_universe_candidate("CONVERTIBLE BOND") is False


def test_operational_universe_candidate_cl_a():
    assert st.operational_universe_candidate("CL A") is True


def test_operational_universe_candidate_adr():
    """Dictamen #63: ADR sigue UNRESOLVED, excluido del operational."""
    assert st.operational_universe_candidate("ADR") is False
    assert st.operational_universe_candidate("SPONSORED ADR") is False


def test_operational_universe_candidate_fund():
    assert st.operational_universe_candidate("FUND") is False


def test_operational_universe_candidate_class_a():
    """CLASS A no autorizado -> UNRESOLVED -> excluido."""
    assert st.operational_universe_candidate("CLASS A") is False
