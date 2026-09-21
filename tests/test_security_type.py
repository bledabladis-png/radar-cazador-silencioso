"""Tests Nivel A - security_type (spec 3.3)."""
from __future__ import annotations

from src.institutional_accumulation import security_type as st


def test_security_type_enum():
    assert st.ALL_SECURITY_TYPES == (
        "EQUITY",
        "NON_EQUITY",
        "ETF",
        "UNKNOWN",
    )


def test_status_enum():
    assert st.ALL_SECURITY_TYPE_STATUSES == (
        "RESOLVED_EQUITY",
        "RESOLVED_NON_EQUITY",
        "UNRESOLVED",
        "CONFLICT",
    )


def test_source_enum():
    assert st.ALL_SECURITY_TYPE_SOURCES == (
        "TITLEOFCLASS_13F",
        "OPENFIGI_METADATA",
        "INTERNAL_CURATED",
        "COMBINED",
    )

def test_convertible_bond_explicito():
    """Spec 3.3 caso explicito (cubre GHISALLO)."""
    assert st.classify_title_of_class("CONVERTIBLE BOND") == (
        "NON_EQUITY", "RESOLVED_NON_EQUITY",
    )


def test_convertible_variantes():
    for toc in ("CONVERTIBLE", "CONVERTIBLE PREFERRED", "Convertible Bond"):
        assert st.classify_title_of_class(toc) == (
            "NON_EQUITY", "RESOLVED_NON_EQUITY",
        )


def test_note_bonos():
    for toc in ("NOTE", "NOTE 3.500% 6/1", "NOTE 0.500% 3/0"):
        assert st.classify_title_of_class(toc) == (
            "NON_EQUITY", "RESOLVED_NON_EQUITY",
        )


def test_note_no_confunde_notebook():
    assert st.classify_title_of_class("NOTEBOOK INC") != (
        "NON_EQUITY", "RESOLVED_NON_EQUITY",
    )

def test_pfd_preferreds():
    for toc in ("PFD", "PFD SER B", "6.25 CNV PFD C", "PREFERRED"):
        assert st.classify_title_of_class(toc) == (
            "NON_EQUITY", "RESOLVED_NON_EQUITY",
        )


def test_com_es_unresolved_nivel_a():
    """COM es equity pero Nivel A no clasifica EQUITY. Queda UNRESOLVED."""
    assert st.classify_title_of_class("COM") == (
        "UNKNOWN", "UNRESOLVED",
    )


def test_common_stock_unresolved_nivel_a():
    assert st.classify_title_of_class("COMMON STOCK") == (
        "UNKNOWN", "UNRESOLVED",
    )


def test_adr_unresolved_nivel_a():
    assert st.classify_title_of_class("SPONSORED ADR") == (
        "UNKNOWN", "UNRESOLVED",
    )

def test_warrant_unresolved_nivel_a():
    """Warrants son NON_EQUITY en spec pero Nivel A no los cubre."""
    assert st.classify_title_of_class("W EXP 08/03/2027") == (
        "UNKNOWN", "UNRESOLVED",
    )


def test_etf_unresolved_nivel_a():
    assert st.classify_title_of_class("ETF") == (
        "UNKNOWN", "UNRESOLVED",
    )


def test_vacio_unresolved():
    for toc in (None, "", "   ", "NAN", "<NA>", "(BLANK)"):
        assert st.classify_title_of_class(toc) == (
            "UNKNOWN", "UNRESOLVED",
        )


def test_case_insensitive():
    for toc in ("convertible bond", "Convertible Bond", "CONVERTIBLE BOND"):
        assert st.classify_title_of_class(toc) == (
            "NON_EQUITY", "RESOLVED_NON_EQUITY",
        )