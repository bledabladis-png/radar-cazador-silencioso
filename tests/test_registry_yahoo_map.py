# -*- coding: utf-8 -*-
"""Tests consolidacion YAHOO_TICKER_MAP + get_market normaliza.

Commit 1: mapa movido a instrument_registry. Re-exports backward-compat
en stock_data_loader y data_loader.

Commit 2: get_market normaliza primero. BRK.B/BF.B caian a UNKNOWN.
"""

from src.instrument_registry import (
    get_market,
    normalize_yahoo_ticker,
    YAHOO_TICKER_MAP,
)


# ---------- Commit 1: consolidacion ----------
def test_mapa_registry_disponible():
    """El mapa vive en instrument_registry (unica fuente)."""
    from src.instrument_registry import YAHOO_TICKER_MAP as M
    assert YAHOO_TICKER_MAP is M
    assert "BRK.B" in YAHOO_TICKER_MAP


def test_normalize_backward_compat():
    """Los imports historicos siguen funcionando."""
    from src.stock_data_loader import normalize_yahoo_ticker as n1
    from src.data_loader import normalize_yahoo_ticker as n2
    assert n1("BRK.B") == "BRK-B"
    assert n2("BRK.B") == "BRK-B"
    assert n1 is n2  # misma referencia


def test_normalize_casos_mapa():
    assert normalize_yahoo_ticker("BRK.B") == "BRK-B"
    assert normalize_yahoo_ticker("BF.B") == "BF-B"
    assert normalize_yahoo_ticker("MOGA") == "MOG-A"
    assert normalize_yahoo_ticker("MOG A") == "MOG-A"
    assert normalize_yahoo_ticker("GEF B") == "GEF-B"
    assert normalize_yahoo_ticker("CRD A") == "CRD-A"
    assert normalize_yahoo_ticker("BH A") == "BH-A"


def test_normalize_sin_match_devuelve_input():
    assert normalize_yahoo_ticker("AAPL") == "AAPL"
    assert normalize_yahoo_ticker("XYZ.L") == "XYZ.L"


# ---------- Commit 2: get_market normaliza ----------
def test_get_market_brk_punto():
    """Bug 2026-09-24: BRK.B caia a UNKNOWN."""
    assert get_market("BRK.B") == "US_EQUITY"


def test_get_market_bf_punto():
    assert get_market("BF.B") == "US_EQUITY"


def test_get_market_brk_guion_regresion():
    """Variante con guion debe seguir funcionando."""
    assert get_market("BRK-B") == "US_EQUITY"
    assert get_market("BF-B") == "US_EQUITY"


def test_get_market_moga():
    """MOGA mapea a MOG-A (guion). Sin punto -> US_EQUITY."""
    assert get_market("MOGA") == "US_EQUITY"
    assert get_market("MOG-A") == "US_EQUITY"


def test_get_market_sufijos_europeos_no_afectados():
    """Los sufijos europeos siguen clasificando bien."""
    assert get_market("ABC.L") == "LSE"
    assert get_market("XYZ.DE") == "XETRA"
    assert get_market("ABC.MC") == "BME"
    assert get_market("ABC.PA") == "EURONEXT"