# -*- coding: utf-8 -*-
"""Tests A2.3 (2026-09-15): get_instrument_class + regresion de get_market.

Cubre:
- Clasificacion de instrumento por clase economica (7 + UNKNOWN).
- Regresion: la semantica de get_market NO cambia.
"""
from src.instrument_registry import get_market, get_instrument_class


# --- get_instrument_class ---

def test_class_equity_usa():
    for t in ("AAPL", "SPY", "BRK-B", "XLE", "IVV"):
        assert get_instrument_class(t) == "EQUITY", t


def test_class_equity_lse():
    assert get_instrument_class("HSBA.L") == "EQUITY"
    assert get_instrument_class("SHEL.L") == "EQUITY"


def test_class_equity_xetra():
    assert get_instrument_class("SAP.DE") == "EQUITY"


def test_class_equity_bme():
    assert get_instrument_class("SAN.MC") == "EQUITY"


def test_class_equity_euronext():
    assert get_instrument_class("AIR.PA") == "EQUITY"
    assert get_instrument_class("ASML.AS") == "EQUITY"
    assert get_instrument_class("UCG.MI") == "EQUITY"


def test_class_index_usa():
    for t in ("^GSPC", "^DJI", "^NDX", "^RUT"):
        assert get_instrument_class(t) == "INDEX", t


def test_class_index_eu():
    for t in ("^FTSE", "^GDAXI", "^IBEX", "^STOXX50E"):
        assert get_instrument_class(t) == "INDEX", t


def test_class_index_commodity():
    assert get_instrument_class("^SPGSCI") == "INDEX"


def test_class_index_dxy():
    assert get_instrument_class("DX-Y.NYB") == "INDEX"


def test_class_volatility_index():
    for t in ("^VIX", "^VIX3M", "^VXN"):
        assert get_instrument_class(t) == "VOLATILITY_INDEX", t


def test_class_rate_yield():
    for t in ("^TNX", "^FVX"):
        assert get_instrument_class(t) == "RATE_YIELD", t


def test_class_future():
    for t in ("CL=F", "BZ=F", "NG=F", "GC=F", "HG=F"):
        assert get_instrument_class(t) == "FUTURE", t


def test_class_fx():
    for t in ("EURUSD=X", "USDJPY=X", "USDCNY=X"):
        assert get_instrument_class(t) == "FX", t


def test_class_unknown_suffix():
    assert get_instrument_class("XYZ.ZZ") == "UNKNOWN"


def test_class_not_string():
    assert get_instrument_class(None) == "UNKNOWN"
    assert get_instrument_class(42) == "UNKNOWN"
    assert get_instrument_class([]) == "UNKNOWN"


# --- get_market: regresion (semantica conservada) ---

def test_market_unchanged_us_equity():
    for t in ("AAPL", "SPY", "BRK-B"):
        assert get_market(t) == "US_EQUITY", t


def test_market_unchanged_european():
    assert get_market("HSBA.L") == "LSE"
    assert get_market("SAP.DE") == "XETRA"
    assert get_market("SAN.MC") == "BME"
    assert get_market("AIR.PA") == "EURONEXT"


def test_market_unchanged_no_equity_classification():
    """get_market NO cambia para no-equity: conserva US_EQUITY/UNKNOWN.

    A2.3 no debe romper el contrato actual de get_market. La nueva
    clasificacion vive en get_instrument_class.
    """
    for t in ("CL=F", "BZ=F", "NG=F", "GC=F", "HG=F"):
        assert get_market(t) == "US_EQUITY", t
    for t in ("EURUSD=X", "USDJPY=X", "USDCNY=X"):
        assert get_market(t) == "US_EQUITY", t
    for t in ("^GSPC", "^VIX", "^TNX", "^FTSE"):
        assert get_market(t) == "US_EQUITY", t
    # F3-19 / A2.3-parcial (2026-09-28): DX-Y.NYB tiene override
    # explicito a US_EQUITY (calendario NYSE). Ver mas abajo.
    assert get_market("DX-Y.NYB") == "US_EQUITY"


def test_market_unchanged_unknown_suffix():
    assert get_market("XYZ.ZZ") == "UNKNOWN"
    assert get_market(None) == "UNKNOWN"


# --- F3-19 / A2.3-parcial (2026-09-28) ---

def test_get_market_dxy_es_us_equity():
    """DX-Y.NYB (ICE US Dollar Index) opera con calendario NYSE.

    Sin el override caia a UNKNOWN por tener punto no-europeo, y
    _compute_by_market (Punto 5 FU-002-bymarket) lo marcaba INVALID,
    inhabilitando la exencion condicional del guard.
    """
    from src.instrument_registry import get_market
    assert get_market("DX-Y.NYB") == "US_EQUITY"


def test_get_market_punto_no_europeo_sigue_siendo_unknown():
    """El fallback UNKNOWN sigue vigente para tickers no reconocidos
    con punto no-europeo. El override de DX-Y.NYB no debe generalizarse.
    """
    from src.instrument_registry import get_market
    assert get_market("FAKE.NYSE") == "UNKNOWN"
    assert get_market("ZZZ.BOGUS") == "UNKNOWN"


def test_get_instrument_class_dxy_sin_cambios():
    """F3-19 no toca la clase economica de DX-Y.NYB (sigue INDEX)."""
    from src.instrument_registry import get_instrument_class
    assert get_instrument_class("DX-Y.NYB") == "INDEX"
