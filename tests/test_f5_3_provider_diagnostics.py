"""F5-3 (2026-09-28): diagnostico stale/failed en los 3 providers EU.

Verifica la API publica get_last_stale_tickers / get_last_failed_tickers
y su reset por llamada a get_prices.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_xetra_diagnostics_vacios_al_inicio():
    from data.providers.xetra_provider import XetraProvider
    p = XetraProvider()
    assert p.get_last_stale_tickers() == []
    assert p.get_last_failed_tickers() == []


def test_euronext_diagnostics_vacios_al_inicio():
    from data.providers.euronext_provider import EuronextProvider
    p = EuronextProvider()
    assert p.get_last_stale_tickers() == []
    assert p.get_last_failed_tickers() == []


def test_bme_diagnostics_vacios_al_inicio():
    from data.providers.bme_provider import BMEProvider
    p = BMEProvider()
    assert p.get_last_stale_tickers() == []
    assert p.get_last_failed_tickers() == []


def test_xetra_failed_por_ticker_no_soportado():
    from data.providers.xetra_provider import XetraProvider
    p = XetraProvider()
    df = p.get_prices(["FAKE_NOT_IN_MAP.DE"], use_cache=False)
    assert df.empty
    assert "FAKE_NOT_IN_MAP.DE" in p.get_last_failed_tickers()


def test_euronext_failed_por_ticker_no_soportado():
    from data.providers.euronext_provider import EuronextProvider
    p = EuronextProvider()
    df = p.get_prices(["FAKE_NOT_IN_MAP.PA"], use_cache=False)
    assert df.empty
    assert "FAKE_NOT_IN_MAP.PA" in p.get_last_failed_tickers()


def test_bme_failed_por_ticker_no_soportado():
    from data.providers.bme_provider import BMEProvider
    p = BMEProvider()
    df = p.get_prices(["FAKE_NOT_IN_MAP.MC"], use_cache=False)
    assert df.empty
    assert "FAKE_NOT_IN_MAP.MC" in p.get_last_failed_tickers()


def test_reset_entre_llamadas():
    """El diagnostico se resetea en cada llamada a get_prices."""
    from data.providers.xetra_provider import XetraProvider
    p = XetraProvider()
    p.get_prices(["FAKE1.DE"], use_cache=False)
    assert "FAKE1.DE" in p.get_last_failed_tickers()
    p.get_prices([], use_cache=False)
    assert p.get_last_failed_tickers() == []
    assert p.get_last_stale_tickers() == []