# -*- coding: utf-8 -*-
"""Tests Commit C (2026-09-24): check_non_market_days multi-mercado.

Contexto: el universo del parquet es multi-mercado. Los europeos operan
en festivos USA. Sus datos son legitimos. Solo es anomalia si hay
tickers US_EQUITY con Close en fecha no bursatil NYSE.
"""

import pandas as pd

from scripts.health_check import (
    OK,
    WARN,
    check_non_market_days,
)


def _make_df_with_markets(markets_tickers, dates):
    """Construye df con columnas Close para los tickers indicados.

    markets_tickers: dict {market_name: [tickers]}
    """
    data = {}
    for market, tickers in markets_tickers.items():
        for t in tickers:
            data[("Close", t)] = [100.0] * len(dates)
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates))


def test_c_solo_europeos_en_festivo_usa_es_ok(monkeypatch):
    """Festivo NYSE con solo datos europeos -> OK (correcto)."""
    from src import instrument_registry as ir
    from src import market_calendar as mc

    # Mock: el 2026-05-25 (Memorial Day) no es bursatil NYSE
    monkeypatch.setattr(mc, "is_market_day",
                        lambda d: d.strftime("%Y-%m-%d") != "2026-05-25")
    monkeypatch.setattr(ir, "get_market",
                        lambda t: "EURONEXT" if t.startswith("EU") else "US_EQUITY")

    # Solo europeos en el festivo
    dates = ["2026-05-22", "2026-05-25"]  # viernes + lunes festivo
    df = pd.DataFrame({
        ("Close", "EU1"): [100.0, 101.0],
        ("Close", "EU2"): [100.0, 101.0],
    }, index=pd.DatetimeIndex(dates))

    results = check_non_market_days(df)
    assert len(results) == 1
    assert results[0].status == OK
    assert "solo datos europeos" in results[0].message


def test_c_usa_en_festivo_es_warn(monkeypatch):
    """Festivo NYSE con datos USA -> WARN (anomalia)."""
    from src import instrument_registry as ir
    from src import market_calendar as mc

    monkeypatch.setattr(mc, "is_market_day",
                        lambda d: d.strftime("%Y-%m-%d") != "2026-05-25")
    monkeypatch.setattr(ir, "get_market",
                        lambda t: "US_EQUITY")

    dates = ["2026-05-22", "2026-05-25"]
    df = pd.DataFrame({
        ("Close", "AAPL"): [100.0, 101.0],
        ("Close", "MSFT"): [100.0, 101.0],
    }, index=pd.DatetimeIndex(dates))

    results = check_non_market_days(df)
    assert len(results) == 1
    assert results[0].status == WARN
    assert "datos USA" in results[0].message
    assert "2 USA" in results[0].message


def test_c_sin_fechas_no_bursatiles_es_ok(monkeypatch):
    """Todas las fechas bursatiles -> OK."""
    from src import market_calendar as mc
    monkeypatch.setattr(mc, "is_market_day", lambda d: d.weekday() < 5)

    dates = ["2026-09-21", "2026-09-22", "2026-09-23"]
    df = pd.DataFrame({
        ("Close", "AAPL"): [100.0] * 3,
    }, index=pd.DatetimeIndex(dates))

    results = check_non_market_days(df)
    assert results[0].status == OK
    assert "todas bursatiles" in results[0].message