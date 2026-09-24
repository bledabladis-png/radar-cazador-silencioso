# -*- coding: utf-8 -*-
"""Tests del bug latente A/D (Commit A 2026-09-24).

compute_sector_breadth calculaba daily_ret = close.iloc[-1] - close.iloc[-2]
sin verificar que la penultima observacion fuera la sesion inmediatamente
anterior. Con el gap del 22-Sep (Yahoo incompleto), 206/313 tickers
contribuian al A/D con un movimiento de 2d etiquetado como 1d.

Fix: exigir continuidad temporal. Si hay gap interno, el ticker no
contribuye al A/D.
"""

import pandas as pd

from indicators.sector_breadth import compute_sector_breadth


def _make_df_market(sector_etf="XLK"):
    """df_market minimo con una columna Close para el ETF sector."""
    idx = pd.date_range("2026-09-14", "2026-09-24", freq="B")
    return pd.DataFrame({("Close", sector_etf): range(len(idx))}, index=idx)


def _make_df_stocks(ticker, dates, prices):
    """df_stocks con una columna Close para un ticker."""
    return pd.DataFrame({("Close", ticker): prices}, index=dates)


def _make_holdings(ticker, etf="XLK"):
    return pd.DataFrame({"etf": [etf], "ticker": [ticker], "weight": [1.0]})


# ---------- Tests ----------
def test_ad_cuenta_con_continuidad():
    """Sin gap: el ticker contribuye al A/D."""
    dates = pd.DatetimeIndex(["2026-09-21", "2026-09-22", "2026-09-23"])
    # Precio sube -> advance
    df_stocks = _make_df_stocks("AAPL", dates, [100.0, 101.0, 102.0])
    df_market = _make_df_market("XLK")
    holdings = _make_holdings("AAPL", "XLK")

    result = compute_sector_breadth(
        df_market, df_stocks, holdings, as_of_date=dates[-1]
    )
    row = result.iloc[0]
    assert row["advances"] == 1
    assert row["declines"] == 0
    assert row["n_valid_ad"] == 1


def test_ad_no_cuenta_si_hay_gap_interno():
    """Gap: 21-Sep -> 23-Sep (falta 22). NO contribuye al A/D."""
    # Ticker solo tiene datos el 21 y el 23 (falta el 22)
    dates = pd.DatetimeIndex(["2026-09-21", "2026-09-23"])
    df_stocks = _make_df_stocks("AAPL", dates, [100.0, 102.0])
    df_market = _make_df_market("XLK")
    holdings = _make_holdings("AAPL", "XLK")

    result = compute_sector_breadth(
        df_market, df_stocks, holdings, as_of_date=dates[-1]
    )
    row = result.iloc[0]
    # No debe contar ni advance ni decline
    assert row["advances"] == 0
    assert row["declines"] == 0
    assert row["unchanged"] == 0
    assert row["n_valid_ad"] == 0


def test_ad_mezcla_tickers_con_y_sin_gap():
    """Mezcla: AAPL con continuidad, MSFT con gap. Solo AAPL contribuye."""
    dates_full = pd.DatetimeIndex(["2026-09-21", "2026-09-22", "2026-09-23"])

    df_stocks = pd.DataFrame({
        ("Close", "AAPL"): [100.0, 101.0, 102.0],
        ("Close", "MSFT"): [200.0, None, 195.0],
    }, index=dates_full)
    # Para MSFT, iloc[-1] es el 23, iloc[-2] tras dropna es el 21 -> gap

    df_market = _make_df_market("XLK")
    holdings = pd.DataFrame({
        "etf": ["XLK", "XLK"],
        "ticker": ["AAPL", "MSFT"],
        "weight": [1.0, 1.0],
    })

    result = compute_sector_breadth(
        df_market, df_stocks, holdings, as_of_date=dates_full[-1]
    )
    row = result.iloc[0]
    # Solo AAPL contribuye: advance (102 > 101)
    assert row["advances"] == 1
    assert row["declines"] == 0
    assert row["n_valid_ad"] == 1


def test_ad_no_cuenta_con_menos_de_2_obs():
    """Con <2 observaciones no contribuye."""
    dates = pd.DatetimeIndex(["2026-09-23"])
    df_stocks = _make_df_stocks("AAPL", dates, [100.0])
    df_market = _make_df_market("XLK")
    holdings = _make_holdings("AAPL", "XLK")

    result = compute_sector_breadth(
        df_market, df_stocks, holdings, as_of_date=dates[-1]
    )
    row = result.iloc[0]
    assert row["advances"] == 0
    assert row["declines"] == 0
    assert row["n_valid_ad"] == 0