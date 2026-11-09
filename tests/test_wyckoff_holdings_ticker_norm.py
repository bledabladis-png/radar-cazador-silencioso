# -*- coding: utf-8 -*-
"""Centinela: _evaluate_holding normaliza el ticker de salida (S34).

Historia: 2026-10-30 (P1-H02) se corrigio que holdings trae BRK.B y el
parquet BRK-B: sin normalizar el lookup, la fila se descartaba. Pero el
output seguia publicando BRK.B. S34 publica ticker_norm, alineado con
IAE (d41c223e) y rs_internal (9d712c28).
"""
import pandas as pd

from indicators.wyckoff_holdings_report import _evaluate_holding


def _make_df_stocks(tickers, n_rows=500):
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    data = {}
    for field in ["Open", "High", "Low", "Close", "Volume"]:
        for t in tickers:
            data[(field, t)] = 100.0
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_evaluate_holding_normaliza_ticker_de_holdings():
    df_stocks = _make_df_stocks(["BRK-B", "XLF"])
    hrow = pd.Series({"ticker": "BRK.B", "ticker_norm": "BRK-B", "weight": 12.3})
    row = _evaluate_holding(
        df_stocks, "XLF", hrow, "sector_usa", False,
        pd.Timestamp("2025-03-01"), 90.0,
    )
    assert row is not None
    assert row["ticker"] == "BRK-B", (
        "output debe usar ticker_norm; obtuvo " + repr(row["ticker"])
    )
    assert row["ticker"] != "BRK.B"


def test_evaluate_holding_aapl_idempotente():
    df_stocks = _make_df_stocks(["AAPL", "XLK"])
    hrow = pd.Series({"ticker": "AAPL", "ticker_norm": "AAPL", "weight": 10.0})
    row = _evaluate_holding(
        df_stocks, "XLK", hrow, "sector_usa", False,
        pd.Timestamp("2025-03-01"), 90.0,
    )
    assert row is not None
    assert row["ticker"] == "AAPL"
