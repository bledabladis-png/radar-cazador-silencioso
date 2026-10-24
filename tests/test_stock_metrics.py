# -*- coding: utf-8 -*-
"""Tests de indicators.stock_metrics.

Cubren el contrato extraido de stock_leader.py en C4.1:
- compute_stock_metrics: metricas por ticker.
- compute_wls: agregacion por sector.
- compute_all_stock_metrics: aplicacion a todos los holdings.
"""
import numpy as np
import pandas as pd

from indicators.stock_metrics import (
    WLS_COMPONENTS,
    _validate_wls_inputs,
    compute_all_stock_metrics,
    compute_stock_metrics,
    compute_wls,
)


def _make_stock_df(tickers, n=400, start="2024-01-01"):
    dates = pd.date_range(start, periods=n, freq="B")
    rng = np.random.RandomState(42)
    data = {}
    for t in tickers:
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
        data[("Open", t)] = close * 0.999
        data[("High", t)] = close * 1.01
        data[("Low", t)] = close * 0.99
        data[("Close", t)] = close
        data[("Volume", t)] = rng.randint(1_000_000, 5_000_000, n)
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _make_market_df(ticker, n=400, start="2024-01-01"):
    dates = pd.date_range(start, periods=n, freq="B")
    close = 400 + np.arange(n) * 0.1
    data = {}
    data[("Open", ticker)] = close * 0.999
    data[("High", ticker)] = close * 1.005
    data[("Low", ticker)] = close * 0.995
    data[("Close", ticker)] = close
    data[("Volume", ticker)] = np.full(n, 5_000_000)
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _make_market_df_multi(tickers, n=400, start="2024-01-01"):
    dates = pd.date_range(start, periods=n, freq="B")
    data = {}
    for i, ticker in enumerate(tickers):
        close = 400 + np.arange(n) * 0.1 + i * 10
        data[("Open", ticker)] = close * 0.999
        data[("High", ticker)] = close * 1.005
        data[("Low", ticker)] = close * 0.995
        data[("Close", ticker)] = close
        data[("Volume", ticker)] = np.full(n, 5_000_000)
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_compute_stock_metrics_basico():
    df_stocks = _make_stock_df(["AAA", "BBB"])
    df_market = _make_market_df("XLK")
    out = compute_stock_metrics(df_market, df_stocks, "XLK", ["AAA", "BBB"])
    assert not out.empty
    assert set(out["ticker"]) == {"AAA", "BBB"}
    for col in ("rs", "rs_mom", "rs_ratio_20d", "flow_proxy_z",
                "wyckoff_score", "wyckoff_phase",
                "persistence_5d", "persistence_10d", "persistence_20d",
                "stability", "spring", "sos"):
        assert col in out.columns, col


def test_compute_stock_metrics_ticker_ausente_skip():
    df_stocks = _make_stock_df(["AAA"])
    df_market = _make_market_df("XLK")
    out = compute_stock_metrics(df_market, df_stocks, "XLK", ["AAA", "ZZZ"])
    assert not out.empty
    assert set(out["ticker"]) == {"AAA"}


def test_compute_stock_metrics_historico_corto_skip():
    df_stocks = _make_stock_df(["AAA"], n=30)
    df_market = _make_market_df("XLK")
    out = compute_stock_metrics(df_market, df_stocks, "XLK", ["AAA"])
    assert out.empty


def test_compute_wls_agrega_por_sector():
    df_metrics = pd.DataFrame([
        {"ticker": "AAA", "sector": "XLK", "rs_mom": 0.05,
         "flow_proxy_z": 0.3, "wyckoff_score": 0.4, "stability": 0.5,
         "persistence_10d": 0.7},
        {"ticker": "BBB", "sector": "XLK", "rs_mom": -0.02,
         "flow_proxy_z": 0.1, "wyckoff_score": 0.1, "stability": 0.3,
         "persistence_10d": 0.5},
        {"ticker": "CCC", "sector": "XLE", "rs_mom": 0.01,
         "flow_proxy_z": 0.2, "wyckoff_score": 0.2, "stability": 0.4,
         "persistence_10d": 0.6},
    ])
    out = compute_wls(df_metrics)
    assert "wls" in out.columns
    assert "wls_status" in out.columns
    assert "sector_rank_pct" in out.columns
    assert len(out) == 3


def test_compute_wls_vacio():
    out = compute_wls(pd.DataFrame())
    assert out.empty


def test_compute_all_stock_metrics_agrega_todos_los_sectores():
    tickers = ["AAPL", "MSFT", "XOM", "CVX"]
    df_stocks = _make_stock_df(tickers)
    df_market = _make_market_df_multi(["XLK", "XLE"])
    holdings = pd.DataFrame({
        "etf": ["XLK", "XLK", "XLE", "XLE"],
        "ticker": tickers,
        "weight": [50.0, 50.0, 50.0, 50.0],
    })
    out = compute_all_stock_metrics(df_market, df_stocks, holdings)
    assert not out.empty
    assert "sector" in out.columns
    assert "wls" in out.columns
    assert set(out["sector"]) == {"XLK", "XLE"}


def test_compute_all_stock_metrics_sin_holdings():
    out = compute_all_stock_metrics(pd.DataFrame(), pd.DataFrame(), None)
    assert out.empty
    out2 = compute_all_stock_metrics(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert out2.empty


def test_validate_wls_inputs_contrato():
    df = pd.DataFrame({
        "rs_z": [0.5, np.nan],
        "flow_proxy_z_norm": [0.3, 0.1],
        "rws_z": [0.2, 0.2],
        "stab_z": [0.1, 0.1],
        "persistence_10d": [1.5, 0.5],
    })
    pers, status, warns = _validate_wls_inputs(df)
    assert pers.iloc[0] == 1.0  # clipped
    assert status.iloc[0] == "PERSISTENCE_OUT_OF_RANGE"
    assert status.iloc[1] == "INSUFFICIENT_COMPONENTS"
    assert len(warns) == 1


def test_wls_components_contrato():
    assert WLS_COMPONENTS == ("rs_z", "flow_proxy_z_norm", "rws_z", "stab_z")
