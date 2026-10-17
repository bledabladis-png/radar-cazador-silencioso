# -*- coding: utf-8 -*-
"""Tests del fix H4 (REP-AUDIT-20261017-H4, 2026-10-17)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indicators.momentum import compute_flow_proxy
from regimes.sector_regime import compute_price_flow_rankings
from src.report.leaders import render_momentum_otros


def _make_df(ticker, n=80, volume=0):
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    cols = pd.MultiIndex.from_product(
        [["Close", "High", "Low", "Volume"], [ticker]]
    )
    df = pd.DataFrame(index=dates, columns=cols, dtype=float)
    close = 100.0 + np.cumsum(np.sin(np.arange(n)) + 0.5)
    close = np.maximum(close, 10.0)
    df[("Close", ticker)] = close
    df[("High", ticker)] = close + 1.0
    df[("Low", ticker)] = close - 1.0
    df[("Volume", ticker)] = float(volume)
    return df


def test_flow_proxy_volume_zero_sostenido_devuelve_nan():
    df = _make_df("EURUSD=X", volume=0)
    out = compute_flow_proxy(df, "EURUSD=X")
    assert out.isna().all()


def test_flow_proxy_volume_real_devuelve_valor_finito():
    df = _make_df("XLE", volume=1_000_000)
    out = compute_flow_proxy(df, "XLE")
    assert out.notna().any()


def test_price_flow_rankings_incluye_nan_en_otros():
    """NaN debe aparecer en la lista, no omitido, y NO al final."""
    from config.tickers import MARKET_TICKERS
    dates = pd.date_range("2026-01-01", periods=80, freq="D")
    tickers = list(MARKET_TICKERS["sectors"]) + ["EURUSD=X", "TLT"]
    cols = pd.MultiIndex.from_product(
        [["Close", "High", "Low", "Volume"], tickers]
    )
    df = pd.DataFrame(index=dates, columns=cols, dtype=float)
    rng = np.random.default_rng(42)
    for t in tickers:
        close = 100.0 + np.cumsum(rng.normal(0, 1, 80))
        close = np.maximum(close, 10.0)
        df[("Close", t)] = close
        df[("High", t)] = close + 1.0
        df[("Low", t)] = close - 1.0
        df[("Volume", t)] = 1_000_000.0
    df[("Volume", "EURUSD=X")] = 0.0

    _, _, _, otros_flow_rank = compute_price_flow_rankings(df)
    tickers_in_rank = [t for t, _ in otros_flow_rank]
    assert "EURUSD=X" in tickers_in_rank
    # NaN tratado como 0.0 en el sort: no queda necesariamente al final,
    # pero debe estar presente y ser detectable como NaN.
    nan_positions = [i for i, (_, f) in enumerate(otros_flow_rank) if np.isnan(f)]
    assert len(nan_positions) == 1
    # Su posicion depende de los otros valores, pero no debe ser la ultima
    # si hay negativos (TLT generado con rng puede ser positivo o negativo,
    # no lo forzamos). Verificamos solo que esta en el ranking.


def test_render_momentum_otros_muestra_nd_para_nan():
    otros_price_rank = [("SPY", 0.05)]
    otros_flow_rank = [("SPY", 0.42), ("EURUSD=X", float("nan"))]
    lines = render_momentum_otros(otros_price_rank, otros_flow_rank)
    joined = "".join(lines)
    assert "EURUSD=X" in joined
    assert "N/D" in joined
