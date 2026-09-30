# -*- coding: utf-8 -*-
"""D33 (2026-09-30): tests para modulos con cobertura baja:

- options_metrics: 7 funciones ratio (los clasificadores ya tienen test).
- rs_internal: compute_rs_internal + _ret_20d (classify_rs ya tiene).
- vol_metrics: compute_vol_metrics + _zscore_last_in_window.

options.py (compute_pcr_signals) NO se cubre: requiere CboeProvider
+ IO de pcr_history.csv. E2E-only.
"""
import numpy as np
import pandas as pd
import pytest

from indicators.options_metrics import (
    institutional_hedge_ratio, index_volume_share, put_share, call_share,
    volume_put_call_ratio, oi_put_call_ratio, relative_volume, oi_change,
)
from indicators.rs_internal import _ret_20d, compute_rs_internal
from indicators.vol_metrics import compute_vol_metrics
from indicators.options import _zscore_last_in_window


# ============================================================
# options_metrics: 8 funciones ratio (con guards de finitud)
# ============================================================
def test_ihr_normal():
    assert institutional_hedge_ratio(1.5, 0.5) == 3.0


def test_ihr_equity_cero():
    assert institutional_hedge_ratio(1.5, 0.0) is None


def test_ihr_equity_none():
    assert institutional_hedge_ratio(1.5, None) is None


def test_ihr_equity_inf():
    assert institutional_hedge_ratio(1.5, np.inf) is None


def test_ihr_index_nan():
    assert institutional_hedge_ratio(np.nan, 0.5) is None


def test_index_volume_share_normal():
    assert index_volume_share(50, 200) == 0.25


def test_index_volume_share_total_cero():
    assert index_volume_share(50, 0) is None


def test_index_volume_share_index_inf():
    assert index_volume_share(np.inf, 200) is None


def test_put_share_normal():
    assert put_share(30, 100) == 0.3


def test_put_share_total_negativo():
    assert put_share(30, -5) is None


def test_call_share_normal():
    assert call_share(70, 100) == 0.7


def test_call_share_guard_finitud():
    """Fix 2026-09-29: guard de finitud anadido para simetria."""
    assert call_share(np.inf, 100) is None
    assert call_share(70, np.inf) is None


def test_volume_put_call_ratio_normal():
    assert volume_put_call_ratio(80, 40) == 2.0


def test_volume_put_call_ratio_call_cero():
    assert volume_put_call_ratio(80, 0) is None


def test_oi_put_call_ratio_normal():
    assert oi_put_call_ratio(200, 100) == 2.0


def test_oi_put_call_ratio_call_cero():
    assert oi_put_call_ratio(200, 0) is None


def test_relative_volume_normal():
    assert relative_volume(150, 100) == 1.5


def test_relative_volume_avg_cero():
    assert relative_volume(150, 0) is None


def test_oi_change_normal():
    assert oi_change(120, 100) == pytest.approx(0.2)


def test_oi_change_yesterday_cero():
    assert oi_change(120, 0) is None


def test_oi_change_yesterday_inf():
    assert oi_change(120, np.inf) is None


# ============================================================
# rs_internal
# ============================================================
def _make_df(tickers, n=100, seed=42):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {}
    for i, t in enumerate(tickers):
        close = 100 * np.cumprod(1 + rng.normal(0.0005 + i * 0.0001, 0.015, n))
        data[("Close", t)] = close
        data[("High", t)] = close * 1.01
        data[("Low", t)] = close * 0.99
        data[("Open", t)] = close * 0.998
        data[("Volume", t)] = rng.randint(1_000_000, 5_000_000, n)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_ret_20d_corto():
    assert pd.isna(_ret_20d(pd.Series([100.0] * 20)))


def test_ret_20d_normal():
    assert _ret_20d(pd.Series([100.0] * 20 + [110.0])) == pytest.approx(0.10)


def test_compute_rs_internal_normal():
    df_stocks = _make_df(["AAPL", "MSFT"], n=100)
    df_market = _make_df(["XLK", "^GSPC"], n=100)
    holdings = pd.DataFrame({"etf": ["XLK", "XLK"], "ticker": ["AAPL", "MSFT"]})
    out = compute_rs_internal(df_stocks, holdings, df_market, benchmark="^GSPC")
    assert not out.empty
    assert "classification" in out.columns
    assert "rs_abs_20d" in out.columns


def test_compute_rs_internal_benchmark_fallback():
    """Si benchmark no esta en df_market -> fallback ^GSPC."""
    df_stocks = _make_df(["AAPL"], n=100)
    df_market = _make_df(["XLK", "^GSPC"], n=100)
    holdings = pd.DataFrame({"etf": ["XLK"], "ticker": ["AAPL"]})
    # Benchmark NOEXISTE no esta; fallback a ^GSPC
    out = compute_rs_internal(df_stocks, holdings, df_market, benchmark="NOEXISTE")
    assert not out.empty


def test_compute_rs_internal_sector_ausente():
    """Sector no en df_market -> skip."""
    df_stocks = _make_df(["AAPL"], n=100)
    df_market = _make_df(["^GSPC"], n=100)
    holdings = pd.DataFrame({"etf": ["XLK"], "ticker": ["AAPL"]})
    out = compute_rs_internal(df_stocks, holdings, df_market)
    assert out.empty


def test_compute_rs_internal_ticker_ausente():
    """Ticker no en df_stocks -> skip."""
    df_stocks = _make_df(["MSFT"], n=100)
    df_market = _make_df(["XLK", "^GSPC"], n=100)
    holdings = pd.DataFrame({"etf": ["XLK"], "ticker": ["AAPL"]})
    out = compute_rs_internal(df_stocks, holdings, df_market)
    assert out.empty


def test_compute_rs_internal_serie_corta():
    """Ticker con <21 filas -> skip."""
    df_stocks = _make_df(["AAPL"], n=15)
    df_market = _make_df(["XLK", "^GSPC"], n=100)
    holdings = pd.DataFrame({"etf": ["XLK"], "ticker": ["AAPL"]})
    out = compute_rs_internal(df_stocks, holdings, df_market)
    assert out.empty


# ============================================================
# vol_metrics
# ============================================================
def test_vol_metrics_normal():
    df = _make_df(["SPY", "^VIX"], n=200)
    out = compute_vol_metrics(df)
    assert "rv_21d" in out
    assert "rv_60d" in out
    assert "vrp_21d" in out
    assert "vrp_60d" in out
    # Con 200 filas, al menos rv_21d debe estar
    assert out["rv_21d"] is not None


def test_vol_metrics_sin_vix():
    """Sin ^VIX -> KeyError -> diccionario con None."""
    df = _make_df(["SPY"], n=200)
    out = compute_vol_metrics(df)
    assert out == {"rv_21d": None, "rv_60d": None,
                    "vrp_21d": None, "vrp_60d": None}


def test_vol_metrics_sin_spy():
    df = _make_df(["^VIX"], n=200)
    out = compute_vol_metrics(df)
    assert out["rv_21d"] is None


# ============================================================
# options._zscore_last_in_window
# ============================================================
def test_zscore_last_in_window_normal():
    rng = np.random.RandomState(42)
    s = pd.Series(list(rng.randn(50)) + [5.0])
    z = _zscore_last_in_window(s)
    assert z > 1.0


def test_zscore_last_in_window_mad_cero():
    s = pd.Series([5.0] * 50)
    assert _zscore_last_in_window(s) == 0.0
