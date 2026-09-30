# -*- coding: utf-8 -*-
"""Fix M: cache-hit debe validar cobertura, no solo fecha.

Bug: stock_data_loader.py:536 y data_loader.py:277 deciden cache-hit
comprobando solo `_df_last >= _last_exp`. Si el parquet tiene la fila
del 30-sep pero con cobertura 89.8% (32 tickers europeos con NaN), la
acepta como valida. No ejecuta el cascade. compute_leaders recibe el
df incompleto y trunca por min_coverage=0.90.

Verificado 2026-09-30: stock_prices.parquet con coverage_pct_last=0.8978
(manifest: VALID_WITH_MISSING). El run del 01-oct no disparo la
descarga. Cache Euronext y BME siguen en 29-sep.

Fix: si la cobertura de la ultima fila < MIN_COVERAGE_CACHE_GATE,
forzar descarga. Fail-closed: si no se puede determinar cobertura,
descargar (mas caro pero seguro).
"""
import pandas as pd
import pytest


def _make_df(n_tickers: int, n_valid: int, last_date: str = "2026-09-30"):
    """df MultiIndex (campo, ticker) con n_valid tickers con Close."""
    import numpy as np
    idx = pd.date_range(end=last_date, periods=5, freq="B")
    cols = []
    for i in range(n_tickers):
        t = f"T{i:03d}"
        for campo in ["Open", "High", "Low", "Close", "Volume"]:
            cols.append((campo, t))
    data = {c: [1.0] * 5 for c in cols}
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    # Invalidar (n_tickers - n_valid) tickers
    for i in range(n_valid, n_tickers):
        t = f"T{i:03d}"
        df[("Close", t)] = float("nan")
    return df


def test_coverage_below_threshold_forces_download():
    """df con 89.8% cobertura -> False (forzar descarga)."""
    from src.stock_data_loader import _last_row_coverage_ok
    df = _make_df(n_tickers=100, n_valid=89)
    assert _last_row_coverage_ok(df, min_coverage=0.90) is False


def test_coverage_above_threshold_accepts_cache():
    """df con 100% cobertura -> True (cache valida)."""
    from src.stock_data_loader import _last_row_coverage_ok
    df = _make_df(n_tickers=100, n_valid=100)
    assert _last_row_coverage_ok(df, min_coverage=0.90) is True


def test_coverage_exact_threshold_accepts():
    """df con exactamente 90% -> True (>= umbral)."""
    from src.stock_data_loader import _last_row_coverage_ok
    df = _make_df(n_tickers=100, n_valid=90)
    assert _last_row_coverage_ok(df, min_coverage=0.90) is True


def test_coverage_df_sin_close_returns_false():
    """df sin columnas Close -> False (fail-closed)."""
    from src.stock_data_loader import _last_row_coverage_ok
    idx = pd.date_range(end="2026-09-30", periods=3, freq="B")
    df = pd.DataFrame({"x": [1, 2, 3]}, index=idx)
    assert _last_row_coverage_ok(df, min_coverage=0.90) is False


def test_coverage_df_vacio_returns_false():
    """df vacio -> False."""
    from src.stock_data_loader import _last_row_coverage_ok
    assert _last_row_coverage_ok(pd.DataFrame(), min_coverage=0.90) is False


def test_coverage_df_none_returns_false():
    """df None -> False."""
    from src.stock_data_loader import _last_row_coverage_ok
    assert _last_row_coverage_ok(None, min_coverage=0.90) is False