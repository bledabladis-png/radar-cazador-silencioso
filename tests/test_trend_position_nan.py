# -*- coding: utf-8 -*-
"""H8: trend_position debe propagar NaN cuando close es NaN.

Contrato: si `close` no tiene observacion, el componente debe ser
NaN, no -1.0. El valor -1.0 queda reservado para una senal bajista
efectivamente observada.

Verificado 2026-09-30: 1.454 filas historicas del parquet tienen
close NaN y trend_position devolvia -1.0.
"""
import numpy as np
import pandas as pd

from indicators.trend import trend_position


def test_trend_position_nan_donde_close_nan():
    """Con close=NaN, trend_position debe devolver NaN, no -1.0."""
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(100 + np.arange(n) * 0.1, index=idx)
    close.iloc[-5:] = np.nan  # ultimos 5 dias sin datos

    trend = trend_position(close)

    assert trend.iloc[-5:].isna().all(), (
        "trend debe ser NaN donde close es NaN. "
        f"Valores: {trend.iloc[-5:].tolist()}"
    )


def test_trend_position_valor_donde_close_valido():
    """Con close valido, trend_position devuelve valor numerico."""
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(100 + np.arange(n) * 0.1, index=idx)

    trend = trend_position(close)

    assert trend.iloc[-1:].notna().all(), "trend debe ser numerico con close valido"
    assert -1.0 <= trend.iloc[-1] <= 1.0


def test_trend_position_nan_inicial():
    """Con close NaN al inicio (warm-up), trend debe ser NaN."""
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(100 + np.arange(n) * 0.1, index=idx)
    close.iloc[:10] = np.nan

    trend = trend_position(close)

    assert trend.iloc[:10].isna().all(), (
        "trend debe ser NaN en warm-up con close NaN"
    )