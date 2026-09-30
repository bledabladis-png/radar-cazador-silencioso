# -*- coding: utf-8 -*-
"""F3-05-sexies (2026-09-30): atr no propaga NaN de festivos.

Bug: df_market es multi-mercado. Un ticker USA tiene NaN en cada
festivo NYSE. Sin dropna, un NaN en tr contamina window sesiones
consecutivas del ATR. Verificado sobre data/market_data.parquet:
XLB con tr NaN en 2026-09-07 -> atr14 con 14 NaN en tail(30).

Fix: atr hace dropna() previo sobre High/Low/Close.

Unico consumidor: regimes/sector_regime.py:68 (componente
volatility_inv del score sectorial).
"""
import numpy as np
import pandas as pd

from indicators.volatility import atr
from src.utils import get_col


def _make_df_with_nan_at(tickers, n=100, nan_positions=None, seed=42):
    """df MultiIndex con NaN puntual en High/Low/Close.

    nan_positions: {ticker: [i1, i2, ...]} posiciones (indices) con NaN.
    """
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {}
    for k, t in enumerate(tickers):
        drift = 0.0001 * (k + 1)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.015, n))
        high = close * 1.01
        low = close * 0.99
        if nan_positions and t in nan_positions:
            for i in nan_positions[t]:
                high[i] = np.nan
                low[i] = np.nan
                close[i] = np.nan
        data[("Close", t)] = close
        data[("Open", t)] = close * 0.998
        data[("High", t)] = high
        data[("Low", t)] = low
        data[("Volume", t)] = rng.randint(1000000, 5000000, n)
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_atr_no_propaga_nan_de_festivo():
    """Un NaN en tr a -16 contamina 14 sesiones del atr sin el fix.

    Con el fix: 0 NaN en tail(30).
    """
    df = _make_df_with_nan_at(["TICK"], n=100, nan_positions={"TICK": [-16]})
    a = atr(df, "TICK")
    # Sin el fix: el NaN a -16 -> 14 NaN consecutivos en el rolling(14)
    # Con el fix: el NaN se dropea antes del rolling
    assert a.tail(30).isna().sum() == 0, (
        f"atr propaga NaN. tail(30) NaN={a.tail(30).isna().sum()}"
    )
    assert pd.notna(a.iloc[-1])


def test_atr_equivalente_a_dropna_manual():
    """Verifica que atr(df_con_nan) == atr_calculado_sobre_df_limpio."""
    df = _make_df_with_nan_at(["TICK"], n=100, nan_positions={"TICK": [-16, -15]})
    a = atr(df, "TICK")

    # Calculo equivalente manual
    high = get_col(df, "TICK", "High")
    low = get_col(df, "TICK", "Low")
    close = get_col(df, "TICK", "Close")
    sub = pd.DataFrame({"High": high, "Low": low, "Close": close}).dropna()
    prev = sub["Close"].shift(1)
    tr = pd.concat([
        sub["High"] - sub["Low"],
        (sub["High"] - prev).abs(),
        (sub["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    a_manual = tr.rolling(14).mean()

    assert a.index.equals(a_manual.index)
    assert np.allclose(a.fillna(-999).values, a_manual.fillna(-999).values)


def test_atr_df_sin_nan_funciona_igual():
    """Control: sin NaN, atr se comporta como antes."""
    df = _make_df_with_nan_at(["TICK"], n=100)
    a = atr(df, "TICK")
    assert len(a) == 100
    assert a.iloc[:13].isna().all()
    assert a.iloc[13:].notna().all()


def test_atr_insuficientes_datos_devuelve_vacio():
    """Si tras dropna quedan < window filas, devuelve Series vacia."""
    df = _make_df_with_nan_at(["TICK"], n=10)
    a = atr(df, "TICK")
    assert a.empty or len(a) < 10
