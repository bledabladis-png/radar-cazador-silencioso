# -*- coding: utf-8 -*-
"""D22 (2026-09-30): tests para 3 modulos de indicators que alimentan
macro_regime y no tenian cobertura directa:

- credit.credit_risk_signal (19% -> cobertura de las ramas principales).
- breadth.compute_breadth (22%).
- macro_fundamental.fundamental_signals (13%).

Ninguno tiene red. Todos reciben DataFrames y devuelven Series/DataFrames.
Los tests usan datos sinteticos con columnas MultiIndex (Close, ticker).
"""
import numpy as np
import pandas as pd

from indicators.credit import credit_risk_signal, credit_spread_signal
from indicators.breadth import compute_breadth
from indicators.macro_fundamental import fundamental_signals
from config.tickers import MARKET_TICKERS


def _make_market_df(tickers, n=300, seed=42):
    """DataFrame MultiIndex (Close/High/Low/Volume, ticker)."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {}
    for i, t in enumerate(tickers):
        drift = 0.0001 * (i + 1)
        close = 100 * np.cumprod(1 + rng.normal(drift, 0.015, n))
        data[("Close", t)] = close
        data[("High", t)] = close * 1.01
        data[("Low", t)] = close * 0.99
        data[("Volume", t)] = rng.randint(1_000_000, 5_000_000, n)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# ---------- credit_risk_signal ----------
def test_credit_risk_signal_normal():
    df = _make_market_df(["HYG", "LQD", "IEF"], n=300)
    out = credit_risk_signal(df)
    assert isinstance(out, pd.Series)
    assert len(out) > 0
    assert out.name == "credit_signal"
    # Valores en [-1, 1] por construccion (mezcla de tanh).
    assert out.dropna().between(-1, 1).all()


def test_credit_risk_signal_columna_ausente():
    df = _make_market_df(["HYG", "LQD"], n=300)  # sin IEF
    out = credit_risk_signal(df)
    assert isinstance(out, pd.Series)
    assert out.empty


def test_credit_risk_signal_menos_de_60_filas():
    """<60 filas: Series indexada con NaN (no vacia)."""
    df = _make_market_df(["HYG", "LQD", "IEF"], n=30)
    out = credit_risk_signal(df)
    assert isinstance(out, pd.Series)
    assert out.dtype == float
    # Todos los valores son NaN
    assert out.isna().all()


def test_credit_spread_signal_deprecated_equivalente():
    """credit_spread_signal mantiene retrocompatibilidad."""
    df = _make_market_df(["HYG", "LQD", "IEF"], n=300)
    a = credit_risk_signal(df)
    b = credit_spread_signal(df)
    pd.testing.assert_series_equal(a, b)


# ---------- compute_breadth ----------
def test_compute_breadth_cobertura_completa():
    sectors = MARKET_TICKERS["sectors"]
    df = _make_market_df(list(sectors), n=300)
    b20, b50, b200, nh, nl = compute_breadth(df)
    for s in (b20, b50, b200, nh, nl):
        assert isinstance(s, pd.Series)
        assert len(s) == 300
    # b20/b50/b200 son ratios en [0, 1].
    assert b20.dropna().between(0, 1).all()
    assert b50.dropna().between(0, 1).all()
    assert b200.dropna().between(0, 1).all()
    # nh/nl tambien.
    assert nh.dropna().between(0, 1).all()
    assert nl.dropna().between(0, 1).all()


def test_compute_breadth_cobertura_parcial():
    """Faltan sectores -> columnas NaN, sin crash."""
    sectors = MARKET_TICKERS["sectors"]
    solo_3 = sectors[:3]
    df = _make_market_df(solo_3, n=300)
    b20, b50, b200, nh, nl = compute_breadth(df)
    # Las series existen; con 8 sectores faltantes el b20 sera < 1.
    assert b20.max() < 1.0
    assert b200.dropna().empty or (b200.dropna() <= 1).all()


def test_compute_breadth_sin_sectores():
    """Sin columnas de sectores -> todas NaN."""
    df = _make_market_df(["^GSPC"], n=300)
    b20, b50, b200, nh, nl = compute_breadth(df)
    # np.nan > ema -> False; mean de False = 0
    assert (b20 == 0).all() or b20.isna().any()


# ---------- fundamental_signals ----------
def _make_macro_df_with(cols, n=24):
    """df_macro con columnas dadas + date mensual."""
    idx = pd.date_range("2026-01-01", periods=n, freq="MS")
    data = {"date": idx}
    for c in cols:
        data[c] = np.linspace(1.0, 2.0, n)
    return pd.DataFrame(data)


def test_fundamental_signals_none():
    assert fundamental_signals(None) is None


def test_fundamental_signals_vacio():
    assert fundamental_signals(pd.DataFrame()) is None


def test_fundamental_signals_inflacion():
    df = _make_macro_df_with(["cpi_total"])
    out = fundamental_signals(df)
    assert out is not None
    assert "inflation" in out.columns
    # tanh -> [-1, 1]
    assert out["inflation"].dropna().between(-1, 1).all()


def test_fundamental_signals_empleo():
    df = _make_macro_df_with(["nfp_total"])
    out = fundamental_signals(df)
    assert "employment" in out.columns


def test_fundamental_signals_actividad():
    df = _make_macro_df_with(["industrial_production_total"])
    out = fundamental_signals(df)
    assert "activity" in out.columns


def test_fundamental_signals_sin_columnas_reconocidas():
    """Columnas que no matchean ningun keyword -> df vacio de columnas."""
    df = _make_macro_df_with(["foo_bar", "baz_qux"])
    out = fundamental_signals(df)
    # signals vacio (sin columnas) pero no None
    assert out is not None
    assert len(out.columns) == 0


def test_fundamental_signals_combinado():
    df = _make_macro_df_with(["cpi_total", "nfp_total", "industrial_production_total"])
    out = fundamental_signals(df)
    assert "inflation" in out.columns
    assert "employment" in out.columns
    assert "activity" in out.columns
