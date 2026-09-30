# -*- coding: utf-8 -*-
"""D30 (2026-09-30): tests para 3 modulos de regimes con <35% cobertura.

- volatility_regime.compute_volatility_regime: 33%.
- tactical_engine.compute_tactical_score: 14%.
- structural_engine.compute_structural_score: 21%.

Los 3 son puros (reciben df_market + parametros). Tests con
DataFrames sinteticos.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd

from regimes.volatility_regime import compute_volatility_regime
from regimes.tactical_engine import compute_tactical_score
from regimes.structural_engine import compute_structural_score


def _make_df(tickers, n=300, seed=42, drift=0.0005):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {}
    for i, t in enumerate(tickers):
        close = 100 * np.cumprod(1 + rng.normal(drift + i * 0.0001, 0.015, n))
        data[("Close", t)] = close
        data[("High", t)] = close * 1.01
        data[("Low", t)] = close * 0.99
        data[("Open", t)] = close * 0.998
        data[("Volume", t)] = rng.randint(1_000_000, 5_000_000, n)
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# ============================================================
# compute_volatility_regime
# Mockeamos volatility_regime (la funcion de indicators/volatility)
# para devolver un z conocido. Asi cubrimos el mapeo completo del
# regimen (LOW/NORMAL/ELEVATED/STRESS) sin depender del rolling.
# ============================================================
def _z_series(value, n=10):
    return pd.Series([value] * n)


def test_volatility_regime_vacio():
    with patch("regimes.volatility_regime.volatility_regime",
               return_value=pd.Series(dtype=float)):
        z, regime, conf = compute_volatility_regime(pd.Series(dtype=float))
    assert z.empty
    assert regime == "N/D"
    assert conf == 0.0


def test_volatility_regime_ultimo_nan():
    with patch("regimes.volatility_regime.volatility_regime",
               return_value=_z_series(float("nan"))):
        z, regime, conf = compute_volatility_regime(pd.Series([0.1] * 10))
    assert regime == "N/D"
    assert conf == 0.0


def test_volatility_regime_low():
    with patch("regimes.volatility_regime.volatility_regime",
               return_value=_z_series(-1.0)):
        z, regime, conf = compute_volatility_regime(pd.Series([0.1] * 10))
    assert regime == "LOW"
    assert conf == 0.5  # min(abs(-1)/2, 1)


def test_volatility_regime_normal():
    with patch("regimes.volatility_regime.volatility_regime",
               return_value=_z_series(0.0)):
        z, regime, conf = compute_volatility_regime(pd.Series([0.1] * 10))
    assert regime == "NORMAL"
    assert conf == 0.0


def test_volatility_regime_elevated():
    with patch("regimes.volatility_regime.volatility_regime",
               return_value=_z_series(1.0)):
        z, regime, conf = compute_volatility_regime(pd.Series([0.1] * 10))
    assert regime == "ELEVATED"
    assert conf == 0.5


def test_volatility_regime_stress():
    with patch("regimes.volatility_regime.volatility_regime",
               return_value=_z_series(2.5)):
        z, regime, conf = compute_volatility_regime(pd.Series([0.1] * 10))
    assert regime == "STRESS"
    # min(abs(2.5)/2, 1) = 1.0
    assert conf == 1.0


def test_volatility_regime_confidence_rango():
    """confidence en [0, 1] siempre, valores extremos incluidos."""
    for value in (-10.0, -1.0, 0.0, 1.0, 10.0):
        with patch("regimes.volatility_regime.volatility_regime",
                   return_value=_z_series(value)):
            _, _, conf = compute_volatility_regime(pd.Series([0.1] * 10))
        assert 0.0 <= conf <= 1.0


# ============================================================
# compute_tactical_score
# ============================================================
def test_tactical_ticker_ausente():
    df = _make_df(["^GSPC"])
    out = compute_tactical_score(df, "XLK")
    assert out == 0.0


def test_tactical_benchmark_ausente():
    df = _make_df(["XLK"])
    out = compute_tactical_score(df, "XLK")
    assert out == 0.0


def test_tactical_normal():
    df = _make_df(["XLK", "^GSPC"])
    out = compute_tactical_score(df, "XLK")
    assert -1.0 <= out <= 1.0


def test_tactical_serie_corta():
    """Serie < 21 filas -> componentes con fallback a 0."""
    df = _make_df(["XLK", "^GSPC"], n=15)
    out = compute_tactical_score(df, "XLK")
    assert -1.0 <= out <= 1.0


def test_tactical_benchmark_custom():
    df = _make_df(["XLK", "SPY"])
    out = compute_tactical_score(df, "XLK", benchmark="SPY")
    assert -1.0 <= out <= 1.0


# ============================================================
# compute_structural_score
# ============================================================
def test_structural_ticker_ausente():
    df = _make_df(["^GSPC"])
    out = compute_structural_score(df, "XLK")
    assert out == 0.0


def test_structural_benchmark_ausente():
    df = _make_df(["XLK"])
    out = compute_structural_score(df, "XLK")
    assert out == 0.0


def test_structural_normal():
    df = _make_df(["XLK", "^GSPC"])
    out = compute_structural_score(df, "XLK")
    assert -1.0 <= out <= 1.0


def test_structural_serie_corta():
    """Serie < min(window) -> rs_momentum devuelve 0."""
    df = _make_df(["XLK", "^GSPC"], n=100)
    out = compute_structural_score(df, "XLK")
    assert -1.0 <= out <= 1.0


def test_structural_persistence_alto():
    df = _make_df(["XLK", "^GSPC"])
    out_alto = compute_structural_score(df, "XLK", persistence=1.0)
    out_bajo = compute_structural_score(df, "XLK", persistence=0.0)
    # persistence empuja el score en direcciones opuestas
    assert out_alto >= out_bajo


def test_structural_flow_structure_positivo():
    df = _make_df(["XLK", "^GSPC"])
    out_pos = compute_structural_score(df, "XLK", flow_structure=1.0)
    out_neg = compute_structural_score(df, "XLK", flow_structure=-1.0)
    assert out_pos >= out_neg


def test_structural_benchmark_custom():
    df = _make_df(["XLK", "SPY"])
    out = compute_structural_score(df, "XLK", benchmark="SPY")
    assert -1.0 <= out <= 1.0
