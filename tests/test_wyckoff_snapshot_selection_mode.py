# -*- coding: utf-8 -*-
"""Tests de selection_mode en compute_wyckoff_snapshot.

A.2a (DICT3-TOP20-COBERTURA): el parametro selection_mode permite pasar
de top-20 (historico) a target_weight (base nueva) sin romper los
consumidores existentes.

Regresion: con selection_mode="nlargest" el output es identico al
comportamiento historico (mismo numero de filas por grupo).

Nuevo: con selection_mode="target_weight" la cobertura de cada grupo
sube al objetivo (o se queda en max_tickers si inalcanzable).
"""
import numpy as np
import pandas as pd
import pytest

from indicators.wyckoff_holdings_report import compute_wyckoff_snapshot


def _df_stocks_sintetico(n_tickers=100, n_bars=500, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    tickers = ["TK{:03d}".format(i) for i in range(n_tickers)]
    base = 100 + rng.standard_normal((n_bars, n_tickers)).cumsum(axis=0)
    data = {}
    for f in ("Open", "High", "Low", "Close", "Volume"):
        if f == "Close":
            data[(f, *[()])] = None
    frames = []
    for f in ("Open", "High", "Low", "Close", "Volume"):
        if f == "Volume":
            arr = rng.integers(1_000_000, 10_000_000, (n_bars, n_tickers)).astype(float)
        else:
            jitter = rng.standard_normal((n_bars, n_tickers)) * 0.5
            arr = base + jitter if f != "Close" else base
        cols = pd.MultiIndex.from_product([[f], tickers])
        frames.append(pd.DataFrame(arr, index=idx, columns=cols))
    return pd.concat(frames, axis=1).sort_index(axis=1)


def _holdings_sintetico(n_tickers=100):
    # 100 tickers con pesos decrecientes que suman 100.
    weights = np.linspace(2.0, 0.02, n_tickers)
    weights = weights * (100.0 / weights.sum())  # normalizar a 100
    return pd.DataFrame({
        "etf": ["XLI"] * n_tickers,
        "ticker": ["TK{:03d}".format(i) for i in range(n_tickers)],
        "identifier": ["XLI-{}".format(i) for i in range(n_tickers)],
        "weight": weights,
    })


def test_selection_mode_nlargest_regresion_top20():
    """Con 'nlargest' el numero de filas por grupo es <= TOP_N_HOLDINGS."""
    df = _df_stocks_sintetico()
    hold = _holdings_sintetico()
    snap = compute_wyckoff_snapshot(
        df, hold, None, None, selection_mode="nlargest")
    assert not snap.empty
    for group, sub in snap.groupby("group"):
        assert len(sub) <= 20, f"grupo {group} tiene {len(sub)} filas (max 20)"


def test_selection_mode_target_weight_aumenta_cobertura():
    """Con 'target_weight' el grupo puede tener mas de 20 filas si el peso
    esta repartido entre muchos tickers."""
    df = _df_stocks_sintetico(n_tickers=100)
    hold = _holdings_sintetico(n_tickers=100)
    snap_n = compute_wyckoff_snapshot(
        df, hold, None, None, selection_mode="nlargest")
    snap_t = compute_wyckoff_snapshot(
        df, hold, None, None, selection_mode="target_weight")
    n_n = len(snap_n[snap_n["group"] == "XLI"])
    n_t = len(snap_t[snap_t["group"] == "XLI"])
    assert n_t > n_n, (
        "target_weight deberia seleccionar mas tickers que nlargest "
        "con pesos planos. n_n={} n_t={}".format(n_n, n_t))


def test_selection_mode_target_weight_cobertura_pct_en_rango():
    """La columna coverage_weight_pct refleja el target (85% +- margen)."""
    df = _df_stocks_sintetico(n_tickers=100)
    hold = _holdings_sintetico(n_tickers=100)
    snap = compute_wyckoff_snapshot(
        df, hold, None, None, selection_mode="target_weight")
    xli = snap[snap["group"] == "XLI"]
    assert not xli.empty
    cob = xli["coverage_weight_pct"].iloc[0]
    assert 80.0 <= cob <= 100.0, (
        "cobertura XLI {} fuera de [80, 100]".format(cob)
    )


def test_selection_mode_invalido_raise():
    df = _df_stocks_sintetico(n_tickers=10, n_bars=300)
    hold = _holdings_sintetico(n_tickers=10)
    with pytest.raises(ValueError, match="selection_mode"):
        compute_wyckoff_snapshot(
            df, hold, None, None, selection_mode="foo_mode")