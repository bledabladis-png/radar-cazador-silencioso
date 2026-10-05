# -*- coding: utf-8 -*-
"""Tests ctx (protocolo v8, seccion 3).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.ctx import (
    compute_ctx,
    ctx_stats,
)


def _make_ohlcv(n=500, seed=42, modo="alcista_despues_caida"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    if modo == "alcista_despues_caida":
        # 300 sesiones subida + 100 correccion + resto
        pre = 100 + np.cumsum(rng.normal(0.3, 1, 300))
        dd = pre[-1] - np.cumsum(rng.normal(0.5, 1, 100))
        post = dd[-1] + np.cumsum(rng.normal(0.1, 1, n - 400))
        close = np.concatenate([pre, dd, post])
    elif modo == "plano":
        close = np.full(n, 100.0)
    else:
        close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "Open": close,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Volume": np.abs(rng.normal(1e6, 2e5, n)),
    }, index=idx)


def test_ctx_serie_booleana_misma_longitud():
    df = _make_ohlcv(500)
    ctx = compute_ctx(df)
    assert len(ctx) == len(df)
    assert ctx.dtype == bool


def test_ctx_nan_iniciales_son_false():
    df = _make_ohlcv(500)
    ctx = compute_ctx(df)
    # Primeros 200+60 sesiones no pueden tener MA200 valido en t-60
    assert not ctx.iloc[:250].any()


def test_ctx_plano_sin_senales():
    df = _make_ohlcv(500, modo="plano")
    ctx = compute_ctx(df)
    # Sin movimiento, no hay ctx=1
    assert not ctx.any()


def test_ctx_caso_construido_activa():
    """Caso disenado con uptrend previo y drawdown >= 15% posterior."""
    df = _make_ohlcv(500, modo="alcista_despues_caida")
    ctx = compute_ctx(df)
    # Debe haber al menos algunos ctx=1
    assert ctx.sum() > 0


def test_ctx_falta_close():
    df = _make_ohlcv(500, modo="plano").drop(columns=["Close"])
    import pytest
    with pytest.raises(ValueError, match="Falta columna"):
        compute_ctx(df)


def test_ctx_stats():
    df = _make_ohlcv(500)
    ctx = compute_ctx(df)
    s = ctx_stats(ctx)
    assert s["n_total"] == len(ctx)
    assert s["n_ctx1"] + s["n_ctx0"] == s["n_total"]


def test_ctx_determinista():
    df = _make_ohlcv(500)
    c1 = compute_ctx(df)
    c2 = compute_ctx(df)
    pd.testing.assert_series_equal(c1, c2)


def test_ctx_usa_solo_ohlcv():
    """ctx no debe usar ninguna columna fuera de OHLCV."""
    df = _make_ohlcv(500)
    # Anadir una columna "basura" que no debe usarse
    df["basura"] = 9999
    c1 = compute_ctx(df)
    df2 = df.drop(columns=["basura"])
    c2 = compute_ctx(df2)
    pd.testing.assert_series_equal(c1, c2)