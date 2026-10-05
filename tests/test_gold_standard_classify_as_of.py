# -*- coding: utf-8 -*-
"""Tests wrapper temporal/as-of.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators.wyckoff_v1 import classify_wyckoff_phase
from scripts.gold_standard.classify_as_of import (
    FROZEN_V19_PARAMS,
    classify_as_of,
    classify_as_of_batch,
)


def _make_ticker_df(n=500, seed=42, ticker="AAA"):
    """OHLCV flat (no MultiIndex) para un ticker."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    high = price + np.abs(rng.normal(0, 0.5, n))
    low = price - np.abs(rng.normal(0, 0.5, n))
    open_ = price + rng.normal(0, 0.3, n)
    close = price
    volume = np.abs(rng.normal(1e6, 2e5, n))
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=idx)


def test_frozen_v19_params_completo():
    for k in ("window", "x_atr", "y_vol", "max_age_m"):
        assert k in FROZEN_V19_PARAMS
    assert FROZEN_V19_PARAMS["window"] == 60
    assert FROZEN_V19_PARAMS["x_atr"] == 0.25
    assert FROZEN_V19_PARAMS["y_vol"] == 1.10
    assert FROZEN_V19_PARAMS["max_age_m"] == 10


def test_classify_as_of_rechaza_none():
    df = _make_ticker_df()
    with pytest.raises(ValueError, match="as_of explicito"):
        classify_as_of(df, "AAA", None)

def test_classify_as_of_equivalencia_bit_a_bit():
    """Wrapper == llamada directa con df truncado y params congelados."""
    df = _make_ticker_df()
    L = df.index[-30]
    r_wrapper = classify_as_of(df, "AAA", L)
    r_directo = classify_wyckoff_phase(
        df.loc[:L], "AAA", as_of=L, sow_params=dict(FROZEN_V19_PARAMS),
    )
    assert r_wrapper == r_directo


def test_classify_as_of_sin_lookahead():
    """Cambiar el futuro no debe cambiar la clasificacion en L."""
    df = _make_ticker_df()
    L = df.index[-30]
    r1 = classify_as_of(df, "AAA", L)
    # Modificar el futuro (post L) de forma agresiva
    df_mod = df.copy()
    df_mod.loc[df_mod.index > L, "Close"] = df_mod.loc[df_mod.index > L, "Close"] * 2.0
    df_mod.loc[df_mod.index > L, "High"] = df_mod.loc[df_mod.index > L, "High"] * 2.0
    df_mod.loc[df_mod.index > L, "Low"] = df_mod.loc[df_mod.index > L, "Low"] * 2.0
    r2 = classify_as_of(df_mod, "AAA", L)
    assert r1 == r2


def test_classify_as_of_sow_params_explicito():
    """Pasar sow_params explicitos debe respetarse (no override)."""
    df = _make_ticker_df()
    L = df.index[-30]
    params_permisivos = {
        "window": 20, "x_atr": 0.0, "y_vol": 1.0, "max_age_m": 10,
    }
    r = classify_as_of(df, "AAA", L, sow_params=params_permisivos)
    assert isinstance(r, str)


def test_classify_as_of_inyecta_frozen_por_defecto():
    """Sin sow_params, debe inyectar FROZEN_V19_PARAMS."""
    df = _make_ticker_df()
    L = df.index[-30]
    r_auto = classify_as_of(df, "AAA", L)
    r_explicito = classify_as_of(df, "AAA", L, sow_params=dict(FROZEN_V19_PARAMS))
    assert r_auto == r_explicito


def test_classify_as_of_batch():
    df = _make_ticker_df()
    Ls = [df.index[-50], df.index[-40], df.index[-30]]
    out = classify_as_of_batch(df, "AAA", Ls)
    assert len(out) == 3
    for L in Ls:
        assert L in out
        assert isinstance(out[L], str)


def test_classify_as_of_no_depende_de_settings():
    """El wrapper debe funcionar con config.settings en None (fail-closed)."""
    import config.settings as s
    assert s.WYCKOFF_SOW_WINDOW_N is None
    assert s.WYCKOFF_SOW_MAX_AGE_M is None
    assert s.WYCKOFF_SOW_X_ATR is None
    assert s.WYCKOFF_SOW_Y_VOL is None
    df = _make_ticker_df()
    L = df.index[-30]
    r = classify_as_of(df, "AAA", L)
    assert isinstance(r, str)
    assert r in (
        "MARKUP", "ACCUMULATION", "RANGE",
        "DISTRIBUTION", "MARKDOWN", "INSUFFICIENT_DATA",
    )


def test_classify_as_of_determinista():
    df = _make_ticker_df()
    L = df.index[-30]
    r1 = classify_as_of(df, "AAA", L)
    r2 = classify_as_of(df, "AAA", L)
    assert r1 == r2


def test_classify_as_of_truncamiento_idempotente():
    """Pasar df ya truncado a L debe dar el mismo resultado."""
    df = _make_ticker_df()
    L = df.index[-30]
    r_full = classify_as_of(df, "AAA", L)
    r_truncado = classify_as_of(df.loc[:L], "AAA", L)
    assert r_full == r_truncado