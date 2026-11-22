# -*- coding: utf-8 -*-
"""Helpers para tests de breadth: construir BreadthResult con defaults.

2026-11-21 (S39, 13.11): tras el cambio a BreadthResult con denominadores
dinamicos, los tests que mockean compute_breadth deben devolver un
BreadthResult valido. Este helper centraliza la construccion.

Defaults derivados (no hardcodeados):
- expected = len(MARKET_TICKERS["sectors"]) si no se pasa.
- denom_* = expected si no se pasa (cobertura completa).
- index = derivado del primer Series de b20/b50/b200/nh/nl si alguno es
  Series; si no, 3 fechas sinteticas.

Uso tipico en mocks:
    from tests._breadth_fixtures import make_breadth_result
    with patch(..., return_value=make_breadth_result(b20=my_series, ...)):
        ...

Uso tipico en tests de cobertura parcial:
    br = make_breadth_result(denom_ema20=8, denom_ema50=8,
                             denom_ema200=8, denom_nhnl=8, expected=11)
"""
from __future__ import annotations

import pandas as pd

from config.tickers import MARKET_TICKERS
from indicators.breadth import BreadthResult


def _derived_expected():
    """Deriva expected del universo real. Sin hardcode."""
    return len(MARKET_TICKERS["sectors"])


def _as_series(v, index):
    if isinstance(v, pd.Series):
        return v
    return pd.Series([v] * len(index), index=index)


def make_breadth_result(
    b20=0.5,
    b50=0.5,
    b200=0.5,
    nh=0.5,
    nl=0.5,
    denom_ema20=None,
    denom_ema50=None,
    denom_ema200=None,
    denom_nhnl=None,
    expected=None,
    index=None,
):
    """Construye BreadthResult para tests. Ver docstring del modulo."""
    if expected is None:
        expected = _derived_expected()
    if index is None:
        # Derivar del primer Series disponible (evita indices desalineados
        # entre bXX y denom_XXX). Si ninguno es Series, index sintetico.
        for v in (b20, b50, b200, nh, nl):
            if isinstance(v, pd.Series):
                index = v.index
                break
        if index is None:
            index = pd.date_range("2026-01-01", periods=3)
    if denom_ema20 is None:
        denom_ema20 = expected
    if denom_ema50 is None:
        denom_ema50 = expected
    if denom_ema200 is None:
        denom_ema200 = expected
    if denom_nhnl is None:
        denom_nhnl = expected
    return BreadthResult(
        b20=_as_series(b20, index),
        b50=_as_series(b50, index),
        b200=_as_series(b200, index),
        nh=_as_series(nh, index),
        nl=_as_series(nl, index),
        denom_ema20=_as_series(denom_ema20, index),
        denom_ema50=_as_series(denom_ema50, index),
        denom_ema200=_as_series(denom_ema200, index),
        denom_nhnl=_as_series(denom_nhnl, index),
        expected=expected,
    )
