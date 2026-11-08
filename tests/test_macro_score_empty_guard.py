# -*- coding: utf-8 -*-
"""Centinela MACRO-SCORE-EMPTY-LATENTE (S34, 2026-11-08).

Cierra el item abierto el 2026-10-31: "render_regimenes / render_macro
probablemente asumen macro_score presente". Verificado en S34: el guard
existe desde la extraccion inicial (commit d5b94209) y cubre los 4 casos
limite (None, Series vacia, Series NaN, NaN escalar) sin excepcion.
Este test congela el contrato para que un refactor futuro no lo rompa.
"""
import numpy as np
import pandas as pd
from src.report.header import render_regimenes


def _call(macro_score, macro_conf=0.5):
    return render_regimenes(
        macro_score=macro_score,
        macro_regime="MIXED",
        macro_conf=macro_conf,
        liquidity_score=pd.Series([0.0]),
        financial_regime="NEUTRAL",
        liq_conf=0.5,
        volatility_score=pd.Series([0.0]),
        vol_regime="NORMAL",
        vol_conf=0.5,
        real_liquidity_regime="NEUTRAL",
        real_liquidity_conf=0.5,
        real_liq_score=pd.Series([0.0]),
        real_liq_prev=0.0,
        sector_regime="MIXED",
    )


def _macro_line(out):
    return [l for l in out if "Macro:" in l][0]


def test_macro_score_none_pinta_nd():
    linea = _macro_line(_call(None))
    assert "Score: N/D" in linea
    assert "Signal Consistency: 50%" in linea


def test_macro_score_series_vacia_pinta_nd():
    linea = _macro_line(_call(pd.Series([], dtype=float)))
    assert "Score: N/D" in linea


def test_macro_score_series_nan_pinta_nd():
    linea = _macro_line(_call(pd.Series([np.nan])))
    assert "Score: N/D" in linea


def test_macro_score_nan_escalar_pinta_nd():
    linea = _macro_line(_call(np.nan))
    assert "Score: N/D" in linea


def test_macro_score_valido_pinta_numero():
    linea = _macro_line(_call(pd.Series([0.5]), macro_conf=0.6))
    assert "Score: 0.50" in linea
    assert "Signal Consistency: 60%" in linea
