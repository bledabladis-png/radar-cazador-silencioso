# -*- coding: utf-8 -*-
"""Centinela H3: render_regimenes tolera macro_conf None/nan.

Cubre B6-H3 (2026-10-31). Antes: macro_conf=None -> TypeError en f-string;
macro_conf=nan -> "nan%" en output. Ahora: ambos pintan N/D (simetria con
liq_conf, que ya tenia guard pd.notna).
"""

import pathlib

import pandas as pd

from src.report.header import render_regimenes


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _reg(**overrides):
    args = dict(
        macro_score=pd.Series([0.5]),
        macro_regime="NEUTRAL",
        macro_conf=0.5,
        liquidity_score=pd.Series([0.4]),
        financial_regime="NEUTRAL",
        liq_conf=0.5,
        volatility_score=pd.Series([0.1]),
        vol_regime="LOW",
        vol_conf=0.5,
        real_liquidity_regime=None,
        real_liquidity_conf=0.5,
        real_liq_score=pd.Series([0.1]),
        real_liq_prev=0.1,
        sector_regime="MIXED",
        confirmation_data=None,
    )
    args.update(overrides)
    return render_regimenes(**args)


def _macro_line(out):
    return [l for l in out if "**Macro:**" in l][0]


def test_macro_conf_none_pinta_nd_sin_crashear():
    line = _macro_line(_reg(macro_conf=None))
    assert "Signal Consistency: N/D" in line
    assert "Signal Consistency: 0%" not in line


def test_macro_conf_nan_pinta_nd_sin_nan_literal():
    line = _macro_line(_reg(macro_conf=float("nan")))
    assert "Signal Consistency: N/D" in line
    assert "nan%" not in line


def test_macro_conf_numeral_pinta_porcentaje():
    line = _macro_line(_reg(macro_conf=0.5))
    assert "Signal Consistency: 50%" in line


def test_macro_conf_baja_con_none_no_dispara_aviso():
    """pd.notna(None) es False: el aviso de <0.30 no debe disparar."""
    out = _reg(macro_conf=None)
    macro_lines = [l for l in out if "**Macro:**" in l or "Signal Consistency baja" in l]
    assert len(macro_lines) == 1


def test_guard_presente_en_source():
    src = (ROOT / "src" / "report" / "header.py").read_text(encoding="utf-8")
    assert "macro_conf_str = f\"{macro_conf:.0%}\" if pd.notna(macro_conf) else \"N/D\"" in src
    assert "if pd.notna(macro_conf) and macro_conf < 0.30:" in src
