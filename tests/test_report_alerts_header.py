"""Tests report: alerts + header.

Render puros. Verifican ramas de decision, formatos y fallbacks.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.report.alerts import render_alerts, render_cross_module
from src.report.header import render_regimenes


# =============================================================================
# alerts.py
# =============================================================================

def test_render_alerts_sin_divergencias():
    out = render_alerts(breadth_values=None, liquidity_regime=None, price_flow_divergences=None)
    lineas = "".join(out)
    assert "Alertas de Divergencia" in lineas
    assert "Sin divergencias relevantes" in lineas


def test_render_alerts_breadth_divergence():
    out = render_alerts(
        breadth_values={"% sobre EMA200": 0.75, "% sobre EMA20": 0.50},
        liquidity_regime=None,
        price_flow_divergences=None,
    )
    lineas = "".join(out)
    assert "Breadth Divergence" in lineas
    assert "75%" in lineas
    assert "50%" in lineas


def test_render_alerts_breadth_sin_divergencia():
    """ema200 alto pero ema20 tambien alto -> no alerta."""
    out = render_alerts(
        breadth_values={"% sobre EMA200": 0.75, "% sobre EMA20": 0.75},
        liquidity_regime=None,
        price_flow_divergences=None,
    )
    lineas = "".join(out)
    assert "Breadth Divergence" not in lineas


def test_render_alerts_high_stress():
    out = render_alerts(
        breadth_values=None,
        liquidity_regime="HIGH_STRESS",
        price_flow_divergences=None,
    )
    lineas = "".join(out)
    assert "Financial Stress vs Credit" in lineas


def test_render_alerts_price_flow_divergence():
    out = render_alerts(
        breadth_values=None, liquidity_regime=None,
        price_flow_divergences={
            "XLK": {"status": "DIVERGENCE"},
            "XLF": {"status": "ALIGNED"},
        },
    )
    lineas = "".join(out)
    assert "Technology Price-Flow" in lineas
    assert "Financials Price-Flow" not in lineas  # ALIGNED


def test_render_cross_module_vacio():
    assert render_cross_module(None) == []
    assert render_cross_module({}) == []


def test_render_cross_module_consensus_icon_ok():
    out = render_cross_module({"conflict_level": "CONSENSUS", "message": "todo ok"})
    lineas = "".join(out)
    assert "OK Cross-Module: CONSENSUS" in lineas


def test_render_cross_module_conflict_icon_warn():
    out = render_cross_module({"conflict_level": "CONFLICT", "message": "conflicto"})
    lineas = "".join(out)
    assert "WARN Cross-Module: CONFLICT" in lineas


def test_render_cross_module_info_icon():
    out = render_cross_module({"conflict_level": "MIXED", "message": "mixto"})
    lineas = "".join(out)
    assert "INFO Cross-Module: MIXED" in lineas


def test_render_cross_module_con_details():
    out = render_cross_module({
        "conflict_level": "CONSENSUS",
        "message": "ok",
        "blocks": "A | B | C",
        "details": {
            "macro": {"state": "MIXED", "bias_financial": 0, "bias_inflation": 0},
            "financial": {"state": "ABUNDANTE", "bias_financial": -1, "bias_inflation": 0},
        },
    })
    lineas = "".join(out)
    assert "**Bloques:** A | B | C" in lineas
    assert "macro: MIXED (Neutral)" in lineas
    assert "financial: ABUNDANTE (Estres Financiero" in lineas


def test_render_cross_module_details_state_none():
    out = render_cross_module({
        "conflict_level": "MIXED", "message": "",
        "details": {"mod1": {"state": None}},
    })
    lineas = "".join(out)
    assert "mod1: N/A" in lineas


# =============================================================================
# header.py
# =============================================================================

def _reg(volatility_score=0.0, macro_score=0.0, liquidity_score=0.0, real_liq_score=None, **overrides):
    """Helper: render_regimenes con valores por defecto, overrideable."""
    args = dict(
        macro_score=pd.Series([macro_score]),
        macro_regime="MIXED",
        macro_conf=0.5,
        liquidity_score=pd.Series([liquidity_score]),
        liquidity_regime="ABUNDANTE",
        liq_conf=0.9,
        volatility_score=volatility_score,
        vol_regime="NORMAL",
        vol_conf=0.5,
        real_liquidity_regime="NEUTRA",
        real_liquidity_conf=0.7,
        real_liq_score=pd.Series([0.5]) if real_liq_score is None else real_liq_score,
        real_liq_prev=None,
        sector_regime="NARROW RALLY",
    )
    args.update(overrides)
    return render_regimenes(**args)


def test_header_macro_basico():
    out = _reg()
    lineas = "".join(out)
    assert "## Resumen de Regimenes" in lineas
    assert "Macro:** MIXED (Score: 0.00" in lineas
    assert "Signal Consistency: 50%" in lineas


def test_header_macro_conf_baja_muestra_nota():
    out = _reg(macro_conf=0.20)
    lineas = "".join(out)
    assert "Signal Consistency baja" in lineas


def test_header_macro_score_nan():
    out = _reg(macro_score=np.nan)
    lineas = "".join(out)
    assert "Score: N/D" in lineas


def test_header_liquidity_high_stress_muestra_nota():
    out = _reg(liquidity_regime="HIGH_STRESS")
    lineas = "".join(out)
    assert "HIGH_STRESS" in lineas
    assert "estres significativo" in lineas


def test_header_real_liq_regime_none_no_muestra_linea():
    out = _reg(real_liquidity_regime=None)
    lineas = "".join(out)
    assert "Liquidez Real (FRED)" not in lineas


def test_header_real_liq_delta_mejora():
    out = _reg(
        real_liq_score=pd.Series([0.60]),
        real_liq_prev=0.50,
    )
    lineas = "".join(out)
    assert "Liquidity Delta" in lineas
    assert "MEJORA" in lineas


def test_header_real_liq_delta_empeora():
    out = _reg(
        real_liq_score=pd.Series([0.40]),
        real_liq_prev=0.50,
    )
    lineas = "".join(out)
    assert "EMPEORA" in lineas


def test_header_real_liq_delta_estable():
    out = _reg(
        real_liq_score=pd.Series([0.50]),
        real_liq_prev=0.50,
    )
    lineas = "".join(out)
    assert "ESTABLE" in lineas


def test_header_vol_neutra_mensaje_especifico():
    """vol_conf < 0.05 + |vol_z| < 0.1 -> mensaje neutro."""
    out = _reg(volatility_score=0.02, vol_conf=0.02)
    lineas = "".join(out)
    assert "Señal neutra" in lineas


def test_header_vol_score_nan():
    out = _reg(volatility_score=np.nan)
    lineas = "".join(out)
    assert "Z-Score: N/D" in lineas


def test_header_vol_muy_cerca_cero_muestra_0_00():
    out = _reg(volatility_score=0.001, vol_conf=0.5)
    lineas = "".join(out)
    assert "Z-Score: 0.00" in lineas