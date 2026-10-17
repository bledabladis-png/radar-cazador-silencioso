# -*- coding: utf-8 -*-
"""Regresion C2 (2026-10-17): aviso de Signal Consistency baja en
Condiciones Financieras, simetrico al de Macro.

Antes: solo macro avisaba con `macro_conf < 0.30`. Financial, con
liq_conf=0.15 y etiqueta HIGH_STRESS, no avisaba. El lector veia
"Signal Consistency: 15%" sin ninguna nota explicando que significa
(desacuerdo entre los 4 componentes: VIX, credito, dolar, curva).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.report.header import render_regimenes


def _base_call(liq_conf):
    return render_regimenes(
        macro_score=pd.Series([-0.30]),
        macro_regime="SLOWDOWN",
        macro_conf=0.56,
        liquidity_score=pd.Series([-0.57]),
        liquidity_regime="HIGH_STRESS",
        liq_conf=liq_conf,
        volatility_score=pd.Series([-0.12]),
        vol_regime="NORMAL",
        vol_conf=0.06,
        real_liquidity_regime="NEUTRAL",
        real_liquidity_conf=0.90,
        real_liq_score=pd.Series([-0.03]),
        real_liq_prev=None,
        sector_regime="ROTATIONAL",
    )


def test_financial_consistency_baja_emite_aviso():
    """liq_conf=0.15 -> nota de consistencia baja."""
    out = _base_call(liq_conf=0.15)
    text = "".join(out)
    assert "Signal Consistency baja" in text
    assert "componentes" in text.lower()


def test_financial_consistency_alta_no_emite_aviso():
    """liq_conf=0.80 -> sin nota de consistencia baja (solo la de HIGH_STRESS)."""
    out = _base_call(liq_conf=0.80)
    text = "".join(out)
    # El texto de HIGH_STRESS de L49 no lleva "Signal Consistency baja".
    assert "Signal Consistency baja: los componentes" not in text