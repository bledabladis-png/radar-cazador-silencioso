# -*- coding: utf-8 -*-
"""REP-AUDIT-20261017-capa2: tests de sostenibilidad del texto de
alertas y resumen de regimenes.
"""
import pandas as pd

from config.settings import (
    NEUTRAL_BAND_HYG_LQD_ZSCORE as H14_UMBRAL,
    FLS_STRESS_THRESHOLD as H15_UMBRAL,
)
from src.report.alerts import render_alerts
from src.report.header import render_regimenes


def _run_alerta(hyg_z):
    conf = {"ratios": {"hyg_lqd_zscore": hyg_z}}
    return "".join(render_alerts(
        breadth_values=None, liquidity_regime="HIGH_STRESS",
        price_flow_divergences=None, confirmation_data=conf,
    ))


def _run_regimenes(liquidity_regime, fls_normalized):
    conf = {}
    if fls_normalized is not None:
        conf["fls"] = {"fls_normalized": fls_normalized}
    return "".join(render_regimenes(
        macro_score=pd.Series([0.5]),
        macro_regime="MIXED", macro_conf=0.6,
        liquidity_score=pd.Series([0.5]),
        liquidity_regime=liquidity_regime, liq_conf=0.5,
        volatility_score=pd.Series([0.0]),
        vol_regime="NORMAL", vol_conf=0.1,
        real_liquidity_regime="NEUTRAL", real_liquidity_conf=0.9,
        real_liq_score=pd.Series([0.0]), real_liq_prev=0.0,
        sector_regime="ROTATIONAL",
        confirmation_data=conf,
    ))


def test_h14_z_cerca_del_umbral_sale_neutral():
    out = _run_alerta(0.6)
    assert "el credito lo confirma" not in out
    assert "neutral" in out.lower()


def test_h14_z_bajo_no_categorico():
    out = _run_alerta(0.15)
    assert "el credito lo confirma" not in out
    assert "neutral" in out.lower()


def test_h14_z_alto_sale_categorico():
    out = _run_alerta(-2.5)
    assert "el credito lo confirma" in out
    assert "neutral" not in out.lower()


def test_h14_umbral_desde_config():
    out = _run_alerta(0.0)
    assert f"|Z|<{H14_UMBRAL}" in out


def test_h15_fls_alto_menciona_financiacion_y_liquidez():
    out = _run_regimenes("HIGH_STRESS", fls_normalized=H15_UMBRAL / 100.0 + 0.05)
    assert "FLS:" in out
    assert "liquidez real (FRED)" in out
    assert "dimension distinta" in out or "dimensión distinta" in out


def test_h15_fls_bajo_mantiene_texto_historico():
    out = _run_regimenes("HIGH_STRESS", fls_normalized=0.5)
    assert "no confirman un deterioro transversal" in out
    assert "FLS:" not in out


def test_h15_fls_ausente_mantiene_texto_historico():
    out = _run_regimenes("HIGH_STRESS", fls_normalized=None)
    assert "no confirman un deterioro transversal" in out


def test_h15_umbral_desde_config():
    assert H15_UMBRAL == 70
