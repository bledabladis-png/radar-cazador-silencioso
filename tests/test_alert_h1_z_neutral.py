# -*- coding: utf-8 -*-
"""Test M1 + H14 (2026-10-17): alerta H1 con banda neutral |Z| < 0.75.

Dictamen auditor externo 2026-10-16: con Z=-0.15 la frase
"el credito lo confirma" era excesivamente categorica. Se introduce
una banda neutral explicita documentada.
"""
from src.report.alerts import render_alerts


def _run(hyg_z):
    conf = {"ratios": {"hyg_lqd_zscore": hyg_z}}
    return "".join(render_alerts(
        breadth_values=None, financial_regime="HIGH_STRESS",
        price_flow_divergences=None, confirmation_data=conf,
    ))


def test_banda_neutral_z_negativo_pequeno():
    out = _run(-0.15)
    assert "proximo a neutral" in out
    assert "|Z|<0.75" in out
    assert "el credito lo confirma" not in out


def test_banda_neutral_z_positivo_pequeno():
    out = _run(0.30)
    assert "proximo a neutral" in out
    assert "Estres localizado" not in out


def test_rama_negativa_para_z_negativo_grande():
    out = _run(-2.50)
    assert "el credito lo confirma" in out
    assert "proximo a neutral" not in out


def test_rama_positiva_para_z_positivo_grande():
    out = _run(2.50)
    assert "Estres localizado" in out
    assert "proximo a neutral" not in out


def test_z_nd():
    out = _run(float("nan"))
    assert "no disponible" in out