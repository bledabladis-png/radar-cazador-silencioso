# -*- coding: utf-8 -*-
"""Test M2: regla FLOW_CONFIDENCE y caso de empate 2-2.

Dictamen auditor externo 2026-10-16: la nota de la regla debe declarar
la banda neutral (|s| <= 0.1) y el caso de empate real 2-2.
"""
from src.report.flows_international import render_flujo_sintesis


def test_nota_declara_banda_neutral_y_empate():
    fs = {
        "flow_proxy_sign": 0.22,
        "etf_primary_flow_sign": -0.78,
        "cftc_flow_sign": 0.01,     # neutral (|s| <= 0.1)
        "european_flow_sign": -1.89,
        "confidence": "MEDIA",
    }
    out = "".join(render_flujo_sintesis(fs))
    assert "banda neutral" in out.lower() or "Banda neutral" in out
    assert "|s| <= 0.1" in out
    assert "Empate real 2-2" in out
    assert "SIN DIRECCION DOMINANTE" in out


def test_sintesis_sin_datos_no_falla():
    out = render_flujo_sintesis(None)
    assert out == []