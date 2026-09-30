# -*- coding: utf-8 -*-
"""Fix F: alerts.py debe usar el mensaje del detector, no literal fijo.

Bug: render_alerts emitia "Precio fuerte" para CUALQUIER status != ALIGNED.
Con PRICE_WEAK_FLOW_SUPPORTIVE, el reporte decia "precio fuerte" cuando
el detector devolvia "precio debil".
"""
from src.report.alerts import render_alerts


def test_alerts_usa_message_de_price_weak():
    """PRICE_WEAK_FLOW_SUPPORTIVE -> mensaje del detector (precio debil)."""
    divs = {
        "XLU": {
            "status": "PRICE_WEAK_FLOW_SUPPORTIVE",
            "message": "Precio debil (-6.0%) con Flow Proxy positivo (z=+0.68).",
        }
    }
    out = render_alerts(breadth_values=None, liquidity_regime=None,
                        price_flow_divergences=divs)
    lineas = "".join(out)
    assert "Precio debil" in lineas, f"Esperado 'Precio debil', obtenido: {lineas[:200]}"
    assert "Precio fuerte" not in lineas, "No debe aparecer el literal viejo"


def test_alerts_usa_message_de_price_strong():
    """PRICE_STRONG_FLOW_UNCONFIRMED -> mensaje del detector (precio fuerte)."""
    divs = {
        "XLK": {
            "status": "PRICE_STRONG_FLOW_UNCONFIRMED",
            "message": "Precio fuerte (+7.0%) sin confirmacion del Flow Proxy.",
        }
    }
    out = render_alerts(breadth_values=None, liquidity_regime=None,
                        price_flow_divergences=divs)
    lineas = "".join(out)
    assert "Precio fuerte" in lineas
    assert "Precio debil" not in lineas


def test_alerts_ignora_aligned():
    """status=ALIGNED -> no genera alerta."""
    divs = {"XLK": {"status": "ALIGNED", "message": ""}}
    out = render_alerts(breadth_values=None, liquidity_regime=None,
                        price_flow_divergences=divs)
    lineas = "".join(out)
    assert "XLK" not in lineas