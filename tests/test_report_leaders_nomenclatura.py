# -*- coding: utf-8 -*-
"""Tests de regresion B1 (nomenclatura del render de flujo).

Contexto: los titulos "Flujo Institucional - Sectores / Otros Activos"
describian la metrica FLOW_PROXY (0.30*flow_smooth + 0.35*obv_z +
0.35*cmf_z) como flujo institucional. La nota de slpm.py:124 aclara
explicitamente "No implica flujo institucional real". Se renombro a
"Flujo de Mercado" para eliminar la contradiccion.

Este test BLOQUEA futuras reversiones al termino "Institucional".
"""

from src.report.leaders import (
    render_momentum_sectores,
    render_momentum_otros,
)


def test_b1_sectores_sin_palabra_institucional():
    """El titulo de sectores no debe contener 'Institucional'."""
    out = render_momentum_sectores(
        sector_price_rank=[('XLK', 0.01)],
        sector_flow_rank=[('XLK', 0.5)],
    )
    body = "".join(out)
    assert "Institucional" not in body
    assert "Flujo de Mercado - Sectores" in body


def test_b1_otros_activos_sin_palabra_institucional():
    """El titulo de otros activos no debe contener 'Institucional'."""
    out = render_momentum_otros(
        otros_price_rank=[('BZ=F', 0.02)],
        otros_flow_rank=[('BZ=F', 0.3)],
    )
    body = "".join(out)
    assert "Institucional" not in body
    assert "Flujo de Mercado - Otros Activos" in body


def test_b1_sectores_sin_datos_tambien_renombrado():
    """Rama 'sin datos' tambien debe usar el titulo nuevo."""
    out = render_momentum_sectores(
        sector_price_rank=[('XLK', 0.01)],
        sector_flow_rank=[],
    )
    body = "".join(out)
    assert "Institucional" not in body
    assert "Flujo de Mercado - Sectores" in body


def test_b1_otros_activos_sin_datos_tambien_renombrado():
    out = render_momentum_otros(
        otros_price_rank=[('BZ=F', 0.02)],
        otros_flow_rank=[],
    )
    body = "".join(out)
    assert "Institucional" not in body
    assert "Flujo de Mercado - Otros Activos" in body