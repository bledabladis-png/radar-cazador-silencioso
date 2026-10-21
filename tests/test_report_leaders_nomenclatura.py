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

# ============================================================
# REP-AUDIT-20261020-REPORT-FLOW-RECONCILIATION (P1 auditor externo
# 2026-10-20): la tabla de Flujo Proxy (composite ETF) y la matriz
# (mediana de componentes) son metricas distintas con nota.
# ============================================================

def test_p1_nota_flow_proxy_etf_vs_mediana():
    out = render_momentum_sectores(
        sector_price_rank=[('XLK', 0.01)],
        sector_flow_rank=[('XLK', 0.5)],
    )
    body = "".join(out)
    assert "composite momentum del ETF sectorial" in body
    assert "mediana del composite de los componentes" in body
    assert "pueden discrepar en signo" in body


# ============================================================
# H1 (2026-10-21): distinguir 'Structural' de 'Struct Wyckoff'
# ============================================================

def test_h1_rankings_renombra_struct_score_a_struct_wyckoff():
    from src.report.rankings import render_ranking_sectorial_unificado
    sr = {"ranking": [("XLP", "Staples", 0.6, "RANGE")],
          "struct_scores": {"XLP": 0.17}, "scores": None, "regime": "ROT",
          "last_scores": None}
    body = "".join(render_ranking_sectorial_unificado(
        sr, tactical_scores=None, structural_scores=None,
        sector_persistence=None, signal_agreements=None,
        signal_agreements_display=None, shock_sensitivities=None,
    ))
    assert "Struct Wyckoff" in body
    assert "Struct Score" not in body
    assert "motor estructural" in body
    assert "struct_score interno" in body


def test_h4_nota_trazabilidad_slpm():
    from src.report.leaders import render_acciones_seleccionadas
    out = render_acciones_seleccionadas(["fake_line\n"])
    body = "".join(out)
    assert "Nota de trazabilidad" in body
    assert "NO han pasado validacion estructural de SLPM" in body
    assert "stock_leader" in body


# ============================================================
# H3 (2026-10-21): nota de sesgo de seleccion en Liderazgo
# relativo interno (top-5 por sector).
# ============================================================

def test_h3_liderazgo_relativo_nota_sesgo_seleccion():
    import pandas as pd
    from src.report.market_context import render_liderazgo_interno
    df = pd.DataFrame([
        {"date": "2026-10-20", "sector": "XLE", "ticker": "CVX",
         "price_ret_20d": 0.10, "rs_abs_20d": 0.05, "rs_internal_20d": 0.03,
         "classification": "Liderazgo relativo doble"},
    ])
    body = "".join(render_liderazgo_interno(df))
    assert "sesgo de seleccion" in body
    assert "top-5" in body
    assert "universo completo" in body
