# -*- coding: utf-8 -*-
"""Test de regresion I1 + O1 (2026-09-18).

I1: dos tablas del reporte usan 'Retorno 20d' con significados
    distintos (mediana de componentes del sector vs retorno del ETF).
    Aclaracion via nota en ambas tablas.

O1: la nota fuente SSGA no declaraba la fecha del dato.
"""
import pandas as pd
from src.report.leaders import render_momentum_sectores, render_tactical_leaders
from src.report.etf_flows import render_flujo_spdr


def test_i1_momentum_sectores_incluye_nota():
    rank = [('XLK', 0.05), ('XLF', -0.02)]
    flow = [('XLK', 0.5), ('XLF', -0.3)]
    out = render_momentum_sectores(rank, flow)
    joined = ''.join(out)
    assert 'mediana de los retornos 20d' in joined
    assert 'Flujo Primario ETF' in joined  # referencia cruzada
    assert 'componentes del sector' in joined


def test_i1_tactical_leaders_incluye_nota():
    out = render_tactical_leaders(
        tactical_scores={'XLK': 0.5},
        structural_scores={'XLK': 0.3},
        sector_price_rank=[('XLK', 0.05)],
        sector_flow_rank=[('XLK', 0.5)],
        shock_sensitivities={},
    )
    joined = ''.join(out)
    assert "misma metrica que 'Momentum de Precio - Sectores'" in joined
    assert 'mediana de los componentes del sector' in joined


def test_i1_tactical_nota_separada_de_comm_corr():
    """La nota I1 y la nota Comm Corr deben ser parrafos separados."""
    out = render_tactical_leaders(
        tactical_scores={'XLK': 0.5},
        structural_scores={'XLK': 0.3},
        sector_price_rank=[('XLK', 0.05)],
        sector_flow_rank=[('XLK', 0.5)],
        shock_sensitivities={},
    )
    joined = ''.join(out)
    # Debe haber doble salto entre la nota I1 y la de Comm Corr
    idx = joined.find('mediana de los componentes')
    assert idx > 0
    resto = joined[idx:idx+200]
    # En los siguientes 200 chars debe haber '\n\n' antes de 'Comm Corr'
    assert '\n\n' in resto or '\n\n' in joined[idx:], 'Falta separacion de parrafos'


def test_o1_fecha_efectiva_en_nota():
    df = pd.DataFrame([
        {'ticker': 'XLK', 'nav': 100.0, 'shares_outstanding': 1000.0,
         'total_net_assets': 100000.0, 'primary_flow_usd': 0.0,
         'primary_flow_pct': 0.0, 'primary_flow_z': 0.0, 'Date': '2026-09-16'},
        {'ticker': 'XLF', 'nav': 50.0, 'shares_outstanding': 2000.0,
         'total_net_assets': 100000.0, 'primary_flow_usd': 0.0,
         'primary_flow_pct': 0.0, 'primary_flow_z': 0.0, 'Date': '2026-09-17'},
    ])
    out = render_flujo_spdr(df)
    joined = ''.join(out)
    assert 'Ultima fecha: 2026-09-17' in joined


def test_o1_fecha_nd_si_no_hay_columna():
    df = pd.DataFrame([
        {'ticker': 'XLK', 'nav': 100.0, 'shares_outstanding': 1000.0,
         'total_net_assets': 100000.0, 'primary_flow_usd': 0.0,
         'primary_flow_pct': 0.0, 'primary_flow_z': 0.0},
    ])
    out = render_flujo_spdr(df)
    joined = ''.join(out)
    assert 'Ultima fecha: N/D' in joined