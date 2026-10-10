# -*- coding: utf-8 -*-
"""Momentum, Tactical Leaders y Structural Ranking.

Extraidos de src/report_generator.py (refactor C1, fase C1-6c1).
"""

from config.settings import (EXPECTED_SECTOR_COUNT, MOMENTUM_PRICE_WINDOW,
                              TOP_N_CANDIDATES, TOP_N_LEADERS)
from config.tickers import SECTOR_NAMES
from src.report.helpers import _fmt_num


def render_momentum_sectores(sector_price_rank, sector_flow_rank):
    """Renderiza momentum de precio e institucional (sectores).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    out.append(f"\n## Momentum de Precio - Sectores ({MOMENTUM_PRICE_WINDOW} dias)\n")
    out.append("| # | Sector | Retorno 20d (%) |\n")
    out.append("|---|--------|------------------|\n")
    for i, (ticker, mom) in enumerate(sector_price_rank[:EXPECTED_SECTOR_COUNT], 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        out.append(f"| {i} | {name} ({ticker}) | {_fmt_num(mom*100, '{:.2f}%')} |\n")

    # I1 (2026-09-18): aclarar metrica para evitar confusion con otras
    # secciones que usan "Retorno 20d" para el retorno del ETF sectorial.
    out.append("\n*Nota: Retorno 20d mide la mediana de los retornos 20d "
               "de los componentes del sector. Distinta del retorno del ETF "
               "sectorial mostrado en 'Flujo Primario ETF - Agregado sectorial'.*\n")

    if sector_flow_rank:
        out.append("\n## Flujo de Mercado - Sectores (Proxy)\n")
        out.append("| # | Sector | Flujo (z-score) |\n")
        out.append("|---|--------|------------------|\n")
    else:
        out.append("\n## Flujo de Mercado - Sectores (Proxy)\n")
        out.append("*No hay datos disponibles para Flow Proxy.*\n")
    for i, (ticker, flow) in enumerate(sector_flow_rank[:EXPECTED_SECTOR_COUNT], 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        out.append(f"| {i} | {name} ({ticker}) | {_fmt_num(flow, '{:.2f}')} |\n")
    return out


def render_momentum_otros(otros_price_rank, otros_flow_rank):
    """Renderiza momentum de precio e institucional (otros activos).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    out.append(f"\n## Momentum de Precio - Otros Activos ({MOMENTUM_PRICE_WINDOW} dias)\n")
    out.append("| # | Activo | Retorno 20d (%) |\n")
    out.append("|---|--------|------------------|\n")
    for i, (ticker, mom) in enumerate(otros_price_rank[:15], 1):
        out.append(f"| {i} | {ticker} | {_fmt_num(mom*100, '{:.2f}%')} |\n")

    out.append("\n*Nota metodologica (commodities): BZ=F, CL=F, GC=F, HG=F "
               "y NG=F usan 'Close' del front-month correspondiente "
               "(Yahoo Finance). No es settlement oficial de exchange.*\n\n")

    if otros_flow_rank:
        out.append("\n## Flujo de Mercado - Otros Activos (Proxy)\n")
        out.append("| # | Activo | Flujo (z-score) |\n")
        out.append("|---|--------|------------------|\n")
    else:
        out.append("\n## Flujo de Mercado - Otros Activos (Proxy)\n")
        out.append("*No hay datos disponibles para Flow Proxy.*\n")
    for i, (ticker, flow) in enumerate(otros_flow_rank[:15], 1):
        out.append(f"| {i} | {ticker} | {_fmt_num(flow, '{:.2f}')} |\n")
    return out


def render_acciones_seleccionadas(leader_lines):
    """Renderiza la seccion Acciones Seleccionadas por el Modelo.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if leader_lines:
        out.append("\n## Acciones Seleccionadas por el Modelo de Liderazgo Sectorial\n")
        out.append("> Solo se muestran sectores en fase ACCUMULATION o MARKUP. El resto se omiten por no cumplir criterios de liderazgo estructural.\n\n")
        out.append(
            f"*Criterio de seleccion: los {TOP_N_LEADERS} lideres por sector "
            f"se eligen en dos pasos: (1) los {TOP_N_CANDIDATES} valores del ETF "
            f"sectorial con mayor peso en cartera; (2) entre esos, los "
            f"{TOP_N_LEADERS} con mayor WLS (score compuesto 35% RS Level + "
            f"30% Flujo proxy + 25% Wyckoff + 10% Estabilidad, ajustado por "
            f"persistencia 10d).*\n\n"
        )
        out.append('*RS = RS Level (precio acción / precio sector). RS Mom = RS Momentum (cambio del RS en 20 días). El WLS combina ambas con pesos 35% y 25% respectivamente.*\n\n')
        out.extend(leader_lines)
    else:
        out.append("\n## Acciones Seleccionadas por el Modelo de Liderazgo Sectorial\n")
        out.append("*No disponibles: ningun sector en fase de acumulación.*\n")

    return out
