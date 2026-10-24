# -*- coding: utf-8 -*-
"""Momentum, Tactical Leaders y Structural Ranking.

Extraidos de src/report_generator.py (refactor C1, fase C1-6c1).
"""

from config.settings import EXPECTED_SECTOR_COUNT, MOMENTUM_PRICE_WINDOW
from config.weights import FLOW_PROXY_WEIGHTS
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
    # P1.2 (2026-10-13, auditor): la nota anterior afirmaba que el
    # retorno era la mediana de los componentes. INCORRECTO: el codigo
    # aplica close.pct_change(20) sobre el ETF sectorial (ver
    # regimes/sector_regime.py::compute_price_momentum). Los valores
    # coinciden con 'Flujo Primario ETF - Agregado sectorial' porque
    # ambos usan el mismo calculo. Nota corregida.
    out.append("\n*Nota: Retorno 20d es el retorno del ETF sectorial "
               "(close.pct_change(20)). Coincide con el retorno mostrado "
               "en 'Flujo Primario ETF - Agregado sectorial' por usar la "
               "misma fuente. Distinto de 'Retorno 20d (%)' de Concentracion "
               "del liderazgo, que mide la mediana de los componentes.*\n")

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
    if sector_flow_rank:
        # REP-AUDIT-20261020-REPORT-FLOW-RECONCILIATION (P1 auditor externo
        # 2026-10-20): esta tabla usa el composite momentum del ETF sectorial
        # (regimes/sector_regime.py: compute_flow_proxy sobre el ETF). La
        # columna 'Flujo Proxy' de la Matriz de Evidencia usa otra metrica
        # (mediana del composite de los componentes del sector, ver
        # sector_concentration.flow_median). Son metricas distintas y pueden
        # discrepar en signo. Ejemplo 2026-10-19: XLE -0.50 aqui, +1 en matriz.
        _fp = FLOW_PROXY_WEIGHTS
        out.append(f"\n*Nota: esta tabla usa el composite momentum del ETF "
                   f"sectorial ({_fp['flow_smooth']:.2f}*flow_smooth + "
                   f"{_fp['obv']:.2f}*obv_z + "
                   f"{_fp['cmf']:.2f}*cmf_z). "
                   f"La columna 'Flujo Proxy' de la Matriz de Evidencia usa la "
                   f"mediana del composite de los componentes del sector. "
                   f"Son metricas distintas y pueden discrepar en signo.*\n")
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
