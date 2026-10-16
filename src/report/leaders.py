# -*- coding: utf-8 -*-
"""Momentum, Tactical Leaders y Structural Ranking.

Extraidos de src/report_generator.py (refactor C1, fase C1-6c1).
"""

from config.settings import (EXPECTED_SECTOR_COUNT, MOMENTUM_PRICE_WINDOW,
                              TOP_N_CANDIDATES, TOP_N_LEADERS)
from config.weights import WLS_WEIGHTS, WLS_PERSISTENCE_FACTOR
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
        # P1.3 (2026-10-13, auditor): la nota anterior se contradecia
        # (4 componentes vs 'RS y RS Mom 35/25'). Corregida a la
        # formula real de stock_leader.py.
        out.append(
            f"*Criterio de seleccion: los {TOP_N_LEADERS} lideres por sector "
            f"se eligen en dos pasos: (1) los {TOP_N_CANDIDATES} valores del ETF "
            f"sectorial con mayor peso en cartera; (2) entre esos, los "
            f"{TOP_N_LEADERS} con mayor WLS.*\n\n"
        )
        # P1.3 (2026-10-13, auditor): texto derivado de WLS_WEIGHTS
        # y WLS_PERSISTENCE_FACTOR. Si cambian los pesos, la nota
        # se actualiza sola.
        _w = WLS_WEIGHTS
        out.append(
            f"*WLS (Weighted Leadership Score, universo sectorial) = "
            f"{_w['rs']:.2f} * rs_z + {_w['flow']:.2f} * flow_proxy_z + "
            f"{_w['wyckoff']:.2f} * wyckoff_z + {_w['stability']:.2f} * "
            f"stability_z, multiplicado por (1 + {WLS_PERSISTENCE_FACTOR:.2f} * "
            f"min(persistence_10d, 1.0)). Las componentes rs_z, "
            f"flow_proxy_z, wyckoff_z y stability_z son z-scores robustos "
            f"intra-sector, clipeados a [-3, +3].*\n\n"
        )
        out.append(
            # H14 (2026-10-16, auditor): reforzar la nota de no
            # comparabilidad y explicitar la decision de diseno.
            # La suma de pesos 0.95 es deliberada, no un error.
            "*RS = RS Level (ratio de precio accion / precio ETF sectorial). "
            "RS Mom = variacion logaritmica de RS a 20 dias. La tabla de "
            "'Indices internacionales' usa una variante de WLS con pesos "
            "0.35/0.25/0.25/0.10 por diseno (flow 25% en vez de 30%; la "
            "suma de pesos es 0.95, no 1.00). Decision metodologica "
            "deliberada, testeada en tests/test_index_leaders_h14_wls_weights.py. "
            "Los valores WLS de esta tabla NO son comparables directamente "
            "con los WLS sectoriales (suma 1.00): se restringen al universo "
            "de cada indice.*\n\n"
        )
        out.extend(leader_lines)
    else:
        out.append("\n## Acciones Seleccionadas por el Modelo de Liderazgo Sectorial\n")
        out.append("*No disponibles: ningun sector en fase de acumulación.*\n")

    return out
