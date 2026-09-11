# -*- coding: utf-8 -*-
"""Momentum, Tactical Leaders y Structural Ranking.

Extraidos de src/report_generator.py (refactor C1, fase C1-6c1).
"""

from config.settings import MOMENTUM_PRICE_WINDOW, MOMENTUM_LONG_WINDOW
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
    for i, (ticker, mom) in enumerate(sector_price_rank[:11], 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        out.append(f"| {i} | {name} ({ticker}) | {mom*100:.2f}% |\n")

    if sector_flow_rank:
        out.append("\n## Flujo Institucional - Sectores (Proxy)\n")
        out.append("| # | Sector | Flujo (z-score) |\n")
        out.append("|---|--------|------------------|\n")
    else:
        out.append("\n## Flujo Institucional - Sectores (Proxy)\n")
        out.append("*No hay datos disponibles para Flow Proxy.*\n")
    for i, (ticker, flow) in enumerate(sector_flow_rank[:11], 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        out.append(f"| {i} | {name} ({ticker}) | {flow:.2f} |\n")
    return out


def render_tactical_leaders(tactical_scores, structural_scores,
                            sector_price_rank, sector_flow_rank,
                            shock_sensitivities):
    """Renderiza la tabla Tactical Leaders.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    out.append("## Tactical Leaders (Momentum de corto plazo)\n")
    out.append("| # | Sector | Tactical | Structural | Retorno 20d | Flow Proxy (z) | Comm Corr |\n")
    out.append("|---|--------|----------|------------|-------------|----------------|------------|\n")
    tactical_ranking = sorted(tactical_scores.items(), key=lambda x: x[1], reverse=True) if tactical_scores else []
    for i, (ticker, t_score) in enumerate(tactical_ranking[:11], 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        s_score = structural_scores.get(ticker, 0.0) if structural_scores else 0.0
        mom = next((m for t, m in sector_price_rank if t == ticker), 0)
        flow = next((f for t, f in sector_flow_rank if t == ticker), None)
        shock = shock_sensitivities.get(ticker, {}) if shock_sensitivities else {}
        comm = shock.get('commodity_level', 'N/A') if shock else 'N/A'
        comm_val = shock.get('commodity_corr_value', None) if shock else None
        comm_display = f"{comm} ({comm_val:+.2f})" if comm_val is not None and comm != 'N/A' else comm
        out.append(f"| {i} | {name} ({ticker}) | {t_score:+.2f} | {s_score:+.2f} | {mom*100:.2f}% | {_fmt_num(flow, '{:+.2f}')} | {comm_display} |\n")
    out.append("\n")
    out.append(f"*Nota: Comm Corr mide la correlación de {MOMENTUM_LONG_WINDOW} dias con ^SPGSCI. No implica causalidad.*\n\n")
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
        out.append(f"| {i} | {ticker} | {mom*100:.2f}% |\n")

    if otros_flow_rank:
        out.append("\n## Flujo Institucional - Otros Activos (Proxy)\n")
        out.append("| # | Activo | Flujo (z-score) |\n")
        out.append("|---|--------|------------------|\n")
    else:
        out.append("\n## Flujo Institucional - Otros Activos (Proxy)\n")
        out.append("*No hay datos disponibles para Flow Proxy.*\n")
    for i, (ticker, flow) in enumerate(otros_flow_rank[:15], 1):
        out.append(f"| {i} | {ticker} | {flow:.2f} |\n")
    return out


def render_structural_ranking(structural_scores, tactical_scores,
                              sector_persistence, signal_agreements,
                              signal_agreements_display):
    """Renderiza la tabla Structural Ranking.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    out.append("## Structural Ranking (Fortaleza de largo plazo)\n")
    out.append("| # | Sector | Structural | Tactical | Persist | Agreement | Signal Consistency |\n")
    out.append("|---|--------|------------|----------|---------|-----------|------------|\n")
    structural_ranking = sorted(structural_scores.items(), key=lambda x: x[1], reverse=True) if structural_scores else []
    for i, (ticker, s_score) in enumerate(structural_ranking[:11], 1):
        name = SECTOR_NAMES.get(ticker, ticker)
        t_score = tactical_scores.get(ticker, 0.0) if tactical_scores else 0.0
        pers_raw = sector_persistence.get(ticker) if sector_persistence else None
        pers_val = pers_raw if pers_raw is not None else 0.0
        pers_str = f"{pers_raw:.0%}" if pers_raw is not None else "N/A"
        agree = signal_agreements.get(ticker, 0.5) if signal_agreements else 0.5
        agree_display = signal_agreements_display.get(ticker, f'{agree:.0%}') if signal_agreements_display else f'{agree:.0%}'
        struct_conf = (pers_val + agree) / 2
        out.append(f"| {i} | {name} ({ticker}) | {s_score:+.2f} | {t_score:+.2f} | {pers_str} | {agree_display} | {struct_conf:.0%} |\n")
    out.append("\n")
    return out
