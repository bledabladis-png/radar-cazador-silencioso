# -*- coding: utf-8 -*-
"""Rankings Sectoriales, Persistencia y Opportunity Map.

Extraidos de src/report_generator.py (refactor C1, fase C1-6c2).
"""

import numpy as np

from config.tickers import SECTOR_NAMES, MARKET_TICKERS
from config.settings import EXPECTED_SECTOR_COUNT
from src.utils import safe_mean
from src.report.helpers import _fmt_num
from src.report.helpers import _fmt_signed


def render_ranking_sectorial_unificado(sector_results, tactical_scores,
                                       structural_scores, sector_persistence,
                                       signal_agreements,
                                       signal_agreements_display,
                                       shock_sensitivities):
    """Ranking sectorial unificado: 1 tabla con todas las columnas.

    Fusion de "Structural Ranking" y "Rankings Sectoriales" (antes
    dos secciones separadas con 5 columnas comunes). Ordenado por
    el Score combinado (el criterio mas informativo disponible).

    Mantiene el titulo "Rankings Sectoriales" porque esta anclado
    por test_report_generator_smoke.py.
    """
    out = []
    out.append("## Rankings Sectoriales\n")
    out.append("> *Nota: Score es el ranking historico del sistema "
               "(momentum, tendencia, volatilidad, breadth, Wyckoff). "
               "Tactical y Structural son los motores independientes.*\n\n")
    header = ("| # | Sector | Score | Tactical | Structural | Persist "
              "| Agreement | Signal Cons | Comm Corr | Fase Wyckoff |\n")
    sep = ("|---|--------|-------|----------|------------|---------"
           "|-----------|-------------|-----------|---------------|\n")
    out.append(header)
    out.append(sep)
    for i, (ticker, name, score, wyckoff) in enumerate(
            sector_results['ranking'][:EXPECTED_SECTOR_COUNT], 1):
        t_score = tactical_scores.get(ticker, 0.0) if tactical_scores else 0.0
        s_score = structural_scores.get(ticker, 0.0) if structural_scores else 0.0
        pers_raw = sector_persistence.get(ticker) if sector_persistence else None
        pers_val = pers_raw if pers_raw is not None else 0.0
        pers_str = f"{pers_raw:.0%}" if pers_raw is not None else "N/A"
        agree = signal_agreements.get(ticker, 0.5) if signal_agreements else 0.5
        agree_display = (signal_agreements_display.get(ticker, f'{agree:.0%}')
                         if signal_agreements_display else f'{agree:.0%}')
        signal_cons = (pers_val + agree) / 2
        shock = shock_sensitivities.get(ticker, {}) if shock_sensitivities else {}
        comm_level = shock.get('commodity_level', 'N/A') if shock else 'N/A'
        comm_val = shock.get('commodity_corr_value', None) if shock else None
        comm_display = (f"{comm_level} ({_fmt_signed(comm_val, '{:+.2f}', '{:.2f}')})"
                        if comm_val is not None and comm_level != 'N/A'
                        else comm_level)
        out.append(
            f"| {i} | {name} ({ticker}) | {_fmt_num(score, '{:.2f}')} "
            f"| {_fmt_signed(t_score, '{:+.2f}', '{:.2f}')} "
            f"| {_fmt_signed(s_score, '{:+.2f}', '{:.2f}')} "
            f"| {pers_str} | {agree_display} | {signal_cons:.0%} "
            f"| {comm_display} | {wyckoff} |\n"
        )
    out.append("\n")
    return out


def render_opportunity_map(tactical_scores, structural_scores,
                           sector_persistence, signal_agreements):
    """Renderiza la tabla Opportunity Map por cuadrantes.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    out.append("## Opportunity Map (basado en medianas Tactical/Structural, independiente del SLPM)\n\n")
    tact_values = [v for v in tactical_scores.values() if v is not None] if tactical_scores else [0]
    struct_values = [v for v in structural_scores.values() if v is not None] if structural_scores else [0]
    tact_median = np.median(tact_values) if tact_values else 0
    struct_median = np.median(struct_values) if struct_values else 0
    
    out.append(f"*Umbrales del dia: Tactical mediana={_fmt_signed(tact_median, '{:+.2f}', '{:.2f}')}, Structural mediana={_fmt_signed(struct_median, '{:+.2f}', '{:.2f}')}*\n\n")
    out.append("| Cuadrante | Sectores | Signal Consistency |\n")
    out.append("|-----------|----------|------------|\n")
    
    quadrants = {
        'Structural Strength': [],
        'Tactical Correction': [],
        'Tactical Strength': [],
        'Structural Weakness': [],
        'Transition': []
    }
    
    for ticker in MARKET_TICKERS['sectors']:
        t = tactical_scores.get(ticker, 0) if tactical_scores else 0
        s = structural_scores.get(ticker, 0) if structural_scores else 0
        name = SECTOR_NAMES.get(ticker, ticker)
        pers_raw = sector_persistence.get(ticker) if sector_persistence else None
        pers_val = pers_raw if pers_raw is not None else 0.0
        agree = signal_agreements.get(ticker, 0.5) if signal_agreements else 0.5
        conf = (pers_val + agree) / 2
        
        if s > struct_median and t > tact_median:
            quadrants['Structural Strength'].append((name, conf))
        elif s > struct_median and t < tact_median:
            quadrants['Tactical Correction'].append((name, conf))
        elif s < struct_median and t > tact_median:
            quadrants['Tactical Strength'].append((name, conf))
        elif s < struct_median and t < tact_median:
            quadrants['Structural Weakness'].append((name, conf))
        else:
            # Valores exactamente en la mediana se clasifican como Transition
            quadrants['Transition'].append((name, conf))
    
    icons = {
        'Structural Strength': 'VERDE',
        'Tactical Correction': 'AMARILLO',
        'Tactical Strength': 'AZUL',
        'Structural Weakness': 'ROJO',
        'Transition': 'GRIS'
    }
    for quadrant, sector_list in quadrants.items():
        icon = icons.get(quadrant, '?')
        if sector_list:
            sector_names = [s[0] for s in sector_list]
            avg_conf = safe_mean([s[1] for s in sector_list])
            out.append(f"| {icon} **{quadrant}** | {', '.join(sector_names)} | {avg_conf:.0%} |\n")
        else:
            out.append(f"| {icon} **{quadrant}** | -- | -- |\n")
    out.append("\n")
    out.append("*Nota: 'Structural Strength' en Opportunity Map identifica posicion relativa en el eje Structural. No implica liderazgo confirmado por SLPM.*\n\n")
    return out
