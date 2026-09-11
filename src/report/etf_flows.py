# -*- coding: utf-8 -*-
"""Flujos ETF primarios: SPDR + Caracteristicas + Divergencia.

Extraidos de src/report_generator.py (refactor C1, fase C1-6d3a).
"""

import pandas as pd

from config.settings import ETF_PRIMARY_FLOW_ZSCORE_WINDOW
from src.report.helpers import _fmt_num


def render_flujo_spdr(etf_primary_flow_data):
    """Renderiza la tabla de Flujo Primario ETF (SPDR).

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if etf_primary_flow_data is not None and not etf_primary_flow_data.empty:
        out.append("## Flujo Primario ETF (SPDR)\n")
        out.append("| Ticker | NAV | Shares Outstanding | Total Net Assets | Primary Flow $ | Flow % AUM | Flow Z |\n")
        out.append("|--------|-----|---------------------|------------------|----------------|------------|--------|\n")
        for _, row in etf_primary_flow_data.iterrows():
            out.append(f"| {row['ticker']} | {row['nav']:.2f} | {row['shares_outstanding']:,.0f} | {row['total_net_assets']:,.0f} | {row['primary_flow_usd']:+,.2f} | {row['primary_flow_pct']:+.2f}% | {row['primary_flow_z']:+.2f} |\n")
        out.append(f"\n*Fuente: State Street Global Advisors (SSGA). ETF Primary Flow = ΔShares Outstanding × NAV. Z-score sobre {ETF_PRIMARY_FLOW_ZSCORE_WINDOW} sesiones.*\n\n")
    return out


def render_flujo_caracteristicas(sector_flow_characteristics_data):
    """Renderiza la tabla Flujo Primario ETF - Caracteristicas.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_flow_characteristics_data is not None and not sector_flow_characteristics_data.empty:
        latest_date = pd.to_datetime(sector_flow_characteristics_data['date']).max()
        flow_latest = sector_flow_characteristics_data[pd.to_datetime(sector_flow_characteristics_data['date']) == latest_date]
        out.append("## Flujo Primario ETF — Características\n")
        out.append("| Sector | Flujo $ | % AUM | Z | 5d Acum | 20d Acum | Pers 5d | Pers 20d | Ret 20d | Régimen |\n")
        out.append("|--------|---------|-------|----|---------|----------|---------|----------|---------|----------|\n")
        for _, row in flow_latest.iterrows():
            regime = row.get('price_flow_regime', None)
            regime_str = regime if pd.notna(regime) else 'N/D'
            out.append(f"| {row['sector']} | {_fmt_num(row['flow_dollar'], '{:+,.2f}')} | {_fmt_num(row['flow_pct_aum'], '{:.2f}%')} | {_fmt_num(row['flow_zscore'], '{:.2f}')} | {_fmt_num(row['flow_5d_sum'], '{:+,.2f}')} | {_fmt_num(row['flow_20d_sum'], '{:+,.2f}')} | {_fmt_num(row['persistence_5d'], '{:.0%}')} | {_fmt_num(row['persistence_20d'], '{:.0%}')} | {_fmt_num(row['price_ret_20d'], '{:.2%}')} | {regime_str} |\n")
        out.append("\n")
    return out


def render_divergencia_precio_flujo(sector_flow_characteristics_data):
    """Renderiza la tabla Divergencia Precio-Flujo Primario.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_flow_characteristics_data is not None and not sector_flow_characteristics_data.empty:
        _flow_filtered = sector_flow_characteristics_data[pd.to_datetime(sector_flow_characteristics_data['date']) == pd.to_datetime(sector_flow_characteristics_data['date']).max()]
        out.append("## Divergencia Precio–Flujo Primario\n")
        out.append("| Sector | Ret 5d | Flujo 5d | Régimen 5d | Ret 20d | Flujo 20d | Régimen 20d |\n")
        out.append("|--------|--------|----------|------------|---------|-----------|-------------|\n")
        for _, row in _flow_filtered.iterrows():
            out.append(f"| {row['sector']} | {row['price_ret_5d']:.2%} | {row['flow_5d_sum']:+,.0f} | {row['price_flow_regime_5d']} | {row['price_ret_20d']:.2%} | {row['flow_20d_sum']:+,.0f} | {row['price_flow_regime_20d']} |\n")
        out.append("\n")
        out.append("*«Absorción potencial» describe una configuración de retorno negativo del precio acompañada de flujo primario acumulado positivo. Puede ser compatible con absorción, pero no confirma por sí sola absorción institucional ni establece causalidad.*\n\n")

    return out
