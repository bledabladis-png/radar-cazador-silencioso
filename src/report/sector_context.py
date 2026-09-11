# -*- coding: utf-8 -*-
"""Contexto sectorial: matriz de regimen, representatividad, distribucion
Wyckoff, divergencia sector-lideres y momentum de amplitud.

Extraidos de src/report_generator.py (refactor C1, fase C1-6d3c).
"""

import pandas as pd


def render_matriz_regimen(sector_regime_matrix_data):
    """Renderiza la tabla Matriz de Regimen Sectorial.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_regime_matrix_data is not None and not sector_regime_matrix_data.empty:
        out.append("## Matriz de Régimen Sectorial\n")
        out.append("| Sector | Precio 20d | % > EMA50 | Flujo 20d | Fase Wyckoff | Positivas | Lectura |\n")
        out.append("|--------|------------|-----------|-----------|--------------|-----------|---------|\n")
        for _, row in sector_regime_matrix_data.iterrows():
            out.append(f"| {row['sector']} | {row['price_ret_20d']:.2%} | {row['pct_above_ema50']:.1f}% | {row['flow_20d_sum']:+,.0f} | {row['wyckoff_phase']} | {row['positive_conditions']:.0f} | {row['regime_reading']} |\n")
        out.append("\n")
    return out


def render_representatividad_lider(leader_representativeness_data):
    """Renderiza la tabla Representatividad del lider.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if leader_representativeness_data is not None and not leader_representativeness_data.empty:
        out.append("## Representatividad del líder\n")
        out.append("| Sector | Líder | RS ΔMed | Mom ΔMed | Flow ΔMed | WLS ΔMed | Rank pct |\n")
        out.append("|--------|-------|---------|----------|-----------|----------|----------|\n")
        for _, row in leader_representativeness_data.iterrows():
            out.append(f"| {row['sector']} | {row['ticker']} | {row['rs_distance_to_median']:+.4f} | {row['mom_distance_to_median']:+.4f} | {row['flow_distance_to_median']:+.2f} | {row['wls_distance_to_median']:+.2f} | {row['sector_rank_pct']:.0%} |\n")
        out.append("\n")
    return out


def render_wyckoff_sectorial(sector_wyckoff_distribution_data):
    """Renderiza la tabla Distribucion Wyckoff sectorial.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_wyckoff_distribution_data is not None and not sector_wyckoff_distribution_data.empty:
        out.append("## Distribución Wyckoff sectorial\n")
        out.append("| Sector | Acc | Markup | Range | Dist | Markdown | N | Cobertura |\n")
        out.append("|--------|-----|--------|-------|------|----------|---|-----------|\n")
        # Mostrar solo la última fecha para evitar duplicados históricos
        df_wy = sector_wyckoff_distribution_data.copy()
        if 'date' in df_wy.columns and df_wy['date'].notna().any():
            latest = pd.to_datetime(df_wy['date']).max()
            df_wy = df_wy[pd.to_datetime(df_wy['date']) == latest]
        for _, row in df_wy.iterrows():
            out.append(f"| {row['sector']} | {row['pct_accumulation']:.0f}% | {row['pct_markup']:.0f}% | {row['pct_range']:.0f}% | {row['pct_distribution']:.0f}% | {row['pct_markdown']:.0f}% | {row['n_valid_wyckoff']} | {row['coverage_wyckoff']:.0f}% |\n")
        out.append("\n")
    return out


def render_divergencia_sector_lideres(sector_leader_divergence_data):
    """Renderiza la tabla Divergencia sector-lideres.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_leader_divergence_data is not None and not sector_leader_divergence_data.empty:
        out.append("## Divergencia sector-líderes\n")
        out.append("| Sector | Ret sector | Líderes + | Líderes - | Líderes > Sector | Válidos | Lectura |\n")
        out.append("|--------|------------|-----------|-----------|------------------|---------|---------|\n")
        for _, row in sector_leader_divergence_data.iterrows():
            out.append(f"| {row['sector']} | {row['sector_ret_20d']:.2%} | {row['n_leaders_positive']} | {row['n_leaders_negative']} | {row['n_leaders_beating_sector']} | {row['n_leaders_valid']} | {row['classification']} |\n")
        out.append("\n")
    return out


def render_momentum_amplitud(sector_breadth_momentum_data):
    """Renderiza la tabla Momentum de amplitud.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_breadth_momentum_data is not None and not sector_breadth_momentum_data.empty:
        out.append("## Momentum de amplitud\n")
        out.append("| Sector | Δ1d EMA20 | Δ5d EMA20 | Δ20d EMA20 | Δ5d EMA50 | Δ5d EMA200 | Expansión | Deterioro |\n")
        out.append("|--------|-----------|-----------|------------|-----------|------------|-----------|-----------|\n")
        # Mostrar solo la última fecha para evitar duplicados históricos
        df_mom = sector_breadth_momentum_data.copy()
        if 'date' in df_mom.columns and df_mom['date'].notna().any():
            latest = pd.to_datetime(df_mom['date']).max()
            df_mom = df_mom[pd.to_datetime(df_mom['date']) == latest]
        for _, row in df_mom.iterrows():
            out.append(f"| {row['sector']} | {row['delta_1d_ema20']:+.1f} | {row['delta_5d_ema20']:+.1f} | {row['delta_20d_ema20']:+.1f} | {row['delta_5d_ema50']:+.1f} | {row['delta_5d_ema200']:+.1f} | {row['breadth_expansion_5d']} | {row['breadth_deterioration_5d']} |\n")
        out.append("\n")
    return out
