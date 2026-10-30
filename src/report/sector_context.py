# -*- coding: utf-8 -*-
"""Contexto sectorial: matriz de regimen, representatividad, distribucion
Wyckoff, divergencia sector-lideres y momentum de amplitud.

Extraidos de src/report_generator.py (refactor C1, fase C1-6d3c).
"""

import pandas as pd
from src.report.helpers import _fmt_signed
from config.settings import WYCKOFF_MIN_COVERAGE_PCT


def render_matriz_regimen(sector_regime_matrix_data):
    """Renderiza la tabla Matriz de Regimen Sectorial.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_regime_matrix_data is not None and not sector_regime_matrix_data.empty:
        out.append("## Matriz de Régimen Sectorial\n")
        out.append("| Sector | Precio 20d | % > EMA50 | Flujo 20d | Fase dominante | Struct | N est | Positivas | Lectura |\n")
        out.append("|--------|------------|-----------|-----------|--------------|--------|-------|-----------|---------|\n")
        for _, row in sector_regime_matrix_data.iterrows():
            _sr = row.get('structure_ratio')
            _sr_str = "N/D" if (_sr is None or pd.isna(_sr)) else f"{float(_sr):.2f}"
            _ne = row.get('n_estructuras', 0)
            _ne_str = str(int(_ne)) if pd.notna(_ne) else "0"
            out.append(f"| {row['sector']} | {row['price_ret_20d']:.2%} | {row['pct_above_ema50']:.1f}% | {_fmt_signed(row['flow_20d_sum'], '{:+,.0f}', '{:,.0f}')} | {row['wyckoff_phase']} | {_sr_str} | {_ne_str} | {row['positive_conditions']:.0f} | {row['regime_reading']} |\n")
        out.append("\n")
        out.append("*Struct = proporcion de holdings con estructura alcista clara*\n")
        out.append("*(ACC + ACC_FORMING + MARKUP) / (todas las fases claras).*\n")
        out.append("*N est = numero de holdings con estructura clara (no-RANGE, no-CONFLICT).*\n")
        out.append("*Fase dominante = fase con mayor porcentaje de holdings del sector "
                   "(pct_accumulation / pct_markup / pct_range / pct_distribution / "
                   "pct_markdown). NO es la fase del ETF sectorial (esa se retiro en "
                   "sesion 3): es la moda de la distribucion de fases de los holdings. "
                   "Un sector puede tener fase dominante RANGE y mayoria de holdings "
                   "en distribucion: son cosas distintas.*\n")
        out.append("*NO es senal predictiva de retorno. Describe estado estructural actual.*\n")
        out.append("*Struct=1.00 con N est=1 es fragil; Struct=0.60 con N est=15 es robusto.*\n")
        out.append("*H-01 auditor externo 2026-10-25: periodo del historico Wyckoff*\n")
        out.append("*= 2022-08-11 a hoy (aprox. 4.2 anos, 1086 sesiones), NO 10 anos.*\n\n")
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
        # C2 fix 2026-09-24: filtrar a la ultima fecha para no concatenar dias
        df_rep = leader_representativeness_data.copy()
        if 'date' in df_rep.columns and df_rep['date'].notna().any():
            latest = pd.to_datetime(df_rep['date']).max()
            df_rep = df_rep[pd.to_datetime(df_rep['date']) == latest]
        for _, row in df_rep.iterrows():
            out.append(f"| {row['sector']} | {row['ticker']} | {_fmt_signed(row['rs_distance_to_median'], '{:+.4f}', '{:.4f}')} | {_fmt_signed(row['mom_distance_to_median'], '{:+.4f}', '{:.4f}')} | {_fmt_signed(row['flow_distance_to_median'], '{:+.2f}', '{:.2f}')} | {_fmt_signed(row['wls_distance_to_median'], '{:+.2f}', '{:.2f}')} | {row['sector_rank_pct']:.0%} |\n")
        out.append("\n")
        out.append("*Criterio: los tickers del sector ordenados por WLS desc. La primera fila de cada sector es el 'lider por WLS'. No coincide con Concentracion del liderazgo ni con Liderazgo interno, que priorizan por retorno 20d.*\n\n")
        # F6-12 (2026-09-28): nota sobre el filtro de sectores.
        out.append("*Solo se muestran los sectores en fase ACCUMULATION o MARKUP. El resto (RANGE, DISTRIBUTION, MARKDOWN) se omite porque no genera seccion de lideres. Filtro por fase, no por cobertura ni rendimiento.*\n\n")
    return out


# P2.2 (2026-10-13, auditor): la distribucion de fases por sector
# se calcula sobre los componentes con datos suficientes. Coberturas
# por debajo del 50% se anotan explicitamente. Regla: cobertura < 50%
# -> la lectura sectorial es indicativa, no representativa.
def render_wyckoff_sectorial(sector_wyckoff_distribution_data):
    """Renderiza la tabla Distribucion Wyckoff sectorial.

    Devuelve lista de lineas markdown. Sin side effects.
    """
    out = []
    if sector_wyckoff_distribution_data is not None and not sector_wyckoff_distribution_data.empty:
        out.append("## Distribución Wyckoff sectorial\n")
        out.append("| Sector | AccF | Acc | Markup | Range | DistF | Dist | Markdown | Conflict | N | Cob Wyckoff |\n")
        out.append("|--------|------|-----|--------|-------|-------|------|----------|----------|---|-----------|\n")
        # Mostrar solo la última fecha para evitar duplicados históricos
        df_wy = sector_wyckoff_distribution_data.copy()
        if 'date' in df_wy.columns and df_wy['date'].notna().any():
            latest = pd.to_datetime(df_wy['date']).max()
            df_wy = df_wy[pd.to_datetime(df_wy['date']) == latest]
        _hay_baja = False
        for _, row in df_wy.iterrows():
            _cov = row['coverage_wyckoff']
            _cov_label = f"{_cov:.0f}%"
            if pd.notna(_cov) and _cov < WYCKOFF_MIN_COVERAGE_PCT:
                _cov_label += " [COB BAJA]"
                _hay_baja = True
            _conflict_pct = row.get('pct_conflict', float('nan'))
            _conflict_str = f"{_conflict_pct:.0f}%" if pd.notna(_conflict_pct) else "-"
            _accf_pct = row.get('pct_accumulation_forming', float('nan'))
            _distf_pct = row.get('pct_distribution_forming', float('nan'))
            _accf_str = f"{_accf_pct:.0f}%" if pd.notna(_accf_pct) else "-"
            _distf_str = f"{_distf_pct:.0f}%" if pd.notna(_distf_pct) else "-"
            out.append(f"| {row['sector']} | {_accf_str} | {row['pct_accumulation']:.0f}% | {row['pct_markup']:.0f}% | {row['pct_range']:.0f}% | {_distf_str} | {row['pct_distribution']:.0f}% | {row['pct_markdown']:.0f}% | {_conflict_str} | {row['n_valid_wyckoff']} | {_cov_label} |\n")
        out.append("\n")
        if _hay_baja:
            out.append(f"*[COB BAJA] = cobertura de componentes < "
                       f"{WYCKOFF_MIN_COVERAGE_PCT:.0f}%. Los "
                       "porcentajes de fase se calculan sobre los componentes "
                       "con datos suficientes (>=60 sesiones). Una cobertura "
                       "baja indica que la distribucion no es representativa "
                       "del sector completo. En esos casos la fase sectorial "
                       "del ranking puede diferir de la fase dominante aqui.*\n\n")
    # H6 (2026-10-21): nota P2.2 debe salir siempre, no solo si hay
    # COB BAJA. La fase sectorial del ranking se decide sobre el ETF,
    # no sobre estos componentes. Sin esta nota, un sector MARKUP con
    # mayoria de componentes RANGE parece contradictorio.
    out.append("*AccF / DistF = fase en formacion: rango identificado con contexto "
               "de acumulacion o distribucion, pero sin SOS/SOW confirmado todavia. "
               "Es señal temprana, no confirmacion.*\n\n")
    out.append("*Conflict = el activo esta simultaneamente dentro de un rango de "
               "distribucion y de uno de acumulacion (precio dentro de ambos). "
               "Es un estado honesto: dos estructuras compitiendo. No se "
               "elige una sin mas evidencia.*\n\n")
    out.append("*La fase sectorial (Rankings) "
               "se decide sobre el ETF, no sobre estos componentes. "
               "Un sector en fase MARKUP puede tener mayoria de "
               "componentes en RANGE: la fase sectorial y la distribucion "
               "de fases por componente miden cosas distintas.*\n\n")
    # P1-H04 dictamen 2026-10-26: la serie historica se ha regenerado con
    # la composicion 2026Q4 de cada ETF aplicada retrospectivamente. No es
    # reconstruccion point-in-time. Debe ser visible en el reporte, no solo
    # en doc de auditoria.
    out.append("*Composicion de holdings: la serie historica "
               "(2022-08-11 a la fecha de observacion) se ha regenerado "
               "con la composicion 2026Q4 de cada ETF aplicada "
               "retrospectivamente. No es reconstruccion point-in-time. "
               "El comportamiento mostrado es el de los componentes "
               "actuales, no el de la composicion historica de cada "
               "fecha.*\n\n")
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
        # C3 fix 2026-09-24: filtrar a la ultima fecha para no concatenar dias
        df_div = sector_leader_divergence_data.copy()
        if 'date' in df_div.columns and df_div['date'].notna().any():
            latest = pd.to_datetime(df_div['date']).max()
            df_div = df_div[pd.to_datetime(df_div['date']) == latest]
        for _, row in df_div.iterrows():
            out.append(f"| {row['sector']} | {row['sector_ret_20d']:.2%} | {row['n_leaders_positive']} | {row['n_leaders_negative']} | {row['n_leaders_beating_sector']} | {row['n_leaders_valid']} | {row['classification']} |\n")
        out.append("\n")
        # F6-12 (2026-09-28): nota sobre el filtro.
        out.append("*Solo se muestran los sectores en fase ACCUMULATION o MARKUP. El resto se omite porque no genera seccion de lideres. Filtro por fase, no por cobertura.*\n\n")
        out.append("*Universo: top-5 lideres por WLS (leader_df ordenado por WLS). Distinto de 'Liderazgo relativo interno', que ordena por retorno 20d. Mismo sector, distinto subconjunto de tickers.*\n\n")
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
            out.append(f"| {row['sector']} | {_fmt_signed(row['delta_1d_ema20'], '{:+.1f}', '{:.1f}')} | {_fmt_signed(row['delta_5d_ema20'], '{:+.1f}', '{:.1f}')} | {_fmt_signed(row['delta_20d_ema20'], '{:+.1f}', '{:.1f}')} | {_fmt_signed(row['delta_5d_ema50'], '{:+.1f}', '{:.1f}')} | {_fmt_signed(row['delta_5d_ema200'], '{:+.1f}', '{:.1f}')} | {row['breadth_expansion_5d']} | {row['breadth_deterioration_5d']} |\n")
        out.append("\n")
        # H16 (2026-10-15, auditoria externa): declarar unidad y reglas.
        out.append("*H16 (2026-10-15, auditor):*\n")
        out.append("*Unidad: PUNTOS PORCENTUALES (pp). Cada delta es la variacion "
                   "en pp del % de componentes del sector por encima de su "
                   "EMA. Ej: +10.0 significa que el % de componentes sobre "
                   "EMA20 subio 10 pp (de 40% a 50%, no un +10% relativo).*\n")
        out.append("*Reglas de los flags:*\n")
        out.append("  - *Expansion = True si delta_5d_ema20 >= +5 pp Y "
                   "nivel actual pct_above_ema20 > 60%.*\n")
        out.append("  - *Deterioro = True si delta_5d_ema20 (redondeado a 1 decimal, igual que se muestra) <= -5 pp.*\n")
        out.append("  - *Ambos flags pueden ser False simultaneamente; no "
                   "pueden ser True a la vez (umbrales opuestos).*\n")
        out.append("*Fuente: indicators/sector_breadth_momentum.py.*\n\n")
    return out
