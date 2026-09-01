# -*- coding: utf-8 -*-
"""
Sensibilidad de Coverage en SLPM v1.2 usando datos reales de lideres.

Simula niveles de cobertura (0.25, 0.50, 0.75, 1.00) manteniendo
los valores reales de breadth, structural y persistence.

Salida: outputs/audit/sensitivity_coverage.csv
"""

from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.insert(0, os.path.abspath("."))

import pandas as pd

from indicators.slpm_v12 import compute_leader_breadth_v2
from indicators.state_machine import classify_leadership_state
from config.settings import SLPM_EXPECTED_LEADERS

LEADERS_CSV = Path('outputs/report/analisis_lideres.csv')
RANKINGS_CSV = Path('outputs/report/sector_rankings.csv')
OUTPUT_PATH = Path('outputs/audit/sensitivity_coverage.csv')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

COVERAGE_LEVELS = [0.25, 0.50, 0.75, 1.00]

def main():
    if not LEADERS_CSV.exists() or not RANKINGS_CSV.exists():
        raise SystemExit("No se encontraron outputs/report/analisis_lideres.csv o sector_rankings.csv")

    leaders = pd.read_csv(LEADERS_CSV)
    rankings = pd.read_csv(RANKINGS_CSV)

    top_etf = rankings.iloc[0]['ticker']  # columna ticker contiene el sector ETF
    top_score = rankings.iloc[0]['score']

    # Obtener métricas de los 5 líderes del sector top
    top_leaders = leaders[leaders['sector'] == top_etf].head(5)
    if top_leaders.empty:
        raise SystemExit(f"No se encontraron líderes para {top_etf}")

    leader_metrics = []
    for _, row in top_leaders.iterrows():
        leader_metrics.append({
            'ticker': row['ticker'],
            'rs': row['rs'] if pd.notna(row.get('rs')) else None,
            'rs_momentum': row['rs_mom'] if pd.notna(row.get('rs_mom')) else None,
            'flow_proxy_z': row['flow_proxy_z'] if pd.notna(row.get('flow_proxy_z')) else None,
            'wyckoff_phase': row['wyckoff_phase'] if pd.notna(row.get('wyckoff_phase')) else ''
        })

    # Calcular breadth real
    breadth_v2 = compute_leader_breadth_v2(leader_metrics, expected_leaders=SLPM_EXPECTED_LEADERS)
    real_coverage = breadth_v2['coverage']
    structural = top_score  # usar score combinado como proxy, documentado
    persistence = top_leaders['persistence_10d'].mean() if 'persistence_10d' in top_leaders.columns else 0.5

    print(f"Sector líder: {top_etf}, coverage real: {real_coverage:.2f}, structural proxy: {structural:.3f}, persistence proxy: {persistence:.3f}")

    rows = []
    for cov in COVERAGE_LEVELS:
        # Simular effective_breadth con cobertura forzada
        composite = breadth_v2['composite']
        effective_breadth = composite * cov if cov < 0.5 else composite
        result = classify_leadership_state(structural, effective_breadth, persistence, coverage=cov)
        rows.append({
            'coverage': cov,
            'composite_breadth': composite,
            'effective_breadth': effective_breadth,
            'state': result['state'],
            'reason': result['reason'],
            'data_quality': result.get('data_quality', 'UNKNOWN')
        })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"\nSensibilidad de Coverage guardada en {OUTPUT_PATH}")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
