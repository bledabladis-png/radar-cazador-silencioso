# -*- coding: utf-8 -*-
"""
Regresión BASE vs BASE+LIS usando datos reales transversales de sectores.

Objetivo: evaluar si añadir LIS (métrica diagnóstica) mejora la
clasificación de estados CONFIRMED/EMERGING frente al resto,
en comparación con BASE (structural, breadth, persistence).

Salida: outputs/audit/regresion_base_vs_lis.csv
"""

from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from indicators.slpm_v12 import compute_leader_breadth_v2, compute_leader_integrity
from indicators.state_machine import classify_leadership_state
from config.settings import SLPM_EXPECTED_LEADERS

LEADERS_CSV = Path('outputs/report/analisis_lideres.csv')
RANKINGS_CSV = Path('outputs/report/sector_rankings.csv')
OUTPUT_PATH = Path('outputs/audit/regresion_base_vs_lis.csv')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    leaders = pd.read_csv(LEADERS_CSV)
    rankings = pd.read_csv(RANKINGS_CSV)

    # Mapear sector -> score
    score_map = dict(zip(rankings['ticker'], rankings['score']))

    rows = []
    for sector, group in leaders.groupby('sector'):
        # solo sectores con al menos 5 líderes
        top = group.head(5)
        if len(top) < 3:
            continue
        leader_metrics = []
        for _, row in top.iterrows():
            leader_metrics.append({
                'ticker': row['ticker'],
                'rs': row['rs'] if pd.notna(row.get('rs')) else None,
                'rs_momentum': row['rs_mom'] if pd.notna(row.get('rs_mom')) else None,
                'flow_proxy_z': row['flow_proxy_z'] if pd.notna(row.get('flow_proxy_z')) else None,
                'wyckoff_phase': row['wyckoff_phase'] if pd.notna(row.get('wyckoff_phase')) else ''
            })
        breadth = compute_leader_breadth_v2(leader_metrics, SLPM_EXPECTED_LEADERS)
        integrity = compute_leader_integrity(leader_metrics)
        structural = score_map.get(sector, 0.0)
        persistence = top['persistence_10d'].mean() if 'persistence_10d' in top else 0.5
        state_result = classify_leadership_state(structural, breadth['effective_composite'], persistence, coverage=breadth['coverage'])
        rows.append({
            'sector': sector,
            'structural': structural,
            'breadth': breadth['effective_composite'],
            'persistence': persistence,
            'lis': integrity['lis'],
            'state': state_result['state'],
            'target': 1 if state_result['state'] in ('CONFIRMED', 'EMERGING') else 0
        })

    df = pd.DataFrame(rows)
    print(df[['sector','structural','breadth','persistence','lis','state']].to_string(index=False))

    if df['target'].nunique() < 2:
        print("No hay suficiente variabilidad en target para regresión.")
        df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
        return

    # BASE: structural, breadth, persistence
    X_base = df[['structural','breadth','persistence']].values
    # BASE+LIS: añade lis
    X_full = df[['structural','breadth','persistence','lis']].values
    y = df['target'].values

    # Validación cruzada simple para AUC
    clf_base = LogisticRegression(max_iter=1000)
    clf_full = LogisticRegression(max_iter=1000)

    auc_base = cross_val_score(clf_base, X_base, y, cv=3, scoring='roc_auc').mean()
    auc_full = cross_val_score(clf_full, X_full, y, cv=3, scoring='roc_auc').mean()

    summary = pd.DataFrame([{
        'n_sectores': len(df),
        'auc_base': auc_base,
        'auc_base_lis': auc_full,
        'diferencia': auc_full - auc_base
    }])
    print("\nResumen de regresión:")
    print(summary.to_string(index=False))

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    summary.to_csv(OUTPUT_PATH.with_name('regresion_base_vs_lis_summary.csv'), index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    main()
