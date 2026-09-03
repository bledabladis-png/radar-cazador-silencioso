# -*- coding: utf-8 -*-
"""
Auditoría de canales RS/Flow → Tactical/Structural/Breadth.

Evalúa cómo las señales de Relative Strength (RS) y Flow Proxy
se relacionan con los canales táctico, estructural y de breadth.

No optimiza pesos ni construye superindicadores.
Salida: outputs/audit/audit_rs_flow_channels.csv
"""

from __future__ import annotations

import sys, os
from pathlib import Path
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from scipy.stats import spearmanr

from data.providers.router import DataRouter
from src.utils import get_col
from indicators.momentum import compute_flow_proxy, compute_price_momentum
from regimes.tactical_engine import compute_tactical_score
from regimes.structural_engine import compute_structural_score
from indicators.slpm_v12 import compute_leader_breadth_v2
from config.settings import SLPM_EXPECTED_LEADERS

SECTORS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
OUTPUT_PATH = Path('outputs/audit/audit_rs_flow_channels.csv')
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def load_market_data():
    cache = Path('data/market_data.csv')
    if cache.exists():
        df = pd.read_csv(cache, header=[0,1], index_col=0, parse_dates=True)
        # Verificar que exista columna Close para algún sector
        if any(('Close', s) in df.columns for s in SECTORS):
            return df
    print("Descargando datos de mercado...")
    router = DataRouter()
    tickers = SECTORS + ['^GSPC']
    data = router.get_market_data(tickers, period="5y")
    if data is None or data.empty:
        raise RuntimeError("No se pudieron obtener datos de mercado")
    data.to_csv(cache)
    return data

def main():
    data = load_market_data()
    rows = []
    for sector in SECTORS:
        try:
            close = get_col(data, sector, 'Close')
            if len(close.dropna()) < 60:
                continue
            rs20 = compute_price_momentum(data, sector, window=20).iloc[-1]
            flow = compute_flow_proxy(data, sector).iloc[-1]
            tactical = compute_tactical_score(data, sector)
            persistence = 0.5  # valor neutro documentado
            structural = compute_structural_score(data, sector, flow_structure=flow, persistence=persistence)
            rows.append({
                'sector': sector,
                'rs20': rs20,
                'flow_proxy': flow,
                'tactical': tactical,
                'structural': structural
            })
        except Exception as e:
            print(f"  {sector}: {e}")

    df = pd.DataFrame(rows).dropna()
    if df.empty:
        raise SystemExit("No se pudieron calcular métricas para la auditoría")

    # Añadir breadth de líderes desde CSV si existe
    leaders_csv = Path('outputs/report/analisis_lideres.csv')
    if leaders_csv.exists():
        leaders = pd.read_csv(leaders_csv)
        breadth_by_sector = {}
        for sector, group in leaders.groupby('sector'):
            top = group.head(5)
            leader_metrics = []
            for _, row in top.iterrows():
                leader_metrics.append({
                    'ticker': row['ticker'],
                    'rs': row['rs'] if pd.notna(row.get('rs')) else None,
                    'rs_momentum': row['rs_mom'] if pd.notna(row.get('rs_mom')) else None,
                    'flow_proxy_z': row['flow_proxy_z'] if pd.notna(row.get('flow_proxy_z')) else None,
                    'wyckoff_phase': row['wyckoff_phase'] if pd.notna(row.get('wyckoff_phase')) else ''
                })
            b = compute_leader_breadth_v2(leader_metrics, SLPM_EXPECTED_LEADERS)
            breadth_by_sector[sector] = b['composite']
        df['breadth'] = df['sector'].map(breadth_by_sector)

    # Calcular correlaciones de Spearman
    corr_rows = []
    pairs = [
        ('rs20', 'tactical'),
        ('flow_proxy', 'structural'),
        ('rs20', 'structural'),
        ('flow_proxy', 'tactical'),
    ]
    if 'breadth' in df.columns and df['breadth'].notna().sum() >= 5:
        pairs += [
            ('rs20', 'breadth'),
            ('flow_proxy', 'breadth'),
            ('tactical', 'breadth'),
            ('structural', 'breadth'),
        ]
    for a, b in pairs:
        rho, pval = spearmanr(df[a], df[b])
        corr_rows.append({'par': f'{a} vs {b}', 'spearman_rho': rho, 'p_value': pval})

    df_out = pd.DataFrame(corr_rows)
    print("\nMatriz de correlaciones:")
    print(df_out.to_string(index=False))

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    df_out.to_csv(OUTPUT_PATH.with_name('audit_rs_flow_channels_corr.csv'), index=False, encoding='utf-8-sig')
    print(f"\nDatos guardados en {OUTPUT_PATH} y audit_rs_flow_channels_corr.csv")

if __name__ == "__main__":
    main()
