# -*- coding: utf-8 -*-
"""Fase 8a del pipeline: forzar lideres SLPM + tactical/structural engines
+ persistence sectorial.

Extraido de run.py (refactor C2, fase C2-8a).
"""

from pathlib import Path

import pandas as pd

from src.utils import append_dedup, get_col, _observation_date_from_df
from indicators.persistence import compute_persistence


SECTOR_ETFS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']


def _forzar_lideres_slpm(sector_results, sector_flow_rank, otros_flow_rank, leader_df):
    """Construye leader_metrics_for_slpm forzando el top sector."""
    leader_metrics_for_slpm = []
    top_sector_ticker = sector_results['ranking'][0][0]
    top_sector_flow = 0.0
    for t, f in sector_flow_rank + otros_flow_rank:
        if t == top_sector_ticker:
            top_sector_flow = f
            break

    try:
        if leader_df is not None and not leader_df.empty:
            top_etf = top_sector_ticker
            sector1_df = leader_df[leader_df['sector'] == top_etf]
            for _, row in sector1_df.head(5).iterrows():
                leader_metrics_for_slpm.append({
                    'ticker': row['ticker'],
                    'rs': row['rs'] if pd.notna(row.get('rs')) else None,
                    'rs_momentum': row['rs_mom'] if pd.notna(row.get('rs_mom')) else None,
                    'flow_proxy_z': row['flow_proxy_z'] if pd.notna(row.get('flow_proxy_z')) else None,
                    'wyckoff_phase': row['wyckoff_phase'] if pd.notna(row.get('wyckoff_phase')) else ''
                })
            print(f"    Lideres forzados para SLPM ({top_etf}): {len(leader_metrics_for_slpm)} tickers")
    except Exception as e:
        print(f"    No se pudieron forzar lideres para SLPM: {e}")

    return leader_metrics_for_slpm, top_sector_flow


def _compute_tactical_structural(df_market):
    tactical_scores = {}
    structural_scores = {}
    try:
        from regimes.tactical_engine import compute_tactical_score
        from regimes.structural_engine import compute_structural_score
        for sector_etf in SECTOR_ETFS:
            try:
                tactical_scores[sector_etf] = compute_tactical_score(df_market, sector_etf)
                structural_scores[sector_etf] = compute_structural_score(df_market, sector_etf)
            except Exception as e:
                print(f"  [WARN] tactical/structural engine: {e}")
                tactical_scores[sector_etf] = 0.0
                structural_scores[sector_etf] = 0.0
        print(f"    Tactical/Structural engines calculados para {len(tactical_scores)} sectores.")
    except Exception as e:
        print(f"    Tactical/Structural engines omitidos: {e}")
    return tactical_scores, structural_scores


def _compute_persistence_and_save(df_market):
    sector_persistence = {}
    try:
        for sector_etf in SECTOR_ETFS:
            try:
                close_sector = get_col(df_market, sector_etf, 'Close')
                close_spy = get_col(df_market, '^GSPC', 'Close')
                rs = close_sector / close_spy
                rs20 = rs.pct_change(20, fill_method=None)
                pers = compute_persistence(rs20, threshold=0.0, lookback=12)
                sector_persistence[sector_etf] = pers
            except Exception as e:
                print(f"  [WARN] persistence: {e}")
                sector_persistence[sector_etf] = None
        print(f"    Persistence calculada para {len(sector_persistence)} sectores.")
    except Exception as e:
        print(f"    Persistence omitida: {e}")
        sector_persistence = {s: None for s in SECTOR_ETFS}

    # Guardar CSV historico de persistencia sectorial
    try:
        persist_rows = []
        date_val = _observation_date_from_df(df_market)
        for sec, val in sector_persistence.items():
            persist_rows.append({'date': date_val, 'sector': sec, 'persistence': val})
        persist_df = pd.DataFrame(persist_rows)
        p_path = Path('outputs/history/sector_persistence.csv')
        p_path.parent.mkdir(parents=True, exist_ok=True)
        if p_path.exists():
            hist_p = pd.read_csv(p_path, encoding='utf-8')
            persist_df = append_dedup(hist_p, persist_df, ['date','sector'])
        persist_df.to_csv(p_path, index=False, encoding='utf-8')
        print("  Sector Persistence CSV guardado.")
    except Exception as e:
        print(f"  Sector Persistence CSV omitido: {e}")

    return sector_persistence


def compute_engines(df_market, sector_results, sector_flow_rank, otros_flow_rank, leader_df):
    """Ejecuta forzar-lideres-SLPM + tactical/structural + persistence.

    Returns:
        dict con keys:
            leader_metrics_for_slpm, top_sector_flow,
            tactical_scores, structural_scores, sector_persistence
    """
    leader_metrics_for_slpm, top_sector_flow = _forzar_lideres_slpm(
        sector_results, sector_flow_rank, otros_flow_rank, leader_df
    )
    tactical_scores, structural_scores = _compute_tactical_structural(df_market)
    sector_persistence = _compute_persistence_and_save(df_market)
    return {
        'leader_metrics_for_slpm': leader_metrics_for_slpm,
        'top_sector_flow': top_sector_flow,
        'tactical_scores': tactical_scores,
        'structural_scores': structural_scores,
        'sector_persistence': sector_persistence,
    }
