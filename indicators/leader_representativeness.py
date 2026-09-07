# -*- coding: utf-8 -*-
"""
Representatividad del líder v1.0
Calcula la distancia del líder a la mediana sectorial.
No alimenta motores, scores, pesos ni State Machine.
No recalcula medianas; consume sector_concentration.csv.
"""
import pandas as pd
import numpy as np

def _latest_medians(conc_path):
    try:
        df = pd.read_csv(conc_path, parse_dates=['date'])
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values('date')
    latest = df.drop_duplicates(subset='sector', keep='last').set_index('sector')
    return latest

def compute_leader_representativeness(leader_df, sector_concentration_path):
    medians = _latest_medians(sector_concentration_path)
    if medians.empty or leader_df is None or leader_df.empty:
        return pd.DataFrame()

    rows = []
    for _, leader in leader_df.iterrows():
        sector = leader['sector']
        if sector not in medians.index:
            continue
        med = medians.loc[sector]

        def distance(leader_val, median_val):
            if pd.isna(leader_val) or pd.isna(median_val):
                return np.nan
            return leader_val - median_val

        # Momentum: usar rs_mom como momentum relativo oficial
        rs_dist = distance(leader.get('rs_mom', np.nan), med.get('rs_median', np.nan))
        mom_dist = distance(leader.get('rs_mom', np.nan), med.get('momentum_median', np.nan))
        flow_dist = distance(leader.get('flow_proxy_z', np.nan), med.get('flow_median', np.nan))
        wls_dist = distance(leader.get('wls', np.nan), med.get('wls_median', np.nan))

        rows.append({
            'date': pd.Timestamp.now().normalize(),
            'sector': sector,
            'ticker': leader['ticker'],
            'rs_distance_to_median': rs_dist,
            'mom_distance_to_median': mom_dist,
            'flow_distance_to_median': flow_dist,
            'wls_distance_to_median': wls_dist,
            'sector_rank_pct': leader.get('sector_rank_pct', np.nan),
            'n_valid_rs': med.get('n_valid_rs', np.nan),
            'n_valid_momentum': med.get('n_valid_momentum', np.nan),
            'n_valid_flow': med.get('n_valid_flow', np.nan),
            'n_valid_wls': med.get('n_valid_wls', np.nan),
        })

    return pd.DataFrame(rows)
