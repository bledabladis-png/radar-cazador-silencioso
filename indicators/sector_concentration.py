# -*- coding: utf-8 -*-
"""
Sector Concentration v1.0 (actualizado)
Describe la concentración del liderazgo y el contexto sectorial.
Incluye medianas de RS, momentum, flow, Wyckoff y WLS, y coberturas.
No alimenta motores, scores, pesos ni State Machine.
"""
import pandas as pd
import numpy as np
from src.utils import get_col

def compute_sector_concentration(df_stocks, holdings_df, leader_df):
    rows = []
    for sector_etf in holdings_df['etf'].unique():
        tickers = holdings_df[holdings_df['etf'] == sector_etf]['ticker'].tolist()
        sector_leaders = leader_df[leader_df['sector'] == sector_etf] if leader_df is not None else pd.DataFrame()

        valid_data = []
        for ticker in tickers:
            try:
                close = get_col(df_stocks, ticker, 'Close').dropna()
            except KeyError:
                continue
            if len(close) < 21:
                continue
            ret20 = close.pct_change(20).iloc[-1]
            if not pd.notna(ret20):
                continue

            if not sector_leaders.empty and ticker in sector_leaders['ticker'].values:
                row = sector_leaders.loc[sector_leaders['ticker'] == ticker].iloc[0]
                rs_mom = row['rs_mom']
                flow_z = row['flow_proxy_z']
                wyckoff_score_val = row['wyckoff_score']
                wls = row['wls']
            else:
                rs_mom = np.nan
                flow_z = np.nan
                wyckoff_score_val = np.nan
                wls = np.nan

            valid_data.append({
                'ticker': ticker,
                'ret20': ret20,
                'rs_mom': rs_mom,
                'momentum20': ret20,  # retorno 20d = momentum de precio
                'flow_proxy_z': flow_z,
                'wyckoff_score': wyckoff_score_val,
                'wls': wls,
            })

        if not valid_data:
            continue

        df_metrics = pd.DataFrame(valid_data)
        positive = df_metrics[df_metrics['ret20'] > 0]
        pos_sum = positive['ret20'].sum()

        top1 = positive.nlargest(1, 'ret20')['ret20'].sum() / pos_sum if len(positive) >= 1 and pos_sum > 0 else np.nan
        top3 = positive.nlargest(3, 'ret20')['ret20'].sum() / pos_sum if len(positive) >= 3 and pos_sum > 0 else np.nan
        top5 = positive.nlargest(5, 'ret20')['ret20'].sum() / pos_sum if len(positive) >= 5 and pos_sum > 0 else np.nan

        leader = df_metrics.loc[df_metrics['ret20'].idxmax()]
        leader_ticker = leader['ticker']
        leader_ret = leader['ret20']

        rs_median = df_metrics['rs_mom'].median()
        momentum_median = df_metrics['momentum20'].median()
        flow_median = df_metrics['flow_proxy_z'].median()
        wyckoff_median = df_metrics['wyckoff_score'].median()
        wls_median = df_metrics['wls'].median()

        n_total = len(tickers)
        n_valid_rs = df_metrics['rs_mom'].notna().sum()
        n_valid_momentum = df_metrics['momentum20'].notna().sum()
        n_valid_flow = df_metrics['flow_proxy_z'].notna().sum()
        n_valid_wyckoff = df_metrics['wyckoff_score'].notna().sum()
        n_valid_wls = df_metrics['wls'].notna().sum()

        coverage_rs = n_valid_rs / n_total * 100 if n_total else np.nan
        coverage_momentum = n_valid_momentum / n_total * 100 if n_total else np.nan
        coverage_flow = n_valid_flow / n_total * 100 if n_total else np.nan
        coverage_wyckoff = n_valid_wyckoff / n_total * 100 if n_total else np.nan
        coverage_wls = n_valid_wls / n_total * 100 if n_total else np.nan

        leader_vs_rs = leader['rs_mom'] - rs_median
        leader_vs_mom = leader['momentum20'] - momentum_median
        leader_vs_flow = leader['flow_proxy_z'] - flow_median
        leader_vs_wyckoff = leader['wyckoff_score'] - wyckoff_median
        leader_vs_wls = leader['wls'] - wls_median

        rows.append({
            'date': pd.Timestamp.now().normalize(),
            'sector': sector_etf,
            'n_total': n_total,
            'n_valid_return20': len(df_metrics),
            'n_positive_return20': len(positive),
            'top1_positive_return_concentration': top1,
            'top3_positive_return_concentration': top3,
            'top5_positive_return_concentration': top5,
            'rs_median': rs_median,
            'momentum_median': momentum_median,
            'flow_median': flow_median,
            'wyckoff_median': wyckoff_median,
            'wls_median': wls_median,
            'leader_ticker': leader_ticker,
            'leader_return20': leader_ret,
            'leader_vs_median_rs': leader_vs_rs,
            'leader_vs_median_mom': leader_vs_mom,
            'leader_vs_median_flow': leader_vs_flow,
            'leader_vs_median_wyckoff': leader_vs_wyckoff,
            'leader_vs_median_wls': leader_vs_wls,
            'n_valid_rs': n_valid_rs,
            'n_valid_momentum': n_valid_momentum,
            'n_valid_flow': n_valid_flow,
            'n_valid_wyckoff': n_valid_wyckoff,
            'n_valid_wls': n_valid_wls,
            'coverage_rs': coverage_rs,
            'coverage_momentum': coverage_momentum,
            'coverage_flow': coverage_flow,
            'coverage_wyckoff': coverage_wyckoff,
            'coverage_wls': coverage_wls,
        })

    return pd.DataFrame(rows)
