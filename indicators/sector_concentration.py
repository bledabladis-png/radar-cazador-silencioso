# -*- coding: utf-8 -*-
"""
Sector Concentration v1.0
Describe la concentración del liderazgo por sector.
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
                rs_mom = sector_leaders.loc[sector_leaders['ticker'] == ticker, 'rs_mom'].iloc[0]
                wls = sector_leaders.loc[sector_leaders['ticker'] == ticker, 'wls'].iloc[0]
            else:
                rs_mom = np.nan
                wls = np.nan

            valid_data.append({
                'ticker': ticker,
                'ret20': ret20,
                'rs_mom': rs_mom,
                'momentum20': ret20,  # momentum 20d = retorno 20d
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
        wls_median = df_metrics['wls'].median()

        leader_vs_rs = leader['rs_mom'] - rs_median
        leader_vs_mom = leader['momentum20'] - momentum_median
        leader_vs_wls = leader['wls'] - wls_median

        rows.append({
            'date': pd.Timestamp.now().normalize(),
            'sector': sector_etf,
            'n_total': len(tickers),
            'n_valid_return20': len(df_metrics),
            'n_positive_return20': len(positive),
            'top1_positive_return_concentration': top1,
            'top3_positive_return_concentration': top3,
            'top5_positive_return_concentration': top5,
            'rs_median': rs_median,
            'momentum_median': momentum_median,
            'wls_median': wls_median,
            'leader_ticker': leader_ticker,
            'leader_return20': leader_ret,
            'leader_vs_median_rs': leader_vs_rs,
            'leader_vs_median_mom': leader_vs_mom,
            'leader_vs_median_wls': leader_vs_wls,
        })

    return pd.DataFrame(rows)
