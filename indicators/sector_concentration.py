# -*- coding: utf-8 -*-
"""
Sector Concentration v1.2
Describe la concentración del liderazgo y el contexto sectorial.
Incluye medianas y percentiles P25/P75 de RS, momentum, flow, Wyckoff y WLS, y coberturas.
No alimenta motores, scores, pesos ni State Machine.
"""
import pandas as pd
import numpy as np
from src.utils import get_col

def _get_series(df, ticker, field):
    try:
        return get_col(df, ticker, field)
    except (KeyError, TypeError):
        if ticker in df.columns:
            s = df[ticker]
            if isinstance(s, pd.DataFrame):
                return s[field] if field in s.columns else pd.Series(dtype=float)
            else:
                return s
        return pd.Series(dtype=float)

def safe_quantile(series, q):
    if len(series.dropna()) >= 5:
        return series.quantile(q)
    return np.nan

def compute_sector_concentration(df_stocks, holdings_df, full_metrics_df):
    rows = []
    if full_metrics_df is None or full_metrics_df.empty:
        return pd.DataFrame()

    for sector_etf in holdings_df['etf'].unique():
        tickers = holdings_df[holdings_df['etf'] == sector_etf]['ticker'].tolist()
        sector_full = full_metrics_df[full_metrics_df['sector'] == sector_etf]

        # Retornos 20d para concentración positiva
        ret20_list = []
        for ticker in tickers:
            close = _get_series(df_stocks, ticker, 'Close').dropna()
            if close.empty:
                continue
            if len(close) >= 21:
                ret20_list.append((ticker, close.pct_change(20).iloc[-1]))
        ret_df = pd.DataFrame(ret20_list, columns=['ticker','ret20']).dropna()

        if ret_df.empty:
            continue

        positive = ret_df[ret_df['ret20'] > 0]
        pos_sum = positive['ret20'].sum() if len(positive) > 0 else 0

        top1 = positive.nlargest(1, 'ret20')['ret20'].sum() / pos_sum if len(positive) >= 1 and pos_sum > 0 else np.nan
        top3 = positive.nlargest(3, 'ret20')['ret20'].sum() / pos_sum if len(positive) >= 3 and pos_sum > 0 else np.nan
        top5 = positive.nlargest(5, 'ret20')['ret20'].sum() / pos_sum if len(positive) >= 5 and pos_sum > 0 else np.nan

        leader_ticker = ret_df.loc[ret_df['ret20'].idxmax(), 'ticker']
        leader_ret = ret_df.loc[ret_df['ret20'].idxmax(), 'ret20']

        # Métricas sectoriales desde full_metrics_df
        def med_col(col):
            s = pd.to_numeric(sector_full[col], errors='coerce') if col in sector_full.columns else pd.Series(dtype=float)
            return safe_quantile(s, 0.25), safe_quantile(s, 0.50), safe_quantile(s, 0.75)

        rs_p25, rs_median, rs_p75 = med_col('rs')
        momentum_p25, momentum_median, momentum_p75 = med_col('rs_mom')
        flow_p25, flow_median, flow_p75 = med_col('flow_proxy_z')
        wls_p25, wls_median, wls_p75 = med_col('wls')
        wyckoff_median = safe_quantile(pd.to_numeric(sector_full['wyckoff_score'], errors='coerce') if 'wyckoff_score' in sector_full.columns else pd.Series(dtype=float), 0.50)

        n_total = len(tickers)
        n_valid_rs = sector_full['rs'].notna().sum() if 'rs' in sector_full.columns else 0
        n_valid_momentum = sector_full['rs_mom'].notna().sum() if 'rs_mom' in sector_full.columns else 0
        n_valid_flow = sector_full['flow_proxy_z'].notna().sum() if 'flow_proxy_z' in sector_full.columns else 0
        n_valid_wyckoff = sector_full['wyckoff_score'].notna().sum() if 'wyckoff_score' in sector_full.columns else 0
        n_valid_wls = sector_full['wls'].notna().sum() if 'wls' in sector_full.columns else 0

        coverage_rs = n_valid_rs / n_total * 100 if n_total else np.nan
        coverage_momentum = n_valid_momentum / n_total * 100 if n_total else np.nan
        coverage_flow = n_valid_flow / n_total * 100 if n_total else np.nan
        coverage_wyckoff = n_valid_wyckoff / n_total * 100 if n_total else np.nan
        coverage_wls = n_valid_wls / n_total * 100 if n_total else np.nan

        # Métricas del líder
        leader_row = sector_full[sector_full['ticker'] == leader_ticker]
        if leader_row.empty:
            leader_row = ret_df[ret_df['ticker'] == leader_ticker]
        leader_rs = leader_row.iloc[0].get('rs', np.nan) if not leader_row.empty else np.nan
        leader_mom = leader_row.iloc[0].get('rs_mom', np.nan) if not leader_row.empty else np.nan
        leader_flow = leader_row.iloc[0].get('flow_proxy_z', np.nan) if not leader_row.empty else np.nan
        leader_wyckoff = leader_row.iloc[0].get('wyckoff_score', np.nan) if not leader_row.empty else np.nan
        leader_wls = leader_row.iloc[0].get('wls', np.nan) if not leader_row.empty else np.nan

        leader_vs_rs = leader_rs - rs_median if pd.notna(leader_rs) and pd.notna(rs_median) else np.nan
        leader_vs_mom = leader_mom - momentum_median if pd.notna(leader_mom) and pd.notna(momentum_median) else np.nan
        leader_vs_flow = leader_flow - flow_median if pd.notna(leader_flow) and pd.notna(flow_median) else np.nan
        leader_vs_wyckoff = leader_wyckoff - wyckoff_median if pd.notna(leader_wyckoff) and pd.notna(wyckoff_median) else np.nan
        leader_vs_wls = leader_wls - wls_median if pd.notna(leader_wls) and pd.notna(wls_median) else np.nan

        rows.append({
            'date': pd.Timestamp.now().normalize(),
            'sector': sector_etf,
            'n_total': n_total,
            'n_valid_return20': len(ret_df),
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
            'rs_p25': rs_p25,
            'rs_p75': rs_p75,
            'momentum_p25': momentum_p25,
            'momentum_p75': momentum_p75,
            'flow_p25': flow_p25,
            'flow_p75': flow_p75,
            'wls_p25': wls_p25,
            'wls_p75': wls_p75,
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