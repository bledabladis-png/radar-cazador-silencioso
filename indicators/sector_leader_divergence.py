# -*- coding: utf-8 -*-
"""
Divergencia sector-líderes v1.0
Detecta alineación/divergencia entre el retorno 20d del ETF sectorial y sus Top 5 líderes.
No alimenta motores, scores, pesos ni State Machine.
"""
import pandas as pd
import numpy as np
from src.utils import get_col

SECTORS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']

def _ret_20d(close):
    if len(close) < 21:
        return np.nan
    return close.iloc[-1] / close.iloc[-21] - 1

def compute_sector_leader_divergence(df_stocks, holdings_df, leader_df, df_market):
    if leader_df is None or leader_df.empty:
        return pd.DataFrame()

    rows = []
    for sector_etf, group in holdings_df.groupby('etf'):
        if sector_etf not in SECTORS:
            continue

        # Retorno del sector
        try:
            sector_close = get_col(df_market, sector_etf, 'Close').dropna()
        except KeyError:
            continue
        sector_ret = _ret_20d(sector_close)
        if pd.isna(sector_ret):
            continue

        # Top 5 líderes del sector desde leader_df
        sector_leaders = leader_df[leader_df['sector'] == sector_etf].head(5)
        if sector_leaders.empty:
            continue

        n_total = len(sector_leaders)
        n_valid = 0
        n_positive = 0
        n_negative = 0
        n_beating = 0

        for _, leader in sector_leaders.iterrows():
            ticker = leader['ticker']
            try:
                close = get_col(df_stocks, ticker, 'Close').dropna()
            except KeyError:
                continue
            ret = _ret_20d(close)
            if pd.isna(ret):
                continue
            n_valid += 1
            if ret > 0:
                n_positive += 1
            elif ret < 0:
                n_negative += 1
            if ret > sector_ret:
                n_beating += 1

        if n_valid < 3:
            classification = 'N/D'
        else:
            if n_positive >= 3 and sector_ret < 0:
                classification = 'Liderazgo relativo de líderes'
            elif n_positive >= 3 and sector_ret > 0:
                classification = 'Alineación positiva'
            elif n_negative >= 3 and sector_ret < 0:
                classification = 'Alineación negativa'
            elif n_negative >= 3 and sector_ret > 0:
                classification = 'Divergencia negativa'
            else:
                classification = 'Mixto'

        rows.append({
            'date': pd.Timestamp.now().normalize(),
            'sector': sector_etf,
            'sector_ret_20d': sector_ret,
            'n_leaders_total': n_total,
            'n_leaders_valid': n_valid,
            'n_leaders_positive': n_positive,
            'n_leaders_negative': n_negative,
            'n_leaders_beating_sector': n_beating,
            'classification': classification,
        })

    return pd.DataFrame(rows)
