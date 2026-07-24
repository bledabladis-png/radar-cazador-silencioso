# -*- coding: utf-8 -*-
"""
shock_sensitivity.py -- Shock Sensitivity v1.1
"""
import pandas as pd
import numpy as np
from src.utils import get_col

def compute_commodity_market_correlation(df_market, sector_etf, benchmark='^GSPC', commodity='^SPGSCI', window=126):
    try:
        close_sector = get_col(df_market, sector_etf, 'Close')
        close_bench = get_col(df_market, benchmark, 'Close')
        close_comm = get_col(df_market, commodity, 'Close')
    except KeyError:
        return {'commodity_corr': None, 'market_corr': None, 'commodity_level': 'N/A', 'market_level': 'N/A', 'commodity_corr_value': None, 'market_corr_value': None}
    
    ret_sector = close_sector.pct_change().dropna()
    ret_bench = close_bench.pct_change().dropna()
    ret_comm = close_comm.pct_change().dropna()
    
    common = ret_sector.index.intersection(ret_bench.index).intersection(ret_comm.index)
    ret_sector = ret_sector[common]
    ret_bench = ret_bench[common]
    ret_comm = ret_comm[common]
    
    if len(ret_sector) < window:
        return {'commodity_corr': None, 'market_corr': None, 'commodity_level': 'N/A', 'market_level': 'N/A', 'commodity_corr_value': None, 'market_corr_value': None}
    
    comm_corr = ret_sector.rolling(window).corr(ret_comm).iloc[-1]
    market_corr = ret_sector.rolling(window).corr(ret_bench).iloc[-1]
    
    def classify_corr(corr):
        if pd.isna(corr):
            return 'N/A'
        if abs(corr) > 0.6:
            return 'HIGH'
        if abs(corr) > 0.3:
            return 'MODERATE'
        return 'LOW'
    
    return {
        'commodity_corr': float(comm_corr) if pd.notna(comm_corr) else None,
        'market_corr': float(market_corr) if pd.notna(market_corr) else None,
        'commodity_level': classify_corr(comm_corr),
        'market_level': classify_corr(market_corr),
        'commodity_corr_value': float(comm_corr) if pd.notna(comm_corr) else None,
        'market_corr_value': float(market_corr) if pd.notna(market_corr) else None
    }

