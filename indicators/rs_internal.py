# -*- coding: utf-8 -*-
"""
RS Interno y Absoluto v1.0
Descompone la fortaleza relativa de cada accion frente al mercado y frente a su sector.
No alimenta motores, scores, pesos ni State Machine.
No recalcula el RS oficial de stock_leader.py.
"""
import pandas as pd
import numpy as np
from src.utils import get_col

def _ret_20d(close):
    if len(close) < 21:
        return np.nan
    return close.iloc[-1] / close.iloc[-21] - 1

def classify_rs(abs_val, int_val):
    if pd.isna(abs_val) or pd.isna(int_val):
        return 'N/D'
    if abs_val > 0 and int_val > 0:
        return 'Liderazgo relativo doble'
    elif abs_val > 0 and int_val <= 0:
        return 'Fortaleza sectorial'
    elif abs_val <= 0 and int_val > 0:
        return 'Liderazgo interno en sector débil'
    else:
        return 'Debilidad relativa doble'

def compute_rs_internal(df_stocks, holdings_df, df_market, benchmark='SPY'):
    rows = []
    bench_ticker = benchmark
    try:
        get_col(df_market, bench_ticker, 'Close')
    except KeyError:
        bench_ticker = '^GSPC'

    for sector_etf, group in holdings_df.groupby('etf'):
        tickers = group['ticker'].tolist()
        try:
            sector_close = get_col(df_market, sector_etf, 'Close').dropna()
            bench_close = get_col(df_market, bench_ticker, 'Close').dropna()
        except KeyError:
            continue

        for ticker in tickers:
            try:
                close = get_col(df_stocks, ticker, 'Close').dropna()
            except KeyError:
                continue
            if len(close) < 21:
                continue

            common = close.index.intersection(sector_close.index).intersection(bench_close.index)
            if len(common) < 21:
                continue

            close = close.loc[common]
            sector = sector_close.loc[common]
            bench = bench_close.loc[common]

            price_ret = _ret_20d(close)
            sector_ret = _ret_20d(sector)
            bench_ret = _ret_20d(bench)

            rs_abs = (1 + price_ret) / (1 + bench_ret) - 1 if pd.notna(price_ret) and pd.notna(bench_ret) else np.nan
            rs_int = (1 + price_ret) / (1 + sector_ret) - 1 if pd.notna(price_ret) and pd.notna(sector_ret) else np.nan
            classification = classify_rs(rs_abs, rs_int)

            rows.append({
                'date': pd.Timestamp.now().normalize(),
                'sector': sector_etf,
                'ticker': ticker,
                'price_ret_20d': price_ret,
                'sector_ret_20d': sector_ret,
                'benchmark_ret_20d': bench_ret,
                'rs_abs_20d': rs_abs,
                'rs_internal_20d': rs_int,
                'classification': classification,
            })

    return pd.DataFrame(rows)
