# -*- coding: utf-8 -*-
"""
Sector Flow Characteristics v1.0
Describe magnitud, direccion y persistencia del flujo primario de ETF sectoriales SSGA.
No alimenta motores, scores, pesos ni State Machine.
No mezcla con Flow Proxy ni otras capas de flujo.
"""
import pandas as pd
import numpy as np
from src.utils import get_col

SECTORS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']

def _persistence(pos, neg):
    denom = pos + neg
    return pos / denom if denom > 0 else np.nan

def compute_sector_flow_characteristics(flow_csv_path, price_df):
    """
    flow_csv_path: ruta al CSV historico de etf_primary_flow.csv
    price_df: DataFrame de mercado con MultiIndex (Close, ticker)
    """
    flow_df = pd.read_csv(flow_csv_path, parse_dates=['Date'])
    flow_df = flow_df.sort_values('Date')
    rows = []

    for sector in SECTORS:
        sec = flow_df[flow_df['ticker'] == sector].copy()
        valid = sec[sec['primary_flow_usd'].notna()]
        if valid.empty:
            continue

        # Ultimas 5 y 20 observaciones reales de flujo (sin imputar)
        last5 = valid['primary_flow_usd'].tail(5)
        last20 = valid['primary_flow_usd'].tail(20)

        # Valores actuales (ultimo observado)
        flow_dollar = valid['primary_flow_usd'].iloc[-1]
        flow_pct_aum = valid['primary_flow_pct'].iloc[-1] if 'primary_flow_pct' in valid.columns else np.nan
        flow_zscore = valid['primary_flow_z'].iloc[-1] if 'primary_flow_z' in valid.columns else np.nan

        # Acumulados solo si hay al menos 3 observaciones
        flow_5d_sum = last5.sum() if len(last5) >= 3 else np.nan
        flow_20d_sum = last20.sum() if len(last20) >= 3 else np.nan

        # Conteos de dirección y persistencia
        pos5 = (last5 > 0).sum()
        neg5 = (last5 < 0).sum()
        pos20 = (last20 > 0).sum()
        neg20 = (last20 < 0).sum()
        persistence_5d = _persistence(pos5, neg5) if len(last5) >= 3 else np.nan
        persistence_20d = _persistence(pos20, neg20) if len(last20) >= 3 else np.nan

        # Precio sectorial
        try:
            close = get_col(price_df, sector, 'Close').dropna()
        except KeyError:
            continue
        if len(close) >= 21:
            price_ret_5d = close.iloc[-1] / close.iloc[-6] - 1
            price_ret_20d = close.iloc[-1] / close.iloc[-21] - 1
        else:
            price_ret_5d = np.nan
            price_ret_20d = np.nan

        # Matriz Precio vs Flujo (dirección por flujo_dollar)
        if flow_dollar > 0 and price_ret_20d > 0:
            regime = 'Confirmación'
        elif flow_dollar > 0 and price_ret_20d < 0:
            regime = 'Absorción potencial'
        elif flow_dollar < 0 and price_ret_20d > 0:
            regime = 'Divergencia bajista'
        elif flow_dollar < 0 and price_ret_20d < 0:
            regime = 'Confirmación de debilidad'
        elif flow_dollar == 0:
            regime = 'Neutral / sin confirmación'
        else:
            regime = 'Indeterminado'

        rows.append({
            'date': valid['Date'].iloc[-1],
            'sector': sector,
            'flow_dollar': flow_dollar,
            'flow_pct_aum': flow_pct_aum,
            'flow_zscore': flow_zscore,
            'flow_5d_sum': flow_5d_sum,
            'flow_20d_sum': flow_20d_sum,
            'positive_days_5d': pos5,
            'negative_days_5d': neg5,
            'positive_days_20d': pos20,
            'negative_days_20d': neg20,
            'persistence_5d': persistence_5d,
            'persistence_20d': persistence_20d,
            'n_obs_5d': len(last5),
            'n_obs_20d': len(last20),
            'price_ret_5d': price_ret_5d,
            'price_ret_20d': price_ret_20d,
            'price_flow_regime': regime,
        })

    return pd.DataFrame(rows)
