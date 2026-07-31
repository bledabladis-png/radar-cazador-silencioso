# -*- coding: utf-8 -*-
"""
indicators/breadth.py -- Amplitud de Mercado Sectorial (v3.15 corregido)
Calcula el %% de sectores sobre EMAs y nuevos maximos/minimos de 52 semanas.
Usa breadth_core.py para NH/NL con shift(1).
"""
import numpy as np
from config.settings import BREADTH_EMA_FAST, BREADTH_EMA_MEDIUM, BREADTH_EMA_SLOW
import pandas as pd
from src.utils import get_col
from config.tickers import MARKET_TICKERS
from indicators.breadth_core import compute_new_highs_lows, validate_coverage

def compute_breadth(df):
    sectors = MARKET_TICKERS['sectors']
    expected = len(sectors)
    
    closes = pd.DataFrame(index=df.index)
    highs = pd.DataFrame(index=df.index)
    lows = pd.DataFrame(index=df.index)

    for s in sectors:
        try:
            closes[s] = get_col(df, s, 'Close')
            highs[s] = get_col(df, s, 'High')
            lows[s] = get_col(df, s, 'Low')
        except KeyError:
            closes[s] = np.nan
            highs[s] = np.nan
            lows[s] = np.nan

    # Validar cobertura
    validate_coverage(closes, expected, "Breadth Sectorial")

    # EMAs con adjust=False (estandar de analisis tecnico)
    ema20 = closes.ewm(span=BREADTH_EMA_FAST, adjust=False, min_periods=20).mean()
    ema50 = closes.ewm(span=BREADTH_EMA_MEDIUM, adjust=False, min_periods=50).mean()
    ema200 = closes.ewm(span=BREADTH_EMA_SLOW, adjust=False, min_periods=200).mean()

    b20 = (closes > ema20).mean(axis=1)
    b50 = (closes > ema50).mean(axis=1)
    b200 = (closes > ema200).mean(axis=1)

    # NH/NL usando la funcion comun (shift(1) + High/Low)
    nh, nl = compute_new_highs_lows(highs)
    nh_pct = nh / expected
    nl_pct = nl / expected

    return b20, b50, b200, nh_pct, nl_pct

