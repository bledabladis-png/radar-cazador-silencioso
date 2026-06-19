import pandas as pd
import numpy as np
from src.utils import get_col

def compute_breadth(df):
    sectors = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
    ema20_positions = []
    ema50_positions = []
    ema200_positions = []
    new_highs = []
    new_lows = []

    for ticker in sectors:
        try:
            close = get_col(df, ticker, 'Close')
        except KeyError:
            continue

        ema20 = close.ewm(span=20, min_periods=20).mean()
        ema50 = close.ewm(span=50, min_periods=50).mean()
        ema200 = close.ewm(span=200, min_periods=200).mean()
        rolling_high = close.rolling(252, min_periods=252).max()
        rolling_low = close.rolling(252, min_periods=252).min()

        ema20_positions.append(close > ema20)
        ema50_positions.append(close > ema50)
        ema200_positions.append(close > ema200)
        new_highs.append(close == rolling_high)
        new_lows.append(close == rolling_low)

    if not ema50_positions:
        empty = pd.Series(dtype=float)
        return empty, empty, empty, empty, empty

    breadth_20 = pd.concat(ema20_positions, axis=1).mean(axis=1)
    breadth_50 = pd.concat(ema50_positions, axis=1).mean(axis=1)
    breadth_200 = pd.concat(ema200_positions, axis=1).mean(axis=1)
    new_highs_pct = pd.concat(new_highs, axis=1).mean(axis=1)
    new_lows_pct = pd.concat(new_lows, axis=1).mean(axis=1)

    return breadth_20, breadth_50, breadth_200, new_highs_pct, new_lows_pct
