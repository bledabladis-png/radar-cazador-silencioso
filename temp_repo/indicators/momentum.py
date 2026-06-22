import pandas as pd
import numpy as np
from src.utils import robust_zscore, get_col

def compute_returns(df, tickers):
    returns = pd.DataFrame()
    for t in tickers:
        try:
            close = get_col(df, t, 'Close')
            returns[t] = close.pct_change(fill_method=None)
        except KeyError:
            pass
    return returns

def momentum_score(returns, window=63):
    ret = returns.rolling(window).mean() * window
    vol = returns.rolling(window).std()
    return ret / (vol + 1e-9)

def normalize_momentum(score_series):
    return np.tanh(robust_zscore(score_series, 60))

def compute_obv(df, ticker):
    close = get_col(df, ticker, 'Close')
    volume = get_col(df, ticker, 'Volume')
    sign = np.sign(close.diff())
    obv = (sign * volume).cumsum()
    return obv

def compute_cmf(df, ticker, window=20):
    high = get_col(df, ticker, 'High')
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    volume = get_col(df, ticker, 'Volume')
    mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
    mfv = mfm * volume
    cmf = mfv.rolling(window).sum() / volume.rolling(window).sum()
    return cmf

def compute_flow_proxy(df, ticker, window=60):
    close = get_col(df, ticker, 'Close')
    volume = get_col(df, ticker, 'Volume')
    dollar_vol = close * volume
    ret = close.pct_change(fill_method=None)
    flow = ret * dollar_vol
    flow_z = robust_zscore(flow, window=window)
    flow_smooth = flow_z.ewm(span=10, min_periods=20).mean()
    # Componentes adicionales
    obv = compute_obv(df, ticker)
    obv_z = robust_zscore(obv.pct_change(fill_method=None), window=window)
    cmf = compute_cmf(df, ticker)
    cmf_z = robust_zscore(cmf, window=window)
    # Combinación: 30% proxy, 35% OBV, 35% CMF
    combined = 0.30 * flow_smooth + 0.35 * obv_z + 0.35 * cmf_z
    return combined

def compute_price_momentum(df, ticker, window=20):
    close = get_col(df, ticker, 'Close')
    return close.pct_change(periods=window, fill_method=None)
