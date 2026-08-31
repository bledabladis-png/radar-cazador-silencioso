import pandas as pd
import numpy as np
from config.settings import FLOW_ZSCORE_WINDOW, FLOW_EWM_SPAN, FLOW_CMF_WINDOW, MOMENTUM_SHARPE_WINDOW, MOMENTUM_PRICE_WINDOW
from src.utils import robust_zscore, get_col
from config.weights import FLOW_PROXY_WEIGHTS

def compute_returns(df, tickers):
    returns = pd.DataFrame()
    for t in tickers:
        try:
            close = get_col(df, t, 'Close')
            returns[t] = close.pct_change(fill_method=None)
        except KeyError:
            pass
    return returns

def momentum_score(returns, window=MOMENTUM_SHARPE_WINDOW):
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

def compute_cmf(df, ticker, window=FLOW_CMF_WINDOW):
    high = get_col(df, ticker, 'High')
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    volume = get_col(df, ticker, 'Volume')
    mfm = ((close - low) - (high - close)) / (high - low + 1e-9)
    mfv = mfm * volume
    cmf = mfv.rolling(window).sum() / volume.rolling(window).sum()
    return cmf

def compute_flow_proxy(df, ticker, window=FLOW_ZSCORE_WINDOW):
    """
    Calcula el Flow Proxy compuesto para un ticker.
    Formula: 0.30*flow_smooth + 0.35*obv_z + 0.35*cmf_z
    donde:
      flow_smooth = EWMA(10) de robust_zscore(ret*signed_volume_pressure, window=60)
      donde signed_volume_pressure = ret * close * volume
      obv_z = robust_zscore(OBV.pct_change(), window=60)  # NOTA: pct_change() sobre serie acumulativa puede generar outliers. Alternativa: obv.diff()
      cmf_z = robust_zscore(CMF(20), window=60)
    Retorna una Serie temporal con el Flow Proxy compuesto.
    """
    close = get_col(df, ticker, 'Close')
    volume = get_col(df, ticker, 'Volume')
    signed_volume_pressure = close * volume
    ret = close.pct_change(fill_method=None)
    flow = ret * signed_volume_pressure
    flow_proxy_z = robust_zscore(flow, window=window)
    flow_smooth = flow_proxy_z.ewm(span=FLOW_EWM_SPAN, min_periods=20).mean()
    # Componentes adicionales
    obv = compute_obv(df, ticker)
    obv_z = robust_zscore(obv.pct_change(fill_method=None), window=window)
    cmf = compute_cmf(df, ticker)
    cmf_z = robust_zscore(cmf, window=window)
    # Combinación: 30% proxy, 35% OBV, 35% CMF
    combined = 0.30 * flow_smooth + 0.35 * obv_z + 0.35 * cmf_z
    return combined


compute_flow_proxy.__doc__ = f"""
Formula: {FLOW_PROXY_WEIGHTS['flow_smooth']:.2f}*flow_smooth + {FLOW_PROXY_WEIGHTS['obv']:.2f}*obv_z + {FLOW_PROXY_WEIGHTS['cmf']:.2f}*cmf_z. flow_smooth = EWMA({FLOW_EWM_SPAN}) de robust_zscore(ret*signed_volume_pressure, window={FLOW_ZSCORE_WINDOW}). obv_z = robust_zscore(OBV.pct_change(), window={FLOW_ZSCORE_WINDOW}). cmf_z = robust_zscore(CMF({FLOW_CMF_WINDOW}), window={FLOW_ZSCORE_WINDOW}).
"""
def compute_price_momentum(df, ticker, window=MOMENTUM_PRICE_WINDOW):
    close = get_col(df, ticker, 'Close')
    return close.pct_change(periods=window, fill_method=None)

