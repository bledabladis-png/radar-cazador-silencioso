import pandas as pd
import numpy as np
from config.settings import FLOW_ZSCORE_WINDOW, FLOW_EWM_SPAN, FLOW_CMF_WINDOW, MOMENTUM_SHARPE_WINDOW, MOMENTUM_PRICE_WINDOW
from src.utils import robust_zscore, get_col

def compute_returns(df, tickers):
    # G4: sin ffill a proposito (a diferencia de compute_obv/compute_cmf/
    # compute_flow_proxy). Rellenar Close antes de pct_change generaria
    # retornos 0 artificiales en festivos locales (ej: BME en festivo,
    # Xetra abierto). Preferimos NaN para que el consumidor filtre
    # (sector_regime usa dropna/min_periods aguas abajo).
    returns = pd.DataFrame()
    for t in tickers:
        try:
            close = get_col(df, t, 'Close')
            returns[t] = close.pct_change(fill_method=None)
        except KeyError:
            pass
    return returns

def momentum_score(returns, window=MOMENTUM_SHARPE_WINDOW, min_periods=None):
    """Momentum tipo Sharpe con tolerancia a huecos."""

    if min_periods is None:
        min_periods = max(window // 3, 10)
    ret = returns.rolling(window, min_periods=min_periods).mean() * window
    vol = returns.rolling(window, min_periods=min_periods).std()
    return ret / (vol + 1e-9)

def normalize_momentum(score_series):
    return np.tanh(robust_zscore(score_series, 60))

def compute_obv(df, ticker):
    close = get_col(df, ticker, 'Close').ffill()
    volume = get_col(df, ticker, 'Volume').ffill()
    sign = np.sign(close.diff())
    obv = (sign * volume).cumsum()
    return obv

def compute_cmf(df, ticker, window=FLOW_CMF_WINDOW):
    high = get_col(df, ticker, 'High').ffill()
    low = get_col(df, ticker, 'Low').ffill()
    close = get_col(df, ticker, 'Close').ffill()
    volume = get_col(df, ticker, 'Volume').ffill()
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
      obv_z = robust_zscore(OBV.diff(), window=60)  # diff() es la magnitud correcta: pct_change() explota cuando OBV cruza por cero
      cmf_z = robust_zscore(CMF(20), window=60)
    Retorna una Serie temporal con el Flow Proxy compuesto.
    """
    close = get_col(df, ticker, 'Close').ffill()
    volume = get_col(df, ticker, 'Volume').ffill()
    signed_volume_pressure = close * volume
    ret = close.pct_change(fill_method=None)
    flow = ret * signed_volume_pressure
    flow_proxy_z = robust_zscore(flow, window=window)
    flow_smooth = flow_proxy_z.ewm(span=FLOW_EWM_SPAN, min_periods=20).mean()
    # Componentes adicionales
    obv = compute_obv(df, ticker)
    obv_z = robust_zscore(obv.diff(), window=window)  # D2: diff() evita outliers cuando OBV cruza por cero
    cmf = compute_cmf(df, ticker)
    cmf_z = robust_zscore(cmf, window=window)
    # Combinación: 30% proxy, 35% OBV, 35% CMF
    combined = 0.30 * flow_smooth + 0.35 * obv_z + 0.35 * cmf_z
    return combined

def compute_price_momentum(df, ticker, window=MOMENTUM_PRICE_WINDOW):
    close = get_col(df, ticker, 'Close')
    return close.pct_change(periods=window, fill_method=None)

