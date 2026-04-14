import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import tickers

# ============================================================
# FUNCIONES DE NORMALIZACIÓN ROBUSTA
# ============================================================
def robust_zscore(series, window=60, min_periods=20):
    median = series.rolling(window, min_periods=min_periods).median()
    mad = (series - median).abs().rolling(window, min_periods=min_periods).median()
    mad = mad.replace(0, np.nan)
    robust_z = (series - median) / (1.4826 * mad)
    robust_z = robust_z.clip(-5, 5)
    return robust_z

# ============================================================
# FUNCIONES DE BREADTH Y FEATURES ORIGINALES
# ============================================================
def weighted_breadth(df, sectors, weights=None):
    # Si no se proporcionan pesos, usar pesos iguales para todos los sectores
    if weights is None:
        n = len(sectors)
        weights = {sec: 1.0/n for sec in sectors}
    signal = 0
    for sec in sectors:
        if sec not in df.columns or sec not in weights:
            continue
        w = weights[sec]
        ma = df[sec].rolling(100).mean()
        above = (df[sec] > ma).astype(int)
        signal += w * (above - 0.5)
    return signal

def compute_features(df):
    sectors = tickers['sectors']
    benchmark = tickers['benchmark']
    vix = tickers['vix']
    features = pd.DataFrame(index=df.index)
    raw_breadth = weighted_breadth(df, sectors)
    features['breadth_signal'] = -raw_breadth
    rel_momentum = {}
    for sec in sectors:
        rel = df[sec] / df[benchmark]
        rel_momentum[sec] = np.log(rel).diff(20)
    rel_df = pd.DataFrame(rel_momentum)
    features['sector_dispersion'] = rel_df.std(axis=1)
    vix_mean = df[vix].rolling(100).mean()
    vix_std = df[vix].rolling(100).std()
    features['vix_z'] = (df[vix] - vix_mean) / vix_std
    features['stress'] = (features['vix_z'] > 1.5).astype(int)
    features = features.dropna()
    return features

# ============================================================
# FUNCIONES DE FLUJOS (ETF FLOWS)
# ============================================================
def compute_etf_flows_robust(df, sectors, window=60):
    flows = {}
    for s in sectors:
        price = df[s]
        dollar_vol = df[f"{s}_dollar_vol_smoothed"]
        ret = price.pct_change()
        flow = ret * dollar_vol
        flow_z = robust_zscore(flow, window=window)
        flows[s] = flow_z
    return pd.DataFrame(flows)

def compute_etf_flows_original(df, sectors, window=60):
    flows = {}
    for s in sectors:
        price = df[s]
        dollar_vol = df[f"{s}_dollar_vol"]
        ret = price.pct_change()
        flow = ret * dollar_vol
        mean = flow.rolling(window, min_periods=20).mean()
        std = flow.rolling(window, min_periods=20).std()
        flow_z = (flow - mean) / std
        flows[s] = flow_z
    return pd.DataFrame(flows)

# Por compatibilidad, alias
compute_etf_flows = compute_etf_flows_robust

def compute_flow_momentum(flows_z, span=10):
    return flows_z.ewm(span=span).mean()

def compute_flow_dispersion(flow_momentum):
    return flow_momentum.std(axis=1)

def compute_flow_breadth(flow_momentum):
    positive = (flow_momentum > 0).sum(axis=1)
    total = flow_momentum.shape[1]
    return positive / total

def compute_flow_acceleration(flow_mom, window=5):
    return flow_mom - flow_mom.rolling(window).mean()

def compute_volume_zscore(df, sectors, window=20):
    vol_z = pd.DataFrame(index=df.index)
    for sec in sectors:
        dollar_vol = df[f"{sec}_dollar_vol_smoothed"]
        mean = dollar_vol.rolling(window).mean()
        std = dollar_vol.rolling(window).std()
        vol_z[sec] = (dollar_vol - mean) / std
    return vol_z

def compute_price_zscore(df, sectors, benchmark='SPY', window=60):
    price_z = pd.DataFrame(index=df.index)
    for sec in sectors:
        rel = df[sec] / df[benchmark]
        mean = rel.rolling(window, min_periods=20).mean()
        std = rel.rolling(window, min_periods=20).std()
        price_z[sec] = (rel - mean) / std
    return price_z

def compute_acceleration_zscore(flow_acc_df, window=20):
    acc_z = pd.DataFrame(index=flow_acc_df.index)
    for col in flow_acc_df.columns:
        mean = flow_acc_df[col].rolling(window, min_periods=10).mean()
        std = flow_acc_df[col].rolling(window, min_periods=10).std()
        acc_z[col] = (flow_acc_df[col] - mean) / std
    return acc_z