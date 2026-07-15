"""
wyckoff.py -- Analisis microestructural de fases Wyckoff v3.0
Score continuo con 4 estados: MARKUP, ACCUMULATION, RANGE, DISTRIBUTION.
No genera senales de trading, solo informacion complementaria.
"""
import pandas as pd
import numpy as np
from src.utils import robust_zscore, get_col

# ---------- COMPONENTES PRIMARIOS ----------

def range_width(df, ticker, window=20):
    high = get_col(df, ticker, 'High')
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    return (high.rolling(window).max() - low.rolling(window).min()) / close

def relative_volume(df, ticker, window=20):
    volume = get_col(df, ticker, 'Volume')
    return volume / volume.rolling(window).mean()

def effort_vs_result(df, ticker, window=20):
    close = get_col(df, ticker, 'Close')
    volume = get_col(df, ticker, 'Volume')
    price_move = close.pct_change(window).abs()
    volume_effort = volume.rolling(window).mean() / volume.rolling(60).mean()
    return price_move / (volume_effort + 1e-9)

def trend_component(df, ticker):
    close = get_col(df, ticker, 'Close')
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    return ma50 / ma200 - 1

# ---------- DETECTORES DE EVENTOS ----------

def detect_spring(df, ticker):
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    open_ = get_col(df, ticker, 'Open')
    volume = get_col(df, ticker, 'Volume')
    prev_low = low.shift(1)
    vol_mean = volume.rolling(20).mean()
    condition = (low < prev_low) & (close > open_) & (volume > vol_mean * 1.5)
    return condition.astype(int)

def detect_sos(df, ticker):
    high = get_col(df, ticker, 'High')
    close = get_col(df, ticker, 'Close')
    volume = get_col(df, ticker, 'Volume')
    high_max = high.rolling(20).max().shift(1)
    vol_mean = volume.rolling(20).mean()
    condition = (close > high_max) & (volume > vol_mean)
    return condition.astype(int)

# ---------- SCORE CONTINUO ----------

def tanh_normalize(series, window=60):
    z = robust_zscore(series, window)
    return np.tanh(z)

def wyckoff_score(df, ticker):
    """Score continuo [-1, 1] basado en 4 componentes independientes."""
    trend = trend_component(df, ticker)
    rw = range_width(df, ticker)
    rv = relative_volume(df, ticker)
    evr = effort_vs_result(df, ticker)
    rv_smooth = rv.rolling(5).mean()
    
    t_norm = tanh_normalize(trend)
    c_norm = -tanh_normalize(rw)
    v_norm = tanh_normalize(rv_smooth)
    e_norm = tanh_normalize(evr)
    
    return (0.35 * t_norm + 0.25 * c_norm + 0.20 * v_norm + 0.20 * e_norm)

# ---------- CLASIFICACIÓN UNIFICADA (4 ESTADOS) ----------

def wyckoff_structure_core(df, ticker):
    close = get_col(df, ticker, 'Close')
    if len(close) < 200:
        return "INSUFICIENT_DATA"
    score = wyckoff_score(df, ticker).dropna()
    if score.empty:
        return "RANGE"
    last = score.iloc[-1]
    if last > 0.3:
        return "MARKUP"
    elif last > 0:
        return "ACCUMULATION"
    elif last > -0.3:
        return "RANGE"
    else:
        return "DISTRIBUTION"

def classify_wyckoff_phase(df, ticker):
    """Clasificación unificada (usa el mismo score que structure_core)."""
    return wyckoff_structure_core(df, ticker)

# ---------- COMPATIBILIDAD HACIA ATRÁS ----------

range_compression = range_width
absorption_score = lambda df, ticker, window=20: relative_volume(df, ticker, window).clip(0, 2) / 2.0
trend_suppression = lambda df, ticker, window=50: (
    abs(get_col(df, ticker, 'Close').rolling(window).mean().diff()) < 
    (get_col(df, ticker, 'Close').rolling(window).std() * 0.1)
).astype(int)
