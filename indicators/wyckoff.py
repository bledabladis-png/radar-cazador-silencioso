"""
wyckoff.py -- Analisis microestructural de fases Wyckoff v3.0
Score continuo con 4 estados: MARKUP, ACCUMULATION, RANGE, DISTRIBUTION.
No genera senales de trading, solo informacion complementaria.
"""
import pandas as pd
import numpy as np
from config.settings import (
    WYCKOFF_RANGE_WINDOW, WYCKOFF_VOLUME_WINDOW,
    WYCKOFF_TREND_FAST_MA, WYCKOFF_TREND_SLOW_MA, WYCKOFF_MIN_PERIODS
)
from src.utils import robust_zscore, get_col

# ---------- COMPONENTES PRIMARIOS ----------

def range_width(df, ticker, window=WYCKOFF_RANGE_WINDOW):
    high = get_col(df, ticker, 'High')
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    return (high.rolling(window).max() - low.rolling(window).min()) / close

def relative_volume(df, ticker, window=WYCKOFF_VOLUME_WINDOW):
    volume = get_col(df, ticker, 'Volume')
    return volume / (volume.rolling(window).mean() + 1e-9)

def effort_vs_result(df, ticker, window=WYCKOFF_VOLUME_WINDOW):
    close = get_col(df, ticker, 'Close')
    volume = get_col(df, ticker, 'Volume')
    price_move = close.pct_change(window).abs()
    volume_effort = volume.rolling(window).mean() / (volume.rolling(60).mean() + 1e-9)
    return price_move / (volume_effort + 1e-9)

def trend_component(df, ticker):
    close = get_col(df, ticker, 'Close')
    ma50 = close.rolling(WYCKOFF_TREND_FAST_MA).mean()
    ma200 = close.rolling(WYCKOFF_TREND_SLOW_MA).mean()
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

def wyckoff_score(df, ticker):
    """Score continuo [-1, 1] basado en 4 componentes independientes."""
    trend = trend_component(df, ticker)
    rw = range_width(df, ticker)
    rv = relative_volume(df, ticker)
    evr = effort_vs_result(df, ticker)
    rv_smooth = rv.rolling(5).mean()

    t_norm = np.tanh(robust_zscore(trend))
    c_norm = -np.tanh(robust_zscore(rw))
    v_norm = np.tanh(robust_zscore(rv_smooth))
    e_norm = np.tanh(robust_zscore(evr))

    return (0.35 * t_norm + 0.25 * c_norm + 0.20 * v_norm + 0.20 * e_norm)

# ---------- CLASIFICACION UNIFICADA (4 ESTADOS) ----------

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
    """Clasificacion unificada (wrapper para compatibilidad de API)."""
    return wyckoff_structure_core(df, ticker)
