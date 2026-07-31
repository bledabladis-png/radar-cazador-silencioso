"""
wyckoff.py -- Analisis microestructural de fases Wyckoff v4.1
Score continuo con 4 estados: MARKUP, ACCUMULATION, RANGE, DISTRIBUTION.
No genera senales de trading, solo informacion complementaria.
"""
import pandas as pd
import numpy as np
from config.settings import (
    WYCKOFF_RANGE_WINDOW, WYCKOFF_VOLUME_WINDOW,
    WYCKOFF_TREND_FAST_MA, WYCKOFF_TREND_SLOW_MA,
    WYCKOFF_THRESHOLD_MARKUP, WYCKOFF_THRESHOLD_ACCUMULATION, WYCKOFF_THRESHOLD_DISTRIBUTION,
    WYCKOFF_ATR_WINDOW, WYCKOFF_VOLUME_ZSCORE_WINDOW,
    WYCKOFF_WEIGHT_TREND, WYCKOFF_WEIGHT_RANGE, WYCKOFF_WEIGHT_VOLUME, WYCKOFF_WEIGHT_EFFORT
)
from src.utils import robust_zscore, get_col

# ---------- COMPONENTES PRIMARIOS ----------

def atr_normalized(df, ticker, window=WYCKOFF_ATR_WINDOW):
    high = get_col(df, ticker, 'High')
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr / close

def range_width(df, ticker, window=WYCKOFF_RANGE_WINDOW):
    # Mantenido por compatibilidad, pero ya no se usa en el score v4.1
    high = get_col(df, ticker, 'High')
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    return (high.rolling(window).max() - low.rolling(window).min()) / close

def relative_volume(df, ticker, window=WYCKOFF_VOLUME_WINDOW):
    # v3.16 legacy
    volume = get_col(df, ticker, 'Volume')
    return volume / (volume.rolling(window).mean() + 1e-9)

def relative_volume_v41(df, ticker, window=WYCKOFF_VOLUME_ZSCORE_WINDOW):
    volume = get_col(df, ticker, 'Volume')
    return robust_zscore(volume, window=window)

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

# ---------- SCORE CONTINUO v4.1 ----------

def wyckoff_score(df, ticker):
    trend = trend_component(df, ticker)
    compression = atr_normalized(df, ticker)
    volume = relative_volume_v41(df, ticker)
    effort = effort_vs_result(df, ticker)

    t_norm = np.tanh(robust_zscore(trend))
    c_norm = -np.tanh(robust_zscore(compression))
    v_norm = np.tanh(robust_zscore(volume))
    e_norm = np.tanh(robust_zscore(effort))

    score = (
        WYCKOFF_WEIGHT_TREND * t_norm +
        WYCKOFF_WEIGHT_RANGE * c_norm +
        WYCKOFF_WEIGHT_VOLUME * v_norm +
        WYCKOFF_WEIGHT_EFFORT * e_norm
    )
    return score, t_norm, c_norm, v_norm, e_norm

def wyckoff_confidence(t_norm, c_norm, v_norm, e_norm):
    components = np.array([t_norm.iloc[-1], c_norm.iloc[-1], v_norm.iloc[-1], e_norm.iloc[-1]])
    dispersion = float(np.std(components))
    confidence = 1.0 / (1.0 + dispersion)
    return confidence, dispersion

# ---------- CLASIFICACION UNIFICADA ----------

def wyckoff_structure_core(df, ticker):
    close = get_col(df, ticker, 'Close')
    if len(close) < 200:
        return "INSUFICIENT_DATA"
    score, t_norm, c_norm, v_norm, e_norm = wyckoff_score(df, ticker)
    score_clean = score.dropna()
    if score_clean.empty:
        return "RANGE"
    last = score_clean.iloc[-1]
    if last > WYCKOFF_THRESHOLD_MARKUP:
        return "MARKUP"
    elif last > WYCKOFF_THRESHOLD_ACCUMULATION:
        return "ACCUMULATION"
    elif last > WYCKOFF_THRESHOLD_DISTRIBUTION:
        return "RANGE"
    else:
        return "DISTRIBUTION"

def classify_wyckoff_phase(df, ticker):
    return wyckoff_structure_core(df, ticker)
