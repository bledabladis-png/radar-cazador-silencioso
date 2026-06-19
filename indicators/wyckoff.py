"""
wyckoff.py -- Analisis microestructural de fases Wyckoff (acumulacion, distribucion, etc.)
Adaptado del Radar_Rotacion v4.0.5 para Macro_Sectorial v1.1.
No genera senales de trading, solo informacion complementaria.
"""

import pandas as pd
import numpy as np
from src.utils import robust_zscore, get_col


def range_compression(df, ticker, window=20):
    """Compresion del rango (high-low) / close."""
    high = get_col(df, ticker, 'High')
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    high_max = high.rolling(window).max()
    low_min = low.rolling(window).min()
    compression = (high_max - low_min) / close
    return compression


def absorption_score(df, ticker, window=20):
    """Score de absorcion: volumen alto + precio plano."""
    close = get_col(df, ticker, 'Close')
    volume = get_col(df, ticker, 'Volume')
    vol_mean = volume.rolling(window).mean()
    vol_std = volume.rolling(window).std()
    vol_z = (volume - vol_mean) / (vol_std + 1e-9)
    price_change = close.pct_change(window, fill_method=None)
    price_z = robust_zscore(price_change, window=60)
    raw = vol_z - price_z
    absorption = 1 / (1 + np.exp(-raw))
    return absorption


def detect_spring(df, ticker):
    """Falsa ruptura bajista + recuperacion + volumen alto."""
    high = get_col(df, ticker, 'High')
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    open_ = get_col(df, ticker, 'Open')
    volume = get_col(df, ticker, 'Volume')
    prev_low = low.shift(1)
    vol_mean = volume.rolling(20).mean()
    condition = (
        (low < prev_low) &
        (close > open_) &
        (volume > vol_mean * 1.5)
    )
    return condition.astype(int)


def detect_sos(df, ticker):
    """Sign of Strength: ruptura de maximo 20d + volumen."""
    high = get_col(df, ticker, 'High')
    close = get_col(df, ticker, 'Close')
    volume = get_col(df, ticker, 'Volume')
    high_max = high.rolling(20).max().shift(1)
    vol_mean = volume.rolling(20).mean()
    condition = (
        (close > high_max) &
        (volume > vol_mean)
    )
    return condition.astype(int)


def trend_suppression(df, ticker, window=50):
    """Pendiente de MA50 pequena vs volatilidad."""
    close = get_col(df, ticker, 'Close')
    ma = close.rolling(window).mean()
    ma_slope = ma.diff()
    ma_std = close.rolling(window).std()
    suppression = abs(ma_slope) < (ma_std * 0.1)
    return suppression.astype(int)


def wyckoff_score(df, ticker):
    """Score combinado (0 a 1)."""
    comp = range_compression(df, ticker)
    absr = absorption_score(df, ticker)
    spring = detect_spring(df, ticker)
    sos = detect_sos(df, ticker)
    trend = trend_suppression(df, ticker)

    comp_score = 1 - comp.clip(upper=1)
    absr_norm = absr.clip(lower=0, upper=2) / 2.0

    score = (
        0.25 * comp_score +
        0.25 * absr_norm +
        0.20 * spring +
        0.20 * sos +
        0.10 * trend
    )
    return score.clip(0, 1)


def wyckoff_structure_core(df, ticker):
    """
    Clasifica la fase Wyckoff para la ultima barra del ticker.
    Retorna: "MARKUP", "DISTRIBUTION", "ACCUMULATION", "RANGE", o "INSUFICIENT_DATA"
    """
    close = get_col(df, ticker, 'Close')
    if len(close) < 200:
        return "INSUFICIENT_DATA"

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    trend = (ma50.iloc[-1] / ma200.iloc[-1] - 1)

    vol = close.pct_change(fill_method=None).rolling(20).std().iloc[-1]
    vol_mean = close.pct_change(fill_method=None).rolling(20).std().mean()
    vol_norm = vol / (vol_mean + 1e-9) if pd.notna(vol_mean) and vol_mean != 0 else 1.0

    compression = range_compression(df, ticker).iloc[-1]
    wyckoff_sc = wyckoff_score(df, ticker).iloc[-1]

    if trend > 0.02 and vol_norm < 1 and wyckoff_sc > 0.6:
        return "MARKUP"
    elif trend < -0.02:
        return "DISTRIBUTION"
    elif compression < 0.3 and wyckoff_sc > 0.5:
        return "ACCUMULATION"
    else:
        return "RANGE"


def classify_wyckoff_phase(df, ticker):
    """Clasificacion de fase micro para la ultima fila."""
    close = get_col(df, ticker, 'Close')
    if len(close) < 60:
        return "INSUFICIENTE"
    score = wyckoff_score(df, ticker).iloc[-1]
    spring = detect_spring(df, ticker).iloc[-1]
    sos = detect_sos(df, ticker).iloc[-1]

    if spring == 1:
        return "SPRING"
    elif sos == 1:
        return "MARKUP INIT"
    elif score > 0.7:
        return "ACCUMULATION"
    elif score > 0.5:
        return "LATE ACCUMULATION"
    else:
        return "NEUTRAL"