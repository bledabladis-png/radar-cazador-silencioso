"""
wyckoff_detector.py – Análisis microestructural de fases Wyckoff (acumulación, spring, SOS)
No genera señales de trading, solo información complementaria.
"""

import pandas as pd
import numpy as np

def range_compression(df, window=20):
    """Compresión del rango (high-low) / close."""
    high_max = df["high"].rolling(window).max()
    low_min = df["low"].rolling(window).min()
    compression = (high_max - low_min) / df["close"]
    return compression

def absorption_score(df, window=20):
    """
    Score de absorción: volumen alto + precio plano.
    Versión mejorada con tanh para evitar outliers.
    """
    vol_mean = df["volume"].rolling(window).mean()
    vol_std = df["volume"].rolling(window).std()
    vol_z = (df["volume"] - vol_mean) / vol_std

    price_change = df["close"].pct_change(window)

    raw = vol_z - price_change
    absorption = np.tanh(raw)   # rango (-1, 1)
    return absorption

def detect_spring(df):
    """Falsa ruptura bajista + recuperación + volumen alto."""
    prev_low = df["low"].shift(1)
    vol_mean = df["volume"].rolling(20).mean()
    condition = (
        (df["low"] < prev_low) &
        (df["close"] > df["open"]) &
        (df["volume"] > vol_mean * 1.5)
    )
    return condition.astype(int)

def detect_sos(df):
    """Sign of Strength: ruptura de máximo 20d + volumen."""
    high_max = df["high"].rolling(20).max().shift(1)
    vol_mean = df["volume"].rolling(20).mean()
    condition = (
        (df["close"] > high_max) &
        (df["volume"] > vol_mean)
    )
    return condition.astype(int)

def trend_suppression(df, window=50):
    """Pendiente de MA50 pequeña vs volatilidad."""
    ma = df["close"].rolling(window).mean()
    ma_slope = ma.diff()
    ma_std = df["close"].rolling(window).std()
    suppression = abs(ma_slope) < (ma_std * 0.1)
    return suppression.astype(int)

def wyckoff_score(df):
    """Score combinado (0 a 1)."""
    comp = range_compression(df)
    absr = absorption_score(df)
    spring = detect_spring(df)
    sos = detect_sos(df)
    trend = trend_suppression(df)

    comp_score = 1 - comp.clip(upper=1)   # menor compresión = mejor
    # Normalizar absorción a [0,1] (cortamos en 2)
    absr_norm = absr.clip(lower=0, upper=2) / 2.0

    score = (
        0.25 * comp_score +
        0.25 * absr_norm +
        0.20 * spring +
        0.20 * sos +
        0.10 * trend
    )
    return score.clip(0, 1)

def classify_wyckoff_phase(df):
    """Clasificación de fase para la última fila."""
    if len(df) < 60:
        return "INSUFICIENTE"
    score = wyckoff_score(df).iloc[-1]
    spring = detect_spring(df).iloc[-1]
    sos = detect_sos(df).iloc[-1]

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