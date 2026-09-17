"""DT3 Fase 2: scoring interno de darkpool.

Funciones puras (sin side effects, sin IO).
Re-exportadas por indicators/darkpool.py para preservar API interna.
"""
import numpy as np
import pandas as pd

from config.settings import DARKPOOL_THRESHOLDS


def robust_zscore(series):
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return np.zeros(len(series))
    return (series - median) / (1.4826 * mad)


def rolling_percentile(series):
    last = series.iloc[-1]
    return (series < last).mean() * 100


def classify_darkpool(z):
    if z >= DARKPOOL_THRESHOLDS['extremadamente_alta']:
        return "Actividad ATS extremadamente alta"
    elif z >= DARKPOOL_THRESHOLDS['muy_alta']:
        return "Actividad ATS muy alta"
    elif z >= DARKPOOL_THRESHOLDS['alta']:
        return "Actividad ATS alta"
    elif z > DARKPOOL_THRESHOLDS['normal']:
        return "Actividad ATS normal"
    elif z > DARKPOOL_THRESHOLDS['baja']:
        return "Actividad ATS baja"
    elif z > DARKPOOL_THRESHOLDS['muy_baja']:
        return "Actividad ATS muy baja"
    else:
        return "Actividad ATS extremadamente baja"


def _compute_z_for_window(hist, window):
    """Calcula Z-Score robusto para una ventana especifica."""
    if len(hist) < window:
        return np.nan, np.nan, np.nan, "Sin historial suficiente"
    sub = hist.iloc[-window:].copy()
    sub['ratio_ewm'] = sub['ratio'].ewm(span=min(4, window//2)).mean()
    z_series = sub['ratio_ewm'].rolling(window).apply(lambda x: robust_zscore(pd.Series(x)).iloc[-1], raw=False)
    z = z_series.iloc[-1]
    percentile = rolling_percentile(sub['ratio_ewm'])
    momentum = z_series.ewm(span=min(4, window//2)).mean().iloc[-1]
    state = classify_darkpool(z)
    return z, momentum, percentile, state
