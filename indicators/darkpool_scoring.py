"""DT3 Fase 2: scoring interno de darkpool.

Funciones puras (sin side effects, sin IO).
Re-exportadas por indicators/darkpool.py para preservar API interna.

Rediseño K-DT3-AUDIT-01 (2026-10-06):
- Se elimina el contrato antiguo `_compute_z_for_window` que devolvia
  (z, momentum, percentile, state). Bug: el `momentum` se calculaba
  sobre una serie de un solo valor no-NaN (el ultimo de la rolling),
  por lo que era identico a `z`. El `percentile` se calculaba sobre
  `ratio_ewm` sin suavizar, distinto de la base del z.
- Se sustituye por `compute_window_stats(hist, window)` que devuelve
  un dict `{z, percentile, state}` con z y percentile calculados
  sobre la MISMA serie suavizada.
"""
import numpy as np
import pandas as pd

from config.settings import DARKPOOL_THRESHOLDS


def robust_zscore(series):
    """Z-score robusto (mediana + MAD) sobre una serie.

    Contrato:
        - serie vacia -> Series vacia (mismo indice)
        - NaN originales permanecen NaN
        - mad == 0 -> todos ceros (mismo indice)
        - resto -> (x - mediana) / (1.4826 * mad) sobre valores no-NaN
    """
    if len(series) == 0:
        return pd.Series([], dtype=float)
    valid = series.dropna()
    if len(valid) == 0:
        return pd.Series(np.full(len(series), np.nan), index=series.index)
    median = valid.median()
    mad = np.median(np.abs(valid - median))
    if mad == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - median) / (1.4826 * mad)


def rolling_percentile(series):
    """Percentil del ultimo valor sobre la serie (0-100)."""
    if len(series) == 0:
        return np.nan
    last = series.iloc[-1]
    if pd.isna(last):
        return np.nan
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return float((valid < last).mean() * 100)


def classify_darkpool(z):
    """Clasifica el z-score robusto en estados categoricos."""
    if z is None or not np.isfinite(z):
        return "Sin historial suficiente"
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


def compute_window_stats(hist, window):
    """Stats robustos para una ventana de N semanas.

    Contrato:
        - Si len(hist) < window: {z: nan, percentile: nan, state: "Sin historial suficiente"}.
        - Si ventana valida: z y percentile calculados sobre la serie
          `ratio` suavizada con EWM(span=min(4, window//2)) restringida
          a las ultimas `window` filas. `state` derivado del z.
        - z es el z-score robusto del ULTIMO valor de la serie suavizada.
        - percentile es el percentil del ULTIMO valor sobre la misma serie.

    Devuelve dict: {z: float, percentile: float, state: str}
    """
    nan_result = {'z': np.nan, 'percentile': np.nan,
                  'state': 'Sin historial suficiente'}
    if hist is None or len(hist) < window:
        return nan_result
    sub = hist.iloc[-window:].copy()
    if 'ratio' not in sub.columns:
        return nan_result
    span = min(4, max(2, window // 2))
    smoothed = sub['ratio'].ewm(span=span).mean().dropna()
    if len(smoothed) == 0:
        return nan_result
    median = smoothed.median()
    mad = np.median(np.abs(smoothed - median))
    if mad == 0:
        z = 0.0
    else:
        z = float((smoothed.iloc[-1] - median) / (1.4826 * mad))
    last = smoothed.iloc[-1]
    percentile = float((smoothed < last).mean() * 100)
    return {'z': z, 'percentile': percentile, 'state': classify_darkpool(z)}
