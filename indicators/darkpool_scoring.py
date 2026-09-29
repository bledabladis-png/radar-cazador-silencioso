"""DT3 Fase 2: scoring interno de darkpool.

Funciones puras (sin side effects, sin IO).
Re-exportadas por indicators/darkpool.py para preservar API interna.
"""
import numpy as np
import pandas as pd

from config.settings import DARKPOOL_THRESHOLDS


def robust_zscore(series):
    # K-DT3-RUNTIMEWARN: serie vacia -> Series vacia sin warnings de numpy
    # (antes: series.median() y np.median() sobre vacio emitidos como
    # RuntimeWarning "Mean of empty slice" + "invalid value in divide").
    #
    # A3.3-01 (2026-09-28): el consumidor (_compute_z_for_window) hace
    # .iloc[-1] sobre el retorno. Contrato implicito: la funcion debe
    # devolver una pd.Series indexada. La rama mad==0 devolvia
    # np.zeros(len(series)) (ndarray) -> .iloc[-1] lanzaba AttributeError
    # cuando MAD==0 en cualquier ventana. Devuelve Series con el mismo
    # indice que la entrada, consistente con las otras dos ramas.
    if len(series) == 0:
        return pd.Series([], dtype=float)
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - median) / (1.4826 * mad)


def rolling_percentile(series):
    last = series.iloc[-1]
    return (series < last).mean() * 100


def classify_darkpool(z):
    # C-08-preventivo (2026-09-29): cascada de comparaciones mapea
    # NaN/None/inf a un estado real sin declararlo. Mismo patron
    # que classify_pcr (options_metrics.py:73-75), que ya lleva
    # guard. Aqui z=NaN caia a "extremadamente baja" (todas las
    # comparaciones False) e inf/-inf idem. Bug latente, no
    # alcanzable con datos actuales (darkpool_history.csv sin NaN),
    # pero mismo contrato que classify_pcr por coherencia.
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
