"""
normalization.py - Funciones de normalización y persistencia para señales financieras.
"""

import numpy as np
import pandas as pd

def rolling_zscore(series, window=252):
    """
    Calcula el z-score rodante de una serie, con protección contra desviación cero.
    """
    mean = series.rolling(window, min_periods=window//2).mean()
    std = series.rolling(window, min_periods=window//2).std()
    # Evitar división por cero: donde std sea 0, poner NaN (luego se rellenará)
    std = std.replace(0, np.nan)
    z = (series - mean) / std
    return z

def normalize_signal(series, window=252, scale=2.0):
    """
    Normaliza una señal:
    1. rolling z-score
    2. tanh(z / scale)  -> resultado en [-1, 1]
    """
    z = rolling_zscore(series, window)
    normalized = np.tanh(z / scale)
    return normalized

def persistence_factor(series, N):
    """
    Calcula un factor de persistencia (0 a 1) basado en días consecutivos con el mismo signo.
    series: pd.Series con valores numéricos.
    N: número de días requeridos para persistencia completa.
    """
    signos = np.sign(series)
    count = 0
    factores = []
    for i in range(len(series)):
        if i == 0:
            count = 1
        else:
            if signos.iloc[i] == signos.iloc[i-1]:
                count += 1
            else:
                count = 1
        factor = min(1.0, count / N)
        factores.append(factor)
    return pd.Series(factores, index=series.index)

def robust_scale(series, window=252, epsilon=1e-6):
    """
    Calcula un factor de escala robusto usando la desviación absoluta mediana (MAD).
    Retorna NaN mientras no haya al menos 'window' observaciones.
    Se añade epsilon para evitar divisiones por cero.
    """
    rolling_median = series.rolling(window, min_periods=window).median()
    abs_dev = (series - rolling_median).abs()
    mad = abs_dev.rolling(window, min_periods=window).median()
    scale = mad * 1.4826 + epsilon
    return scale

def normalize_module_series(series, window=252, method='zscore_tanh'):
    """
    Normaliza una serie de un módulo (opcional).
    """
    if method == 'zscore_tanh':
        z = rolling_zscore(series, window)
        return np.tanh(z)
    elif method == 'zscore':
        return rolling_zscore(series, window)
    else:
        raise ValueError(f"Método no soportado: {method}")