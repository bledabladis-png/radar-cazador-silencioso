# global_stability_v4.py – Stability layer para Radar Global v4.0
import pandas as pd
import numpy as np

def signal_stability(series, window=12):
    """
    Calcula la estabilidad de una serie como 1 - (std reciente / std histórica).
    Valores cercanos a 1 = señal estable. < 0.5 = alta variabilidad.
    """
    if len(series.dropna()) < window:
        return np.nan
    recent_std = series.iloc[-window:].std()
    historic_std = series.dropna().rolling(252, min_periods=window).std().iloc[-1]
    if pd.isna(historic_std) or historic_std == 0:
        return np.nan
    if historic_std == 0:
        return 1.0
    stability = 1 - min(recent_std / historic_std, 1.0)
    return stability

def stability_flags(flow_rotation_series, risk_breadth_series, alignment_series):
    """
    Retorna una lista de avisos si alguna métrica principal es inestable.
    """
    flags = []
    rot_stab = signal_stability(flow_rotation_series)
    bread_stab = signal_stability(risk_breadth_series)
    align_stab = signal_stability(alignment_series)

    if rot_stab is not None and rot_stab < 0.5:
        flags.append("⚠️ Capital Rotation muestra alta variabilidad reciente.")
    if bread_stab is not None and bread_stab < 0.5:
        flags.append("⚠️ Risk Breadth muestra alta variabilidad reciente.")
    if align_stab is not None and align_stab < 0.5:
        flags.append("⚠️ Region Alignment muestra alta variabilidad reciente.")
    return flags

def regime_shift_detector(volatility_regime_series, threshold=2.5):
    """
    Detecta si el régimen de volatilidad actual está fuera de lo normal.
    Retorna True si la volatilidad actual supera 'threshold' desviaciones MAD.
    """
    if len(volatility_regime_series.dropna()) < 60:
        return False
    current = volatility_regime_series.iloc[-1]
    median = volatility_regime_series.rolling(60).median().iloc[-1]
    mad = (volatility_regime_series - median).abs().rolling(60).median().iloc[-1]
    if pd.isna(mad) or mad == 0:
        return False
    z = (current - median) / (1.4826 * mad)
    return abs(z) > threshold