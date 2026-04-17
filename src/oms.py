# src/oms.py
# Options Market Structure Layer (OMS v1.1)
# Basado en datos observables de CBOE y HKEX.

import numpy as np
import pandas as pd

# =========================
# UTILIDAD BASE
# =========================
def robust_zscore(series, window=60):
    median = series.rolling(window).median()
    mad = (series - median).abs().rolling(window).median()
    z = (series - median) / (1.4826 * mad + 1e-9)
    return z.clip(-5, 5)

def normalize(series, window):
    """Normaliza una serie a [-1,1] usando robust_zscore + tanh."""
    z = robust_zscore(series, window)
    return np.tanh(z)

# =========================
# 1. SENTIMENT (HKEX PCR)
# =========================
def compute_sentiment(pcr_series):
    # PCR alto → miedo → señal contrarian positiva
    return normalize(-pcr_series, window=60)

# =========================
# 2. ACTIVITY HEAT (CBOE)
# =========================
def compute_activity_heat(df):
    """
    df: DataFrame con columnas 'Trade Date' y 'Volume'.
    Retorna Serie con actividad normalizada.
    """
    volume = df.groupby('Trade Date')['Volume'].sum()
    # Ratio sobre media móvil de 60 días (elimina tendencia secular)
    rel = volume / volume.rolling(60, min_periods=1).mean()
    return normalize(rel, window=60)

# =========================
# 3. FRAGMENTATION (HHI interno)
# =========================
def compute_fragmentation(df):
    """
    df: DataFrame con columnas 'Trade Date', 'Underlying', 'Volume'.
    Retorna Serie con fragmentación normalizada.
    """
    daily = df.groupby(['Trade Date', 'Underlying'])['Volume'].sum()
    total = daily.groupby(level=0).sum()
    share = daily / total
    hhi = share.groupby(level=0).apply(lambda x: (x**2).sum())
    # Suavizado estructural para reducir ruido
    hhi = hhi.rolling(5, min_periods=1).mean()
    return normalize(hhi, window=120)

# =========================
# 4. OMS FINAL
# =========================
def compute_oms(pcr_series, df_volume):
    """
    pcr_series: Serie de PCR de HKEX (indexada por fecha).
    df_volume: DataFrame de CBOE con columnas 'Trade Date', 'Underlying', 'Volume'.
    Retorna DataFrame con columnas: sentiment, activity, fragmentation, oms.
    """
    sentiment = compute_sentiment(pcr_series)
    activity = compute_activity_heat(df_volume)
    frag = compute_fragmentation(df_volume)

    # Alinear índices
    idx = sentiment.index.intersection(activity.index).intersection(frag.index)
    sentiment = sentiment.reindex(idx).fillna(0)
    activity = activity.reindex(idx).fillna(0)
    frag = frag.reindex(idx).fillna(0)

    oms = (0.40 * sentiment + 0.35 * activity + 0.25 * frag).clip(-1, 1)

    return pd.DataFrame({
        "sentiment": sentiment,
        "activity": activity,
        "fragmentation": frag,
        "oms": oms
    })

def classify_oms(x):
    if x > 0.5:
        return "ESTABLE"
    elif x < -0.5:
        return "FRÁGIL"
    else:
        return "NEUTRO"

def oms_modifier(oms_value):
    if oms_value > 0.5:
        return {"risk_bias": 1.10, "confidence_boost": 1.05}
    elif oms_value < -0.5:
        return {"risk_bias": 0.85, "confidence_boost": 0.90}
    else:
        return {"risk_bias": 1.00, "confidence_boost": 1.00}