# -*- coding: utf-8 -*-
"""Modulo Wyckoff v1 - Fases estructurales.

Contrato: docs/auditoria/wyckoff/01_contrato_semantico_v1.md

Este modulo implementa el contrato v1. El legacy (indicators/wyckoff.py)
permanece intacto y operativo. La migracion de consumidores es un paso
posterior (Fase 5d del plan).

Diferencias clave respecto al legacy:
  - 5 fases reales + INSUFFICIENT_DATA (no fallback RANGE).
  - ACCUMULATION con condiciones estructurales explicitas.
  - MARKDOWN como fase real.
  - effort_vs_result en forma clasica (effort - result).
  - stability como dispersion pura.
  - ALL_NAN nunca devuelve una fase de mercado.

No genera senales de trading.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    WYCKOFF_TREND_FAST_MA,
    WYCKOFF_TREND_SLOW_MA,
    WYCKOFF_ATR_WINDOW,
    WYCKOFF_VOLUME_ZSCORE_WINDOW,
    WYCKOFF_STRUCT_WEIGHT_TREND,
    WYCKOFF_STRUCT_WEIGHT_COMPRESSION,
    WYCKOFF_TACT_WEIGHT_VOLUME,
    WYCKOFF_TACT_WEIGHT_EFFORT,
    WYCKOFF_COMBINED_STRUCT_WEIGHT,
    WYCKOFF_COMBINED_TACT_WEIGHT,
)
from src.utils import robust_zscore, get_col


# --- Parametros del contrato v1 (ver 02_proveniencia) ---
STABILITY_MAD_WINDOW = 10       # sesiones
STABILITY_K = 1.0               # escala (PROPUESTO, calibrar en Fase 5b)
SPRING_VOLUME_MULT = 1.5        # spring: volumen > 1.5x MA20
SOS_VOLUME_MULT = 1.0           # sos: volumen > 1.0x MA20
EVENT_WINDOW = 20               # ventana para detectar eventos

# Umbrales de clasificacion (contrato §5)
T_NORM_STRONG = 0.30
T_NORM_WEAK = 0.30
C_NORM_COMPRESSION = 0.30
COMBINED_MARKUP = 0.30
COMBINED_DISTRIBUTION = -0.10
COMBINED_MARKDOWN = -0.30

# Constantes de fase
FASE_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
FASE_ACCUMULATION = "ACCUMULATION"
FASE_MARKUP = "MARKUP"
FASE_DISTRIBUTION = "DISTRIBUTION"
FASE_MARKDOWN = "MARKDOWN"
FASE_RANGE = "RANGE"
ALL_FASES = (FASE_ACCUMULATION, FASE_MARKUP, FASE_DISTRIBUTION,
             FASE_MARKDOWN, FASE_RANGE)


# =====================================================================
# Componentes primarios
# =====================================================================

def _atr_normalized(df, ticker, window=WYCKOFF_ATR_WINDOW):
    """ATR / Close. Alto = expansion; bajo = compresion."""
    high = get_col(df, ticker, 'High')
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean()
    return atr / close


def _trend_component(df, ticker):
    """MA_fast / MA_slow - 1. Positivo = tendencia alcista."""
    close = get_col(df, ticker, 'Close')
    ma_fast = close.rolling(WYCKOFF_TREND_FAST_MA,
                            min_periods=WYCKOFF_TREND_FAST_MA).mean()
    ma_slow = close.rolling(WYCKOFF_TREND_SLOW_MA,
                            min_periods=WYCKOFF_TREND_SLOW_MA).mean()
    return ma_fast / (ma_slow + 1e-9) - 1


def _volume_z(df, ticker):
    """Z-score robusto del volumen."""
    volume = get_col(df, ticker, 'Volume')
    return robust_zscore(volume, window=WYCKOFF_VOLUME_ZSCORE_WINDOW,
                         min_periods=20)


def _effort_vs_result(df, ticker, window=20):
    """Effort vs result (forma clasica Wyckoff).

    effort = z_robusto(Volume)
    result = z_robusto(|Close_t / Close_{t-window} - 1|)
    return tanh(effort - result)

    Positivo: esfuerzo > resultado -> absorcion.
    Negativo: resultado > esfuerzo -> movimiento sin esfuerzo.
    """
    close = get_col(df, ticker, 'Close')
    effort = _volume_z(df, ticker)
    price_move = (close / close.shift(window) - 1).abs()
    result = robust_zscore(price_move, window=60, min_periods=20)
    return np.tanh(effort - result)


# =====================================================================
# Score compuesto
# =====================================================================

def wyckoff_score(df, ticker):
    """Score compuesto y sus componentes.

    Returns:
        (combined, struct_score, tact_score, t_norm, c_norm, v_norm, e_norm)
        Cada elemento es una Series con DatetimeIndex.
    """
    # Estructural
    trend = _trend_component(df, ticker)
    compression = _atr_normalized(df, ticker, window=WYCKOFF_ATR_WINDOW)
    t_norm = np.tanh(robust_zscore(trend, window=200, min_periods=60))
    c_norm = -np.tanh(robust_zscore(compression, window=200, min_periods=60))
    struct_score = (
        WYCKOFF_STRUCT_WEIGHT_TREND * t_norm
        + WYCKOFF_STRUCT_WEIGHT_COMPRESSION * c_norm
    )

    # Tactico
    v_norm = np.tanh(_volume_z(df, ticker))
    e_norm = _effort_vs_result(df, ticker)
    tact_score = (
        WYCKOFF_TACT_WEIGHT_VOLUME * v_norm
        + WYCKOFF_TACT_WEIGHT_EFFORT * e_norm
    )

    # Compuesto
    combined = (
        WYCKOFF_COMBINED_STRUCT_WEIGHT * struct_score
        + WYCKOFF_COMBINED_TACT_WEIGHT * tact_score
    )

    return combined, struct_score, tact_score, t_norm, c_norm, v_norm, e_norm


def wyckoff_stability(combined, window=STABILITY_MAD_WINDOW, K=STABILITY_K):
    """Estabilidad temporal del score. Rango (-1, 1).

    stability = 1 - 2*tanh(score_mad / K)

    Donde score_mad = MAD(combined en ventana). NO depende del nivel.
    """
    mad = combined.rolling(window, min_periods=window).apply(
        lambda x: float(np.median(np.abs(x - np.median(x)))),
        raw=True,
    )
    return 1.0 - 2.0 * np.tanh(mad / (K + 1e-9))


# =====================================================================
# Clasificacion
# =====================================================================

def _has_sufficient_data(df, ticker):
    """True si hay al menos 200 observaciones validas de Close."""
    try:
        close = get_col(df, ticker, 'Close')
    except (KeyError, ValueError, TypeError):
        return False
    return close.dropna().shape[0] >= WYCKOFF_TREND_SLOW_MA


def classify_wyckoff_phase(df, ticker):
    """Clasifica la fase segun el contrato v1 (§5).

    Returns:
        Uno de: MARKUP | ACCUMULATION | RANGE | DISTRIBUTION | MARKDOWN
                INSUFFICIENT_DATA
    """
    if not _has_sufficient_data(df, ticker):
        return FASE_INSUFFICIENT_DATA

    try:
        combined, struct_score, tact_score, t_norm, c_norm, v_norm, e_norm = (
            wyckoff_score(df, ticker)
        )
    except (KeyError, ValueError, TypeError, IndexError):
        return FASE_INSUFFICIENT_DATA

    combined_clean = combined.dropna()
    if combined_clean.empty:
        return FASE_INSUFFICIENT_DATA

    last = combined_clean.iloc[-1]
    t_last = t_norm.dropna().iloc[-1] if not t_norm.dropna().empty else np.nan
    c_last = c_norm.dropna().iloc[-1] if not c_norm.dropna().empty else np.nan
    tact_last = (tact_score.dropna().iloc[-1]
                 if not tact_score.dropna().empty else np.nan)

    if pd.isna(t_last) or pd.isna(c_last) or pd.isna(tact_last):
        return FASE_INSUFFICIENT_DATA

    # --- MARKUP (§5.3) ---
    if t_last > T_NORM_STRONG and c_last < C_NORM_COMPRESSION and last > COMBINED_MARKUP:
        return FASE_MARKUP

    # --- MARKDOWN (§5.5) ---
    if t_last < -T_NORM_STRONG and c_last < 0 and last < COMBINED_MARKDOWN:
        return FASE_MARKDOWN

    # --- ACCUMULATION (§5.2) ---
    # 1) tendencia previa negativa o lateral-baja
    trend_ok = (t_last < 0) or (abs(t_last) < T_NORM_WEAK)
    # 2) compresion alta
    compression_ok = c_last > C_NORM_COMPRESSION
    # 3) base/rango: ATR actual <= mediana historica (aprox. via c_norm > 0)
    base_ok = c_last > 0
    # 4) confirmacion tactica
    tact_ok = tact_last > 0
    if trend_ok and compression_ok and base_ok and tact_ok:
        return FASE_ACCUMULATION

    # --- DISTRIBUTION (§5.4) ---
    trend_dist_ok = (t_last > 0) or (abs(t_last) < T_NORM_WEAK)
    compression_dist_ok = c_last > C_NORM_COMPRESSION
    if trend_dist_ok and compression_dist_ok and last < COMBINED_DISTRIBUTION:
        return FASE_DISTRIBUTION

    # --- RANGE (§5.6) ---
    return FASE_RANGE


# =====================================================================
# Eventos
# =====================================================================

def detect_spring(df, ticker):
    """Spring inspirado en Wyckoff (contrato §6.1)."""
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    open_ = get_col(df, ticker, 'Open')
    volume = get_col(df, ticker, 'Volume')
    vol_mean = volume.rolling(EVENT_WINDOW, min_periods=1).mean()
    condition = (
        (low < low.shift(1))
        & (close > open_)
        & (volume > vol_mean * SPRING_VOLUME_MULT)
    )
    return condition.astype(int)


def detect_sos(df, ticker):
    """Sign of strength inspirado en Wyckoff (contrato §6.2)."""
    high = get_col(df, ticker, 'High')
    close = get_col(df, ticker, 'Close')
    volume = get_col(df, ticker, 'Volume')
    high_max = high.rolling(EVENT_WINDOW, min_periods=1).max().shift(1)
    vol_mean = volume.rolling(EVENT_WINDOW, min_periods=1).mean()
    condition = (close > high_max) & (volume > vol_mean * SOS_VOLUME_MULT)
    return condition.astype(int)


# =====================================================================
# Preparacion de input
# =====================================================================

def build_ticker_df(df, ticker):
    """DataFrame OHLCV con mascaras de validez por campo (contrato §7.3).

    Regla del contrato v1:
      - Close es el campo autoritativo. Filas con Close NaN se eliminan.
      - Open/High/Low NaN se rellenan con Close (mismo dia).
      - Volume NaN se rellena con 0 (sin volumen observado ese dia).
      - Filas sin ninguna informacion se eliminan.
    """
    raw = pd.DataFrame({
        'Open': get_col(df, ticker, 'Open'),
        'High': get_col(df, ticker, 'High'),
        'Low': get_col(df, ticker, 'Low'),
        'Close': get_col(df, ticker, 'Close'),
        'Volume': get_col(df, ticker, 'Volume'),
    })
    # Close es autoritativo
    raw = raw.dropna(subset=['Close'])
    # Open/High/Low: rellenar con Close si NaN
    for col in ('Open', 'High', 'Low'):
        raw[col] = raw[col].fillna(raw['Close'])
    # Volume: 0 si NaN
    raw['Volume'] = raw['Volume'].fillna(0.0)
    return raw
