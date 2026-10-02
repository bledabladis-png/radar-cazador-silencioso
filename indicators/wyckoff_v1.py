# -*- coding: utf-8 -*-
"""Modulo Wyckoff v1.5 - Fases estructurales con precedente selectivo.

Contrato: docs/auditoria/wyckoff/01_contrato_semantico_v1_1.md
Revision v1 -> v1.1: docs/auditoria/wyckoff/04_revision_contrato_v1_1.md

Cambios respecto a v1.0:
  - D1: MARKUP sin condicion c_norm < 0.30 (incompatible con bull market
        ordenado).
  - D2: MARKUP sin veto de combined. Usa struct_score como umbral.
  - D3: ACCUMULATION sin exigir tact > 0 (opcional).
  - Maquina de estados con precedente estructural historico.
  - Precedente sobre variables continuas (struct_score, t_norm), NO sobre
    etiquetas de fase previas. Evita circularidad.
  - Regla de continuidad: MARKUP -> ACCUMULATION no permitida sin
    precedente compatible.

Legacy (indicators/wyckoff.py) permanece intacto.
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
    WYCKOFF_T_NORM_K,
)
from src.utils import robust_zscore, get_col


# =====================================================================
# Parametros del contrato v1.1 (ver 02_proveniencia)
# =====================================================================

# Ventanas
STABILITY_MAD_WINDOW = 10                # sesiones
STABILITY_K = 1.0                        # escala (PROPUESTO, calibrar Fase 5b)
PRECEDENT_WINDOW = 60                    # sesiones (PROPUESTO, calibrar Fase 5b)
SPRING_VOLUME_MULT = 1.5
SOS_VOLUME_MULT = 1.0
EVENT_WINDOW = 20

# Umbrales estructurales actuales
T_NORM_STRONG = 0.30                     # t_norm fuerte
T_NORM_WEAK = 0.30                       # t_norm lateral (banda)
C_NORM_COMPRESSION = 0.30                # compresion alta
STRUCT_STRONG = 0.30                     # struct_score fuerte
STRUCT_WEAK = -0.30                      # struct_score debil
STRUCT_BASE_LOW = -0.20                  # banda de base: limite inferior
STRUCT_BASE_HIGH = 0.20                  # banda de base: limite superior
STRUCT_DETERIORO = -0.10                 # para DISTRIBUTION
C_NORM_DISTR_MIN = -0.10                 # v1.4: compresion no extrema

# Umbrales de precedente (PROPUESTOS, calibrar Fase 5b)
PREC_STRUCT_WEAK = -0.20                 # hubo debilidad reciente
PREC_STRUCT_STRONG = 0.30                # hubo fortaleza reciente
PREC_STRUCT_NEG = -0.30                  # hubo bajista reciente

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
    close = get_col(df, ticker, 'Close')
    ma_fast = close.rolling(WYCKOFF_TREND_FAST_MA,
                            min_periods=WYCKOFF_TREND_FAST_MA).mean()
    ma_slow = close.rolling(WYCKOFF_TREND_SLOW_MA,
                            min_periods=WYCKOFF_TREND_SLOW_MA).mean()
    return ma_fast / (ma_slow + 1e-9) - 1


def _volume_z(df, ticker):
    volume = get_col(df, ticker, 'Volume')
    return robust_zscore(volume, window=WYCKOFF_VOLUME_ZSCORE_WINDOW,
                         min_periods=20)


def _effort_vs_result(df, ticker, window=20):
    close = get_col(df, ticker, 'Close')
    effort = _volume_z(df, ticker)
    price_move = (close / close.shift(window) - 1).abs()
    result = robust_zscore(price_move, window=60, min_periods=20)
    return np.tanh(effort - result)


# =====================================================================
# Score compuesto
# =====================================================================

def wyckoff_score(df, ticker):
    trend = _trend_component(df, ticker)
    compression = _atr_normalized(df, ticker, window=WYCKOFF_ATR_WINDOW)
    # v1.3 (P1 critico): t_norm = tanh(trend / K). Mide nivel de
    # tendencia escalado. NO usar robust_zscore: mide desviacion de
    # regimen, no nivel.
    t_norm = np.tanh(trend / (WYCKOFF_T_NORM_K + 1e-9))
    c_norm = -np.tanh(robust_zscore(compression, window=200, min_periods=60))
    struct_score = (
        WYCKOFF_STRUCT_WEIGHT_TREND * t_norm
        + WYCKOFF_STRUCT_WEIGHT_COMPRESSION * c_norm
    )

    v_norm = np.tanh(_volume_z(df, ticker))
    e_norm = _effort_vs_result(df, ticker)
    tact_score = (
        WYCKOFF_TACT_WEIGHT_VOLUME * v_norm
        + WYCKOFF_TACT_WEIGHT_EFFORT * e_norm
    )

    combined = (
        WYCKOFF_COMBINED_STRUCT_WEIGHT * struct_score
        + WYCKOFF_COMBINED_TACT_WEIGHT * tact_score
    )

    return combined, struct_score, tact_score, t_norm, c_norm, v_norm, e_norm


def wyckoff_stability(combined, window=STABILITY_MAD_WINDOW, K=STABILITY_K):
    mad = combined.rolling(window, min_periods=window).apply(
        lambda x: float(np.median(np.abs(x - np.median(x)))),
        raw=True,
    )
    return 1.0 - 2.0 * np.tanh(mad / (K + 1e-9))


# =====================================================================
# Precedente estructural (contrato v1.1 §4)
# =====================================================================

def _compute_precedent(struct_clean, t_norm_clean, window=PRECEDENT_WINDOW):
    """Calcula variables de precedente sobre ventana [t-N, t-1].

    Contrato v1.5 §4 (formalizacion): struct_max(t) := max(struct_score
    en [t-N, t-1]). Estrictamente historico, sin usar datos posteriores
    a t-1. Test de no-look-ahead en test_distribution_no_lookahead.

    Solo usa datos <= t-1 (excluye t).

    Args:
        struct_clean: Series de struct_score sin NaN.
        t_norm_clean: Series de t_norm sin NaN.
        window: N (ventana de precedente).

    Returns:
        dict con struct_min, struct_max, struct_mean, t_norm_min, t_norm_max.
        None si no hay suficientes datos (< window+1).
    """
    n_struct = len(struct_clean)
    if n_struct < window + 1:
        return None

    # [t-N, t-1]: excluye la ultima observacion (t)
    struct_window = struct_clean.iloc[-(window + 1):-1]
    t_norm_window = t_norm_clean.iloc[-(window + 1):-1] if len(t_norm_clean) >= window + 1 else pd.Series(dtype=float)

    out = {
        'struct_min': float(struct_window.min()),
        'struct_max': float(struct_window.max()),
        'struct_mean': float(struct_window.mean()),
        't_norm_min': float(t_norm_window.min()) if not t_norm_window.empty else np.nan,
        't_norm_max': float(t_norm_window.max()) if not t_norm_window.empty else np.nan,
    }
    return out


# =====================================================================
# Clasificacion v1.1
# =====================================================================

def _has_sufficient_data(df, ticker):
    try:
        close = get_col(df, ticker, 'Close')
    except (KeyError, ValueError, TypeError):
        return False
    return close.dropna().shape[0] >= WYCKOFF_TREND_SLOW_MA


def classify_wyckoff_phase(df, ticker, as_of=None):
    """Clasifica la fase segun contrato v1.2 (§5).

    Args:
        df: DataFrame OHLCV (MultiIndex o flat).
        ticker: str.
        as_of: timestamp opcional. Si se pasa, clasifica en esa fecha
            usando exclusivamente datos <= as_of. Util para tests de
            no-look-ahead. Si None, usa el ultimo valor disponible.

    Returns:
        Uno de: MARKUP | ACCUMULATION | RANGE | DISTRIBUTION | MARKDOWN
                INSUFFICIENT_DATA
    """
    if as_of is not None:
        df = df.loc[:as_of]

    if not _has_sufficient_data(df, ticker):
        return FASE_INSUFFICIENT_DATA

    try:
        combined, struct_score, tact_score, t_norm, c_norm, v_norm, e_norm = (
            wyckoff_score(df, ticker)
        )
    except (KeyError, ValueError, TypeError, IndexError):
        return FASE_INSUFFICIENT_DATA

    struct_clean = struct_score.dropna()
    t_norm_clean = t_norm.dropna()
    c_norm_clean = c_norm.dropna()

    if struct_clean.empty or t_norm_clean.empty or c_norm_clean.empty:
        return FASE_INSUFFICIENT_DATA

    last_struct = float(struct_clean.iloc[-1])
    last_t = float(t_norm_clean.iloc[-1])
    last_c = float(c_norm_clean.iloc[-1])

    # ---------------- MARKUP (contrato v1.2 §5.3) ----------------
    # Estado direccional alcista. Sin precedente. Sin veto tactico.
    if (
        last_struct > STRUCT_STRONG
        and last_t > T_NORM_STRONG
    ):
        return FASE_MARKUP

    # ---------------- MARKDOWN (contrato v1.2 §5.5) ----------------
    # Estado direccional bajista. Sin precedente. Con expansion (c_norm < 0).
    if (
        last_struct < STRUCT_WEAK
        and last_t < -T_NORM_STRONG
        and last_c < 0
    ):
        return FASE_MARKDOWN

    # Precedente estructural (solo para ACCUMULATION y DISTRIBUTION)
    prec = _compute_precedent(struct_clean, t_norm_clean, window=PRECEDENT_WINDOW)
    if prec is None:
        return FASE_INSUFFICIENT_DATA

    struct_min = prec['struct_min']
    struct_max = prec['struct_max']

    # ---------------- ACCUMULATION (contrato v1.2 §5.2) ----------------
    # Requiere precedente (formacion de base).
    trend_weak = (last_t < 0) or (abs(last_t) < T_NORM_WEAK)
    in_base_band = STRUCT_BASE_LOW <= last_struct <= STRUCT_BASE_HIGH
    compression = last_c > C_NORM_COMPRESSION
    prec_weak = struct_min < PREC_STRUCT_WEAK
    if trend_weak and in_base_band and compression and prec_weak:
        # Regla de continuidad: MARKUP -> ACCUMULATION directa prohibida.
        if struct_max >= PREC_STRUCT_STRONG:
            return FASE_RANGE
        return FASE_ACCUMULATION

    # ---------------- DISTRIBUTION (contrato v1.2 §5.4) ----------------
    # Requiere precedente (formacion de techo).
    # v1.5 (dictamen 5c O1): DISTRIBUTION = perdida estructural de fuerza
    # tras subida previa. Eliminada la condicion c_norm: durante el
    # deterioro la volatilidad expande (c_norm < 0), por lo que exigir
    # compresion era un defecto de modelado.
    #
    # Definicion final:
    #   struct_score_t < STRUCT_DETERIORO (-0.10)
    #   AND t_norm_t > -T_NORM_STRONG (-0.30)
    #   AND struct_max (historico, [t-W+1, t-1]) > PREC_STRUCT_STRONG (0.30)
    #
    # Simplificacion: t_norm > 0 OR |t| < T_NORM_WEAK equivale a
    # t_norm > -T_NORM_STRONG. Los dos casos (t>0 y -0.30<t<0) ya estan
    # contenidos en t > -0.30.
    trend_dist = last_t > -T_NORM_STRONG
    deterioration = last_struct < STRUCT_DETERIORO
    prec_strong = struct_max > PREC_STRUCT_STRONG
    if trend_dist and deterioration and prec_strong:
        return FASE_DISTRIBUTION

    # ---------------- RANGE (contrato v1.2 §5.6) ----------------
    return FASE_RANGE


# =====================================================================
# Eventos
# =====================================================================

def detect_spring(df, ticker):
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
    raw = pd.DataFrame({
        'Open': get_col(df, ticker, 'Open'),
        'High': get_col(df, ticker, 'High'),
        'Low': get_col(df, ticker, 'Low'),
        'Close': get_col(df, ticker, 'Close'),
        'Volume': get_col(df, ticker, 'Volume'),
    })
    raw = raw.dropna(subset=['Close'])
    for col in ('Open', 'High', 'Low'):
        raw[col] = raw[col].fillna(raw['Close'])
    raw['Volume'] = raw['Volume'].fillna(0.0)
    return raw
