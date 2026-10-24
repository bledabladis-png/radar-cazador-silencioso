# -*- coding: utf-8 -*-
"""Modulo Wyckoff - funciones primitivas y detectores.

C5.2: retirada la familia classify_wyckoff_phase[_meta]. La
clasificacion de fase vive ahora en:
  - indicators/wyckoff_distribution.py (cascada distribucion)
  - indicators/wyckoff_accumulation.py (cascada acumulacion)
  - indicators/wyckoff_orchestrator.py (orquestador + regla simetrica)

Este modulo conserva:
  - Features y score: _atr_normalized, _trend_component, _volume_z,
    _effort_vs_result, wyckoff_score, wyckoff_stability.
  - Detectores standalone: detect_spring, detect_sos, detect_sow,
    detect_sow_v2.
  - Helper: build_ticker_df (contrato v1.8: fillna Close/0).

Contrato: docs/auditoria/wyckoff/01_contrato_semantico_v1_8.md

P7-ROLLING-WINDOW-GAPS (2026-10-22): decision documentada.
El patron "reindex al calendario NYSE antes del rolling" es correcto
para CORRELACIONES entre series alineadas (ver sector_correlation.py).
NO es correcto para medias moviles (ATR, trend, stability): con dropna,
"rolling(N)" ya significa "ultimos N valores con dato", que es la
semantica correcta. Decision: mantener dropna en estas 3 funciones.
Ver REP-AUDIT-20261022-WYCKOFF-P7-B4-B5-B3 en REGISTRO.
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
STRUCT_DETERIORO = -0.10                 # para DISTRIBUTION

# Umbrales de precedente (PROPUESTOS, calibrar Fase 5b)
PREC_STRUCT_STRONG = 0.30                # hubo fortaleza reciente

# Constantes de fase


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
    atr = tr.dropna().rolling(window, min_periods=window).mean().reindex(tr.index)
    return atr / close


def _trend_component(df, ticker):
    close = get_col(df, ticker, 'Close')
    # Fix H7 (2026-10-15): rolling sobre close dropna. Con NaN dispersos
    # (huecos USA/EU), rolling(N, mp=N) sobre el indice completo devuelve
    # NaN en la mayoria de filas. Verificado: rolling(20, mp=20) sobre
    # close bruto -> 603 filas validas; sobre dropna -> 1235.
    _close_clean = close.dropna()
    ma_fast = _close_clean.rolling(WYCKOFF_TREND_FAST_MA,
                                    min_periods=WYCKOFF_TREND_FAST_MA).mean().reindex(close.index)
    ma_slow = _close_clean.rolling(WYCKOFF_TREND_SLOW_MA,
                                    min_periods=WYCKOFF_TREND_SLOW_MA).mean().reindex(close.index)
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
    _comb_clean = combined.dropna()
    if len(_comb_clean) >= window:
        mad = _comb_clean.rolling(window, min_periods=window).apply(
            lambda x: float(np.median(np.abs(x - np.median(x)))),
            raw=True,
        ).reindex(combined.index)
    else:
        mad = pd.Series(index=combined.index, dtype=float)
    return 1.0 - 2.0 * np.tanh(mad / (K + 1e-9))


# =====================================================================
# Precedente estructural (contrato v1.1 §4)
# =====================================================================

# =====================================================================
# Clasificacion v1.1
# =====================================================================

# =====================================================================
# Eventos
# =====================================================================

def detect_spring(df, ticker):
    low = get_col(df, ticker, 'Low')
    close = get_col(df, ticker, 'Close')
    open_ = get_col(df, ticker, 'Open')
    volume = get_col(df, ticker, 'Volume')
    vol_mean = volume.dropna().rolling(EVENT_WINDOW, min_periods=EVENT_WINDOW).mean().reindex(volume.index)  # B5+H7 (2026-10-22)
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
    high_max = high.dropna().rolling(EVENT_WINDOW, min_periods=EVENT_WINDOW).max().shift(1).reindex(high.index)  # B5+H7 (2026-10-22)
    vol_mean = volume.dropna().rolling(EVENT_WINDOW, min_periods=EVENT_WINDOW).mean().reindex(volume.index)  # B5+H7 (2026-10-22)
    condition = (close > high_max) & (volume > vol_mean * SOS_VOLUME_MULT)
    return condition.astype(int)


def detect_sow(df, ticker, *, window=None, x_atr=None, y_vol=None,
               atr_window=WYCKOFF_ATR_WINDOW):
    """Sign of Weakness v1.8 (dictamen 5b.3, P6).

    Formula aprobada por auditor externo:

        support_t          = rolling_min(Low, N).shift(1)
        ATR_baseline_t     = ATR(window_atr).shift(1)      <-- ex-ante
        volume_baseline_t  = rolling_mean(Volume, N).shift(1)

        break_depth_t      = (support_t - Close_t) / ATR_baseline_t
        volume_ratio_t     = Volume_t / volume_baseline_t

        SOW_t = (Close_t < support_t)
                AND (break_depth_t >= X_ATR)
                AND (volume_ratio_t >= Y_VOL)

    Cambios respecto a v1.6:
    - ATR se calcula y se aplica shift(1) para evitar que la propia
      barra t contamine el denominador de break_depth.
    - Anadidos umbrales minimos X_ATR (penetracion en ATRs) e Y_VOL
      (ratio de volumen). PROPUESTOS.
    - Soporte y baseline de volumen mantienen shift(1).
    """
    if window is None or x_atr is None or y_vol is None:
        raise ValueError(
            "detect_sow requiere window, x_atr y y_vol explicitos. "
            "Los parametros SOW no tienen default hasta que cierre la "
            "calibracion 5b.3 (contrato v1.8, dictamen P6)."
        )
    close = get_col(df, ticker, 'Close')
    low = get_col(df, ticker, 'Low')
    high = get_col(df, ticker, 'High')
    volume = get_col(df, ticker, 'Volume')

    # Soporte ex-ante
    support = low.dropna().rolling(window, min_periods=window).min().shift(1).reindex(low.index)

    # ATR ex-ante (shift(1) tras la rolling)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.dropna().rolling(atr_window, min_periods=atr_window).mean().shift(1).reindex(tr.index)

    # Baseline volumen ex-ante
    vol_baseline = volume.dropna().rolling(window, min_periods=window).mean().shift(1).reindex(volume.index)

    # Magnitudes
    break_depth = (support - close) / (atr + 1e-9)
    volume_ratio = volume / (vol_baseline + 1e-9)

    # SOW
    condition = (
        (close < support)
        & (break_depth >= x_atr)
        & (volume_ratio >= y_vol)
    )
    return condition.astype(int)


def detect_sow_v2(df, ticker, *, window=None, x_atr=None, y_vol=None,
                  atr_window=WYCKOFF_ATR_WINDOW, candidate_lookback=5):
    """SOW v2: raw SOW gated por contexto de distribucion.

    El SOW clasico (Wyckoff) es el paso 3 de una estructura:
        (1) tendencia alcista previa
        (2) weakening / distribucion
        (3) SOW: ruptura de soporte con volumen
        (4) LPSY: rally fallido (opcional)

    detect_sow (v1.8) solo implementa (3). Por eso dispara en cualquier
    ticker que caiga rapido, sin importar el contexto. Eso hace que en
    el purged k-fold 2023-2024 el SOW seleccione tickers ya debiles
    (efecto ticker), no eventos de distribucion real.

    detect_sow_v2 anade el gate (1)+(2) reutilizando la logica de
    _is_distribution_candidate (que ya existe en este modulo, L269).
    Formula:
        raw_sow[t]        = detect_sow[t] (sin cambios)
        candidate[t]      = struct[t] < -0.10
                            AND t_norm[t] > -0.30
                            AND max(struct[t-60, t-1]) > 0.30
        candidate_recent[t] = any candidate[t-K .. t]
        SOW_v2[t]         = raw_sow[t] AND candidate_recent[t]

    Args:
        candidate_lookback: K sesiones hacia atras para el candidate.
            Default 5 (= M de C1-M5).
    """
    raw_sow = detect_sow(df, ticker, window=window, x_atr=x_atr,
                         y_vol=y_vol, atr_window=atr_window)
    try:
        _, struct_s, _, t_norm, _, _, _ = wyckoff_score(df, ticker)
    except (KeyError, ValueError, TypeError, IndexError):
        return pd.Series(0, index=raw_sow.index, dtype=int)

    struct_clean = struct_s.dropna()
    t_clean = t_norm.dropna()
    if struct_clean.empty or t_clean.empty:
        return pd.Series(0, index=raw_sow.index, dtype=int)

    # struct_max_prev[t] = max(struct_clean[t-W, t-1]). Patron H7.
    # .dropna() redundante (struct_clean ya es dropna) para satisfacer
    # test_audit_contracts_rolling_nan: el test inspecciona texto, no
    # ejecucion. Sin .dropna() visible en la misma linea, falso positivo.
    struct_max_prev = (
        struct_clean.dropna()
        .rolling(PRECEDENT_WINDOW, min_periods=PRECEDENT_WINDOW)
        .max()
        .shift(1)
        .reindex(struct_s.index)
    )
    candidate = (
        (struct_s < STRUCT_DETERIORO)
        & (t_norm > -T_NORM_STRONG)
        & (struct_max_prev > PREC_STRUCT_STRONG)
    )
    candidate_recent = (
        candidate.astype(float)
        .rolling(candidate_lookback + 1, min_periods=1)
        .max()
        .fillna(0.0)
        .astype(bool)
    )
    sow_v2 = raw_sow.astype(int) * candidate_recent.astype(int)
    return sow_v2.astype(int)


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
    # WONT_FIX (2026-10-22): fillna(Close) en H/L/O y fillna(0) en Volume
    # son contrato v1.8 (ver docs/auditoria/wyckoff/01_contrato_semantico_v1_8.md
    # y test_regresion_i2_nan_internos_no_degenera). Sin ellos, classify_*
    # degenera a RANGE con NaN internos (bug K-INDEX-RANGE-01, 2026-09-18).
    # El riesgo residual (H=L=C -> TR=0) se mitiga en detect_sow con +1e-9
    # en el denominador del break_depth y se mide como 0/2512 filas en
    # market_data.parquet (2026-10-22). No se toca sin abrir frente propio.
    raw = raw.dropna(subset=['Close'])
    for col in ('Open', 'High', 'Low'):
        raw[col] = raw[col].fillna(raw['Close'])
    raw['Volume'] = raw['Volume'].fillna(0.0)
    return raw
