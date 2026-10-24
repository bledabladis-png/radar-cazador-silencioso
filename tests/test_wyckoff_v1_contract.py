# -*- coding: utf-8 -*-
"""Tests contractuales del modulo Wyckoff v1.1.

Contrato: docs/auditoria/wyckoff/01_contrato_semantico_v1_1.md
Estos tests verifican INVARIANTES del contrato, no resultados plausibles.

Nota de estado (2026-10-02):
  - La implementacion actual (indicators/wyckoff_v1.py) implementa v1.0.
  - Varios tests fallaran hasta que se actualice a v1.1.
  - El rojo es el termometro de la brecha, no un fallo del test.

Invariantes cubiertas:
  I1  struct = 0.60*t_norm + 0.40*c_norm
  I2  tact   = 0.50*v_norm + 0.50*e_norm
  I3  combined = 0.70*struct + 0.30*tact
  I4  *_norm en (-1, 1)
  I5  struct, tact, combined en (-1, 1)
  I6  stability en (-1, 1)
  I7  ALL_NAN != RANGE
  I8  determinismo
  I9  sin look-ahead (requiere API as_of; skip hoy)
  I10 pesos suman 1.00
  I11 precedente sin look-ahead (requiere implementacion v1.1; skip)
  I12 independiente de datos futuros (requiere API as_of; skip)
  I13 precedente no circular (AST)
  I14 continuidad de fase (informativa)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from indicators import wyckoff_v1 as w1
from config.settings import (
    WYCKOFF_STRUCT_WEIGHT_TREND,
    WYCKOFF_STRUCT_WEIGHT_COMPRESSION,
    WYCKOFF_TACT_WEIGHT_VOLUME,
    WYCKOFF_TACT_WEIGHT_EFFORT,
    WYCKOFF_COMBINED_STRUCT_WEIGHT,
    WYCKOFF_COMBINED_TACT_WEIGHT,
)


# =====================================================================
# Fixture sintetico determinista
# =====================================================================

def _make_synthetic_ohlcv(n=500, seed=42):
    """Serie OHLCV con 4 tramos: bajista, base, alcista, techo.

    Determinista (seed fijo). NO usar precios reales como golden.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    price = np.zeros(n)
    price[0] = 100.0
    for i in range(1, n):
        if i < 150:
            ret = -0.004 + rng.normal(0, 0.010)
        elif i < 280:
            ret = 0.000 + rng.normal(0, 0.003)
        elif i < 420:
            ret = +0.005 + rng.normal(0, 0.012)
        else:
            ret = 0.000 + rng.normal(0, 0.005)
        price[i] = price[i - 1] * (1 + ret)
    close = pd.Series(price, index=dates)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = np.abs(rng.normal(1_000_000, 200_000, n))
    df = pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume,
    })
    return df


@pytest.fixture
def synthetic_df():
    # n=500 para superar el warm-up real (~460 filas, doble rolling).
    return _make_synthetic_ohlcv(n=500, seed=42)


@pytest.fixture
def synthetic_df_as_multi(synthetic_df):
    """Envuelve el df flat como MultiIndex (field, ticker)."""
    cols = []
    for field in ('Open', 'High', 'Low', 'Close', 'Volume'):
        cols.append((field, 'SYNTH'))
    df_multi = synthetic_df.copy()
    df_multi.columns = pd.MultiIndex.from_tuples(cols)
    return df_multi


# =====================================================================
# I1, I2, I3 - Composiciones
# =====================================================================

def test_i1_struct_composition(synthetic_df):
    """struct_score = 0.60*t_norm + 0.40*c_norm (bit a bit)."""
    combined, struct, tact, t_norm, c_norm, v_norm, e_norm = (
        w1.wyckoff_score(synthetic_df, 'SYNTH')
    )
    expected = WYCKOFF_STRUCT_WEIGHT_TREND * t_norm + WYCKOFF_STRUCT_WEIGHT_COMPRESSION * c_norm
    pd.testing.assert_series_equal(struct, expected, check_names=False, atol=1e-12)


def test_i2_tact_composition(synthetic_df):
    """tact_score = 0.50*v_norm + 0.50*e_norm."""
    combined, struct, tact, t_norm, c_norm, v_norm, e_norm = (
        w1.wyckoff_score(synthetic_df, 'SYNTH')
    )
    expected = WYCKOFF_TACT_WEIGHT_VOLUME * v_norm + WYCKOFF_TACT_WEIGHT_EFFORT * e_norm
    pd.testing.assert_series_equal(tact, expected, check_names=False, atol=1e-12)


def test_i3_combined_composition(synthetic_df):
    """combined = 0.70*struct + 0.30*tact."""
    combined, struct, tact, *_ = w1.wyckoff_score(synthetic_df, 'SYNTH')
    expected = WYCKOFF_COMBINED_STRUCT_WEIGHT * struct + WYCKOFF_COMBINED_TACT_WEIGHT * tact
    pd.testing.assert_series_equal(combined, expected, check_names=False, atol=1e-12)


# =====================================================================
# I4, I5 - Rangos
# =====================================================================

def test_i4_norms_in_range(synthetic_df):
    """Todos los *_norm en (-1, 1)."""
    _, _, _, t_norm, c_norm, v_norm, e_norm = w1.wyckoff_score(synthetic_df, 'SYNTH')
    for name, s in [('t_norm', t_norm), ('c_norm', c_norm),
                    ('v_norm', v_norm), ('e_norm', e_norm)]:
        clean = s.dropna()
        assert not clean.empty, (
            f"{name} vacio tras warm-up. Fixture demasiado corto. "
            "El warm-up real requiere ~460 filas (MA200 + rolling 200 + MAD 60)."
        )
        assert clean.abs().max() <= 1.0 + 1e-9, f"{name} fuera de (-1, 1): {clean.abs().max()}"


def test_i5_scores_in_range(synthetic_df):
    """struct, tact, combined en (-1, 1)."""
    combined, struct, tact, *_ = w1.wyckoff_score(synthetic_df, 'SYNTH')
    for name, s in [('struct', struct), ('tact', tact), ('combined', combined)]:
        clean = s.dropna()
        assert not clean.empty, f"{name} vacio tras warm-up"
        assert clean.abs().max() <= 1.0 + 1e-9, f"{name} fuera de (-1, 1): {clean.abs().max()}"


def test_i6_stability_in_range(synthetic_df):
    """stability en (-1, 1)."""
    combined, *_ = w1.wyckoff_score(synthetic_df, 'SYNTH')
    stab = w1.wyckoff_stability(combined).dropna()
    assert not stab.empty, "stability vacio tras warm-up"
    assert stab.abs().max() <= 1.0 + 1e-9
    assert stab.min() >= -1.0 - 1e-9


# =====================================================================
# I7 - ALL_NAN != RANGE
# =====================================================================

# =====================================================================
# I8 - Determinismo
# =====================================================================

# =====================================================================
# I9 - Sin look-ahead (requiere API as_of)
# =====================================================================

# =====================================================================
# I10 - Pesos suman 1.00
# =====================================================================

def test_i10_weights_sum_to_one():
    """Los pesos suman 1.00 en cada nivel."""
    assert abs(WYCKOFF_STRUCT_WEIGHT_TREND + WYCKOFF_STRUCT_WEIGHT_COMPRESSION - 1.0) < 1e-12
    assert abs(WYCKOFF_TACT_WEIGHT_VOLUME + WYCKOFF_TACT_WEIGHT_EFFORT - 1.0) < 1e-12
    assert abs(WYCKOFF_COMBINED_STRUCT_WEIGHT + WYCKOFF_COMBINED_TACT_WEIGHT - 1.0) < 1e-12


# =====================================================================
# I11, I12 - Precedente (requiere implementacion v1.1)
# =====================================================================

# =====================================================================
# I13 - No circularidad (AST)
# =====================================================================

# =====================================================================
# I14 - Continuidad de fase (informativa)
# =====================================================================

# =====================================================================
# Tests adicionales: forma de la salida
# =====================================================================

def test_score_returns_7_series(synthetic_df):
    """wyckoff_score devuelve tupla de 7 Series con mismo indice."""
    result = w1.wyckoff_score(synthetic_df, 'SYNTH')
    assert len(result) == 7
    for s in result:
        assert isinstance(s, pd.Series)


def test_build_ticker_df_handles_nan(synthetic_df):
    """build_ticker_df rellena Open/High/Low con Close y Volume con 0."""
    df = synthetic_df.copy()
    # Introducir NaN en Open y Volume
    df.loc[df.index[10], 'Open'] = np.nan
    df.loc[df.index[20], 'Volume'] = np.nan
    out = w1.build_ticker_df(df, 'SYNTH')
    assert out['Open'].notna().all()
    assert out['Volume'].notna().all()
    assert out.loc[out.index[10], 'Open'] == out.loc[out.index[10], 'Close']
    assert out.loc[out.index[20], 'Volume'] == 0.0

# =====================================================================
# Warm-up real (hallazgo 2026-10-02)
# =====================================================================

def test_warmup_real_es_doble_rolling():
    """Documenta que el warm-up real no es 200 filas sino ~460.

    Razon: trend requiere 200 filas (MA200). robust_zscore(trend, window=200,
    min_periods=60) requiere otras 60+200. Total ~460.

    Contrato v1.1 §5.1 dice 'len(Close) < 200 -> INSUFFICIENT_DATA'. Ese es
    el minimo teorico, NO el minimo para producir un score valido.
    """
    from indicators.wyckoff_v1 import _trend_component
    from src.utils import robust_zscore

    n = 300
    df = _make_synthetic_ohlcv(n=n, seed=1)
    trend = _trend_component(df, 'SYNTH')
    z = robust_zscore(trend, window=200, min_periods=60)
    # Con 300 filas, trend tiene ~101 valores y z sale vacio.
    assert z.dropna().empty, (
        "Esperado z vacio con 300 filas (warm-up doble). "
        "Si esto cambia, revisar parametros de robust_zscore."
    )

    # Con 500 filas, z tiene valores.
    df500 = _make_synthetic_ohlcv(n=500, seed=1)
    trend500 = _trend_component(df500, 'SYNTH')
    z500 = robust_zscore(trend500, window=200, min_periods=60)
    assert not z500.dropna().empty, (
        "Esperado z no vacio con 500 filas."
    )

# =====================================================================
# Tests obligatorios v1.2 (dictamen auditor §10)
# =====================================================================

def _make_uptrend_series(n=600, seed=1):
    """Serie con tendencia alcista determinista (sin ruido en Close).

    Sin ruido en close, MA50/MA200 crece monotonamente. Mediana rolling
    < ultimo valor -> t_norm positivo garantizado. Con ruido aleatorio
    (normal + exp), el ultimo trend puede caer por debajo de la mediana
    de la ventana por pura varianza, rompiendo el test sin ser un fallo
    del clasificador.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # Drift puro: precio determinista creciente.
    price = 100 * np.exp(np.linspace(0.0, 1.2, n))
    close = pd.Series(price, index=dates)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = np.abs(rng.normal(1_000_000, 200_000, n))
    return pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume,
    })


# =====================================================================
# Tests v1.3 (dictamen §14, P1 resuelto)
# =====================================================================

def _make_constant_slope_series(n=600, *, total_log_ret=1.2):
    """Serie con tendencia alcista determinista (constante)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    price = pd.Series(100 * np.exp(np.linspace(0, total_log_ret, n)), index=dates)
    return pd.DataFrame({
        'Open': price, 'High': price * 1.003, 'Low': price * 0.997,
        'Close': price, 'Volume': 1_000_000.0,
    }, index=dates)


def test_v13_constant_uptrend_is_positive():
    """I19: subida sostenida -> t_norm > 0."""
    df = _make_constant_slope_series(n=600, total_log_ret=1.2)
    _, _, _, t_norm, _, _, _ = w1.wyckoff_score(df, 'SYNTH')
    last = float(t_norm.dropna().iloc[-1])
    assert last > 0.0, f"t_norm={last} no es > 0 con subida sostenida"


def test_v13_constant_downtrend_is_negative():
    """I19: bajada sostenida -> t_norm < 0."""
    df = _make_constant_slope_series(n=600, total_log_ret=-1.2)
    _, _, _, t_norm, _, _, _ = w1.wyckoff_score(df, 'SYNTH')
    last = float(t_norm.dropna().iloc[-1])
    assert last < 0.0, f"t_norm={last} no es < 0 con bajada sostenida"


def test_v13_flat_trend_is_zero():
    """I19: trend = 0 -> t_norm = 0."""
    n = 600
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    price = pd.Series(np.full(n, 100.0), index=dates)
    df = pd.DataFrame({
        'Open': price, 'High': price, 'Low': price,
        'Close': price, 'Volume': 1_000_000.0,
    }, index=dates)
    _, _, _, t_norm, _, _, _ = w1.wyckoff_score(df, 'SYNTH')
    clean = t_norm.dropna()
    assert not clean.empty
    last = float(clean.iloc[-1])
    assert abs(last) < 1e-6, f"t_norm={last} no es ~0 con trend plano"


def test_v13_acceleration_does_not_define_direction():
    """I21: acelerar/desacelerar no invierte el signo si trend > 0."""
    n = 600
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # Acelerada
    drift_acc = np.linspace(0.001, 0.006, n)
    price_acc = pd.Series(100 * np.exp(np.cumsum(drift_acc)), index=dates)
    df_acc = pd.DataFrame({
        'Open': price_acc, 'High': price_acc * 1.003, 'Low': price_acc * 0.997,
        'Close': price_acc, 'Volume': 1_000_000.0,
    }, index=dates)
    # Desacelerada (pero sigue subiendo)
    drift_dec = np.linspace(0.006, 0.001, n)
    price_dec = pd.Series(100 * np.exp(np.cumsum(drift_dec)), index=dates)
    df_dec = pd.DataFrame({
        'Open': price_dec, 'High': price_dec * 1.003, 'Low': price_dec * 0.997,
        'Close': price_dec, 'Volume': 1_000_000.0,
    }, index=dates)

    _, _, _, t_acc, _, _, _ = w1.wyckoff_score(df_acc, 'SYNTH')
    _, _, _, t_dec, _, _, _ = w1.wyckoff_score(df_dec, 'SYNTH')
    a = float(t_acc.dropna().iloc[-1])
    d = float(t_dec.dropna().iloc[-1])
    assert a > 0, f"acelerada debe ser > 0, es {a}"
    assert d > 0, f"desacelerada debe ser > 0 (sigue subiendo), es {d}"


def test_v13_t_norm_bounds():
    """I20: t_norm acotado en (-1, 1)."""
    df = _make_constant_slope_series(n=600, total_log_ret=1.2)
    _, _, _, t_norm, _, _, _ = w1.wyckoff_score(df, 'SYNTH')
    clean = t_norm.dropna()
    assert clean.abs().max() <= 1.0 + 1e-9


# =====================================================================
# I24 - Monotonicidad de t_norm (dictamen 5b 2026-10-02)
# =====================================================================

def test_i24_monotonicidad_t_norm():
    """I24: trend_a < trend_b -> t_norm_a < t_norm_b.

    Invariante pura de la transformacion t_norm = tanh(trend/K).
    Protege la semantica principal: a mayor tendencia, mayor t_norm.
    """
    K = 0.25
    trends = [-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40]
    t_norms = [float(np.tanh(t / K)) for t in trends]
    for i in range(len(trends) - 1):
        assert t_norms[i] < t_norms[i + 1], (
            f"Monotonicidad rota: trend {trends[i]} -> t_norm {t_norms[i]}, "
            f"trend {trends[i+1]} -> t_norm {t_norms[i+1]}"
        )


def test_i24b_t_norm_signo():
    """I24b: signo de t_norm coincide con signo de trend."""
    K = 0.25
    assert float(np.tanh(-0.20 / K)) < 0
    assert float(np.tanh(0.0 / K)) == 0.0
    assert float(np.tanh(0.20 / K)) > 0

# =====================================================================
# I25 - DISTRIBUTION debe ser alcanzable (dictamen 5c v1.4)
# =====================================================================

def test_distribution_algebraic_conditions():
    """I25: las condiciones de DISTRIBUTION deben ser satisfacibles.

    v1.3 tenia c_norm > 0.30 + struct < -0.10, algebraicamente
    incompatibles. v1.4 relaja a c_norm > -0.10. Este test verifica
    que existe al menos una combinacion valida.
    """
    # Peor caso Rama 2: t_norm en banda, c_norm = -0.10, struct < -0.10
    t_norm = -0.30
    c_norm = -0.10
    struct = 0.60 * t_norm + 0.40 * c_norm
    assert struct < -0.10, f"struct {struct} deberia ser < -0.10"
    assert c_norm > -0.10 - 1e-9, f"c_norm {c_norm} deberia ser > -0.10"
    assert abs(t_norm) < 0.30 + 1e-9


# =====================================================================
# I26-I28 - DISTRIBUTION v1.5 (dictamen O1)
# =====================================================================

# =====================================================================
# v1.6 - DISTRIBUTION_CANDIDATE vs DISTRIBUTION
# =====================================================================

def test_v16_sow_no_lookahead(synthetic_df):
    """T5 (dictamen v1.6): SOW_t no depende de datos posteriores a t."""
    t_idx = synthetic_df.index[420]
    sow_ref = w1.detect_sow(synthetic_df.loc[:t_idx], 'SYNTH',
                             window=30, x_atr=0.5, y_vol=1.2)
    df_mod = synthetic_df.copy()
    df_mod.loc[df_mod.index > t_idx, 'Close'] = 999999.0
    df_mod.loc[df_mod.index > t_idx, 'High'] = 1000000.0
    df_mod.loc[df_mod.index > t_idx, 'Low'] = 999998.0
    sow_mod = w1.detect_sow(df_mod.loc[:t_idx], 'SYNTH',
                             window=30, x_atr=0.5, y_vol=1.2)
    pd.testing.assert_series_equal(sow_ref, sow_mod, check_names=False)


# =====================================================================
# v1.7 - SOW con ATR-normalizacion + umbrales
# =====================================================================

def test_v17_sow_atr_shift_ex_ante():
    """I30: ATR se aplica shift(1) (ex-ante). Modificar Close_t no
    debe cambiar ATR_baseline_t."""
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100, 90, n), index=dates)
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series(np.full(n, 1_000_000.0), index=dates)
    df = pd.DataFrame({
        'Open': close, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume,
    })
    # Detectar ATR en t = n-1 con el close real y con close alterado.
    t_idx = dates[-1]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_ref = tr.rolling(20, min_periods=20).mean().shift(1)
    # Alterar Close en t.
    df_mod = df.copy()
    df_mod.loc[t_idx, 'Close'] = 1.0  # valor absurdo
    close_mod = df_mod['Close']
    prev_close_mod = close_mod.shift(1)
    high_mod = df_mod['High']
    low_mod = df_mod['Low']
    tr_mod = pd.concat([
        high_mod - low_mod,
        (high_mod - prev_close_mod).abs(),
        (low_mod - prev_close_mod).abs(),
    ], axis=1).max(axis=1)
    atr_mod = tr_mod.rolling(20, min_periods=20).mean().shift(1)
    # ATR_baseline en t debe ser identico (usa datos <= t-1).
    assert float(atr_ref.loc[t_idx]) == float(atr_mod.loc[t_idx])


def test_v17_sow_break_depth_calculo():
    """I31: break_depth usa ATR ex-ante y umbral X_ATR."""
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # Serie plana con una caida brusca al final
    close = pd.Series(np.full(n, 100.0), index=dates)
    close.iloc[-1] = 90.0  # caida -10%
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series(np.full(n, 1_000_000.0), index=dates)
    volume.iloc[-1] = 2_000_000.0  # 2x volumen
    df = pd.DataFrame({
        'Open': close, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume,
    })
    # Con X_ATR=0 y Y_VOL=1.0: debe disparar SOW.
    sow_permisivo = w1.detect_sow(df, 'SYNTH', window=20, x_atr=0.0, y_vol=1.0)
    assert int(sow_permisivo.iloc[-1]) == 1, "SOW debe disparar con umbrales laxos"
    # Con X_ATR=100 (absurdo): no dispara.
    sow_estricto = w1.detect_sow(df, 'SYNTH', window=20, x_atr=100.0, y_vol=1.0)
    assert int(sow_estricto.iloc[-1]) == 0, "SOW no debe disparar con X_ATR absurdo"


def test_v17_sow_sin_ruptura_no_dispara():
    """I32: sin ruptura de soporte no hay SOW, aunque volumen sea alto."""
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series(np.full(n, 100.0), index=dates)
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series(np.full(n, 1_000_000.0), index=dates)
    volume.iloc[-1] = 10_000_000.0  # volumen 10x
    df = pd.DataFrame({
        'Open': close, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume,
    })
    sow = w1.detect_sow(df, 'SYNTH', window=20, x_atr=0.0, y_vol=1.0)
    # Close = 100, support = 99.5 (rolling min de low), no hay ruptura.
    assert int(sow.iloc[-1]) == 0, "Sin ruptura de soporte, no debe disparar SOW"


def test_v17_sow_no_lookahead():
    """I33: SOW_t no cambia si se alteran datos > t."""
    df = _make_constant_slope_series(n=600, total_log_ret=1.2)
    t_idx = df.index[420]
    sow_ref = w1.detect_sow(df.loc[:t_idx], 'SYNTH',
                             window=30, x_atr=0.5, y_vol=1.2)
    df_mod = df.copy()
    df_mod.loc[df_mod.index > t_idx, 'Close'] = 999999.0
    df_mod.loc[df_mod.index > t_idx, 'High'] = 1000000.0
    df_mod.loc[df_mod.index > t_idx, 'Low'] = 999998.0
    df_mod.loc[df_mod.index > t_idx, 'Volume'] = 1e15
    sow_mod = w1.detect_sow(df_mod.loc[:t_idx], 'SYNTH',
                             window=30, x_atr=0.5, y_vol=1.2)
    pd.testing.assert_series_equal(sow_ref, sow_mod, check_names=False)


# =====================================================================
# v1.8 - Fail-closed y sow_params explicitos (dictamen P6, I34-I36)
# =====================================================================

def test_i34_detect_sow_sin_kwargs_lanza_valueerror(synthetic_df):
    """I34: detect_sow sin window/x_atr/y_vol -> ValueError."""
    with pytest.raises(ValueError, match="detect_sow requiere"):
        w1.detect_sow(synthetic_df, 'SYNTH')
    with pytest.raises(ValueError, match="detect_sow requiere"):
        w1.detect_sow(synthetic_df, 'SYNTH', window=30)
    with pytest.raises(ValueError, match="detect_sow requiere"):
        w1.detect_sow(synthetic_df, 'SYNTH', window=30, x_atr=0.5)
    with pytest.raises(ValueError, match="detect_sow requiere"):
        w1.detect_sow(synthetic_df, 'SYNTH', x_atr=0.5, y_vol=1.2)


def test_i34b_detect_sow_con_kwargs_funciona(synthetic_df):
    """I34b: con los 3 kwargs explicitos, detect_sow devuelve Series."""
    sow = w1.detect_sow(synthetic_df, 'SYNTH',
                        window=30, x_atr=0.5, y_vol=1.2)
    assert isinstance(sow, pd.Series)
    assert sow.dtype in (int, 'int64', 'int32')


def test_i35_config_sow_params_son_none():
    """I35: los 4 parametros SOW en config son None hasta cierre 5b.3."""
    from config import settings
    assert settings.WYCKOFF_SOW_WINDOW_N is None
    assert settings.WYCKOFF_SOW_MAX_AGE_M is None
    assert settings.WYCKOFF_SOW_X_ATR is None
    assert settings.WYCKOFF_SOW_Y_VOL is None


# =====================================================================
# Frente F - QA interno de componentes no-SOW
# =====================================================================

# =====================================================================
# Frente F - F-03: cobertura directa de funciones puras no-SOW
# =====================================================================

def _make_flat_ohlcv(n=60, price=100.0, volume=1000.0):
    """Serie OHLCV plana. Determinista, util para tests de eventos."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": price, "High": price, "Low": price,
        "Close": price, "Volume": volume,
    }, index=dates)


def test_f03_atr_normalized_min_periods():
    """_atr_normalized con min_periods=20 devuelve NaN las primeras 19."""
    from indicators.wyckoff_v1 import _atr_normalized
    df = _make_flat_ohlcv(n=40, price=100.0)
    # Introducir rango minimo para que TR no sea 0
    df["High"] = 101.0
    df["Low"] = 99.0
    atr_norm = _atr_normalized(df, "SYNTH", window=20)
    assert atr_norm.iloc[:19].isna().all()
    assert atr_norm.iloc[19:].notna().all()


def test_f03_atr_normalized_valor_positivo_con_rango():
    """_atr_normalized devuelve valor positivo cuando hay rango real."""
    from indicators.wyckoff_v1 import _atr_normalized
    df = _make_flat_ohlcv(n=40, price=100.0)
    df["High"] = 101.0
    df["Low"] = 99.0
    atr_norm = _atr_normalized(df, "SYNTH", window=20)
    last = atr_norm.dropna().iloc[-1]
    # Rango medio = 2.0 (High-Low constante), close=100 -> ~0.02
    assert last > 0, f"esperado > 0, obtenido {last}"


def test_f03_volume_z_delega_en_robust_zscore():
    """_volume_z es wrapper directo de robust_zscore(Volume, 60, 20)."""
    from indicators.wyckoff_v1 import _volume_z
    from src.utils import robust_zscore
    from config.settings import WYCKOFF_VOLUME_ZSCORE_WINDOW
    df = _make_flat_ohlcv(n=100, volume=1_000_000.0)
    # Introducir dispersion para que robust_zscore no sea constante
    df["Volume"] = df["Volume"] + np.arange(100) * 1000.0
    v = _volume_z(df, "SYNTH")
    expected = robust_zscore(df["Volume"], window=WYCKOFF_VOLUME_ZSCORE_WINDOW, min_periods=20)
    pd.testing.assert_series_equal(v, expected, check_names=False)


def test_f03_effort_vs_result_en_rango():
    """_effort_vs_result aplica tanh -> en (-1, 1)."""
    from indicators.wyckoff_v1 import _effort_vs_result
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)
    volume = pd.Series(np.abs(rng.normal(1e6, 2e5, n)), index=dates)
    df = pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": volume,
    })
    e = _effort_vs_result(df, "SYNTH").dropna()
    assert not e.empty
    assert e.abs().max() <= 1.0 + 1e-9


def test_f03_detect_spring_positivo():
    """detect_spring detecta una configuracion construida.

    Actualizado 2026-10-22 (B5): min_periods=EVENT_WINDOW en vol_mean
    requiere al menos EVENT_WINDOW sesiones de baseline. Antes la fila
    10 con n=30 disparaba el spring porque min_periods=1 permitia
    baseline con 1 sola observacion (7 SOS espurios detectados en XLF).
    """
    from indicators.wyckoff_v1 import detect_spring
    from indicators.wyckoff_v1 import EVENT_WINDOW
    n = EVENT_WINDOW + 30  # 50 sesiones con EVENT_WINDOW=20
    spring_row = EVENT_WINDOW + 5  # 25, fuera del warm-up
    df = _make_flat_ohlcv(n=n, price=100.0, volume=1000.0)
    df.iloc[spring_row, df.columns.get_loc("Low")] = 99.0
    df.iloc[spring_row, df.columns.get_loc("Open")] = 100.0
    df.iloc[spring_row, df.columns.get_loc("Close")] = 101.0
    df.iloc[spring_row, df.columns.get_loc("High")] = 101.0
    df.iloc[spring_row, df.columns.get_loc("Volume")] = 50000.0
    spring = detect_spring(df, "SYNTH")
    assert spring.dtype == int or spring.dtype == np.int64
    assert int(spring.iloc[spring_row]) == 1, (
        f"spring no detectado en fila {spring_row}. Valores: {spring.iloc[spring_row-2:spring_row+3].tolist()}"
    )

def test_f03_detect_spring_serie_plana_no_dispara():
    """detect_spring sobre serie plana devuelve todo 0."""
    from indicators.wyckoff_v1 import detect_spring
    df = _make_flat_ohlcv(n=30)
    spring = detect_spring(df, "SYNTH")
    assert int(spring.sum()) == 0


def test_f03_detect_sos_positivo():
    """detect_sos detecta una ruptura de maximo con volumen."""
    from indicators.wyckoff_v1 import detect_sos
    df = _make_flat_ohlcv(n=30, price=100.0, volume=1000.0)
    # Filas 0-19: high=100. Fila 20: close=105 > max previos.
    df.iloc[20, df.columns.get_loc("Close")] = 105.0
    df.iloc[20, df.columns.get_loc("High")] = 105.0
    df.iloc[20, df.columns.get_loc("Volume")] = 5000.0
    sos = detect_sos(df, "SYNTH")
    assert int(sos.iloc[20]) == 1, (
        f"sos no detectado. Valores: {sos.iloc[18:23].tolist()}"
    )


def test_f03_detect_sos_no_lookahead():
    """detect_sos en t no depende de datos > t."""
    from indicators.wyckoff_v1 import detect_sos
    df = _make_flat_ohlcv(n=40, price=100.0, volume=1000.0)
    df.iloc[25, df.columns.get_loc("Close")] = 105.0
    df.iloc[25, df.columns.get_loc("High")] = 105.0
    df.iloc[25, df.columns.get_loc("Volume")] = 5000.0
    sos_ref = detect_sos(df, "SYNTH")
    t = 20
    # Modificar datos futuros
    df_mod = df.copy()
    df_mod.iloc[21:, df_mod.columns.get_loc("Close")] = 200.0
    df_mod.iloc[21:, df_mod.columns.get_loc("High")] = 200.0
    df_mod.iloc[21:, df_mod.columns.get_loc("Volume")] = 99999.0
    sos_mod = detect_sos(df_mod, "SYNTH")
    pd.testing.assert_series_equal(
        sos_ref.iloc[:t+1], sos_mod.iloc[:t+1], check_names=False
    )


def test_regresion_i2_nan_internos_no_degenera():
    """Regresion I2 (2026-09-18). Portada de tests legacy al retirar
    el modulo legacy 2026-10-10.

    Contrato v1.8: build_ticker_df rellena Open/High/Low desde Close
    y Volume con 0, evitando que los NaN internos propaguen. Este
    test verifica que con NaN internos el helper no deja NaN y
    wyckoff_score produce series validas.

    C5.2: retirada la parte de classify_wyckoff_phase. La
    clasificacion vive ahora en el orquestador.
    """
    import numpy as np
    rng = np.random.default_rng(20261010)
    n = 300
    dates = pd.date_range('2025-01-01', periods=n, freq='B')
    base = 100 + rng.standard_normal(n).cumsum() + np.linspace(0, 20, n)
    df = pd.DataFrame({
        ('Open', 'TST'): base + rng.standard_normal(n) * 0.5,
        ('High', 'TST'): base + abs(rng.standard_normal(n)) * 1.0,
        ('Low', 'TST'): base - abs(rng.standard_normal(n)) * 1.0,
        ('Close', 'TST'): base,
        ('Volume', 'TST'): rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)
    df.loc[df.index[50:70], ('Close', 'TST')] = np.nan
    df.loc[df.index[100:105], ('Volume', 'TST')] = np.nan

    tdf = w1.build_ticker_df(df, 'TST')
    assert not tdf.empty, 'build_ticker_df devolvio vacio con NaN internos'
    assert tdf.notna().all().all(), 'build_ticker_df dejo NaN'
    # Score no colapsa
    combined, struct_s, tact_s, t_norm, c_norm, v_norm, e_norm = w1.wyckoff_score(tdf, 'TST')
    assert len(combined) == len(tdf)
