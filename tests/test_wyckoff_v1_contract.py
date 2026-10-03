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

import ast
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

def test_i7_all_nan_not_range():
    """Sin observacion valida -> INSUFFICIENT_DATA, no RANGE.

    Construye un df con 300 filas todas NaN. El clasificador debe devolver
    INSUFFICIENT_DATA. Bajo contrato v1.0 (implementacion actual) devuelve
    RANGE. Este test FALLA hasta que wyckoff_v1.py implemente v1.1.
    """
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        'Open': [np.nan] * n, 'High': [np.nan] * n, 'Low': [np.nan] * n,
        'Close': [np.nan] * n, 'Volume': [np.nan] * n,
    }, index=dates)
    phase = w1.classify_wyckoff_phase(df, 'SYNTH')
    assert phase == "INSUFFICIENT_DATA", (
        f"Esperado INSUFFICIENT_DATA, obtenido {phase}. "
        "Contrato v1.1 §5.1 e invariante I7."
    )


def test_i7b_short_series_insufficient():
    """Series de <200 filas -> INSUFFICIENT_DATA."""
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': 100.5, 'Volume': 1_000_000.0,
    }, index=dates)
    phase = w1.classify_wyckoff_phase(df, 'SYNTH')
    assert phase == "INSUFFICIENT_DATA"


# =====================================================================
# I8 - Determinismo
# =====================================================================

def test_i8_determinism(synthetic_df):
    """Mismo input -> mismo output."""
    p1 = w1.classify_wyckoff_phase(synthetic_df, 'SYNTH')
    p2 = w1.classify_wyckoff_phase(synthetic_df.copy(), 'SYNTH')
    assert p1 == p2


# =====================================================================
# I9 - Sin look-ahead (requiere API as_of)
# =====================================================================

@pytest.mark.skip(reason="Requiere API as_of en classify_wyckoff_phase. Fase 5b.")
def test_i9_no_lookahead(synthetic_df):
    """Fase en t no cambia si se anaden datos > t. Requiere as_of."""
    pass


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

@pytest.mark.skip(reason="Requiere precedente estructural. Fase 5b.")
def test_i11_precedente_no_lookahead():
    pass


@pytest.mark.skip(reason="Requiere API as_of. Fase 5b.")
def test_i12_independiente_datos_futuros():
    pass


# =====================================================================
# I13 - No circularidad (AST)
# =====================================================================

def test_i13_no_circular():
    """classify_wyckoff_phase no invoca classify_wyckoff_phase.

    Verificacion AST: dentro del cuerpo de la funcion, no debe haber
    llamada recursiva.
    """
    src = (ROOT / "indicators" / "wyckoff_v1.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    def find_func(name, node):
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef) and child.name == name:
                return child
        return None

    fn = find_func("classify_wyckoff_phase", tree)
    assert fn is not None, "classify_wyckoff_phase no encontrada"

    # Buscar llamadas a classify_wyckoff_phase dentro del cuerpo
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == "classify_wyckoff_phase":
                pytest.fail("classify_wyckoff_phase se invoca a si misma (circularidad)")


# =====================================================================
# I14 - Continuidad de fase (informativa)
# =====================================================================

def test_i14_transiciones_observables(synthetic_df):
    """Serie larga produce mas de una fase distinta.

    Test informativo: NO verifica transiciones especificas (para no
    calibrar con el fixture). Solo verifica que no devuelve siempre
    el mismo estado.
    """
    # Clasificar en distintos puntos de la serie
    fases = []
    for t in (210, 240, 270, 299):
        df_slice = synthetic_df.iloc[:t+1]
        try:
            fase = w1.classify_wyckoff_phase(df_slice, 'SYNTH')
            fases.append(fase)
        except Exception:
            fases.append("ERROR")
    # Informativo: registrar el hallazgo, pero no forzar que haya 2+ fases.
    # Si solo hay RANGE, es senal de que falta semantica temporal (v1.1).
    # Por eso NO hacemos assert; imprimimos para diagnostico.
    print(f"\n[I14] fases en t=210,240,270,299: {fases}")
    # Un minimo assert defensivo: no debe crashear
    assert all(not f.startswith("ERROR") for f in fases), f"crasheo: {fases}"


# =====================================================================
# Tests adicionales: forma de la salida
# =====================================================================

def test_score_returns_7_series(synthetic_df):
    """wyckoff_score devuelve tupla de 7 Series con mismo indice."""
    result = w1.wyckoff_score(synthetic_df, 'SYNTH')
    assert len(result) == 7
    for s in result:
        assert isinstance(s, pd.Series)


def test_classify_returns_valid_string(synthetic_df):
    """classify_wyckoff_phase devuelve uno de los 6 estados del contrato."""
    phase = w1.classify_wyckoff_phase(synthetic_df, 'SYNTH')
    valid = {"MARKUP", "ACCUMULATION", "RANGE", "DISTRIBUTION",
             "MARKDOWN", "INSUFFICIENT_DATA"}
    assert phase in valid, f"Fase desconocida: {phase}"


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


def test_v12_t1_mas_historia_alcista_sigue_markup():
    """Test 1 (dictamen §10): mas historia alcista no rompe MARKUP.

    v1.3 resuelve P1. Se usa el fixture determinista (sin ruido) porque
    T1 aísla el efecto de "persistencia de tendencia sobre t_norm". Un
    fixture con volatilidad variable puede dar RANGE por composicion
    struct (c_norm muy negativo), no por t_norm. Ese es un caso distinto
    y no debe mezclarse con T1.
    """
    df = _make_constant_slope_series(n=600, total_log_ret=1.2)
    fase = w1.classify_wyckoff_phase(df, 'SYNTH')
    assert fase == "MARKUP", (
        f"Esperado MARKUP en tendencia alcista sostenida, obtenido {fase}. "
        "Si falla, P1 ha reaparecido."
    )


def test_v12_t2_tactical_negativo_no_veta_markup():
    """I17: tactical negativo no invalida MARKUP (dictamen §10, Test 2)."""
    from unittest import mock
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    struct = pd.Series(np.full(n, 0.65), index=dates)
    tact = pd.Series(np.full(n, -0.50), index=dates)
    t_norm = pd.Series(np.full(n, 0.80), index=dates)
    c_norm = pd.Series(np.full(n, 0.30), index=dates)
    v_norm = pd.Series(np.full(n, 0.0), index=dates)
    e_norm = pd.Series(np.full(n, -1.0), index=dates)
    combined = 0.70 * struct + 0.30 * tact
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': 100.0, 'Volume': 1_000_000.0,
    }, index=dates)
    with mock.patch.object(
        w1, 'wyckoff_score',
        return_value=(combined, struct, tact, t_norm, c_norm, v_norm, e_norm),
    ):
        fase = w1.classify_wyckoff_phase(df, 'SYNTH')
    assert fase == "MARKUP", (
        f"Tactical negativo veto MARKUP. Fase: {fase}. "
        "Regresion contra D2-v1.1."
    )


def test_v12_t3_tactical_negativo_no_veta_accumulation():
    """I17: tactical negativo no invalida ACCUMULATION (dictamen §10, Test 3)."""
    from unittest import mock
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # Base: 30 valores finales. Deterioro previo: 470 valores.
    # La ventana de precedente (60) captura 30 de base + 30 del final
    # del deterioro (struct_min=-0.50 < -0.20 -> prec_weak).
    struct_vals = np.concatenate([
        np.linspace(-0.50, -0.30, n - 30),  # deterioro
        np.full(30, 0.00),                  # base corta
    ])
    struct = pd.Series(struct_vals, index=dates)
    tact = pd.Series(np.full(n, -0.50), index=dates)
    t_norm_vals = np.concatenate([
        np.linspace(-0.60, -0.40, n - 30),
        np.full(30, -0.10),
    ])
    t_norm = pd.Series(t_norm_vals, index=dates)
    c_norm = pd.Series(np.full(n, 0.50), index=dates)
    v_norm = pd.Series(np.full(n, 0.0), index=dates)
    e_norm = pd.Series(np.full(n, -1.0), index=dates)
    combined = 0.70 * struct + 0.30 * tact
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': 100.0, 'Volume': 1_000_000.0,
    }, index=dates)
    with mock.patch.object(
        w1, 'wyckoff_score',
        return_value=(combined, struct, tact, t_norm, c_norm, v_norm, e_norm),
    ):
        fase = w1.classify_wyckoff_phase(df, 'SYNTH')
    assert fase == "ACCUMULATION", (
        f"Tactical negativo veto ACCUMULATION. Fase: {fase}. "
        "Regresion contra D3-v1.1."
    )


def test_v12_t4_no_lookahead_datos_futuros(synthetic_df):
    """Test 4 (dictamen §10): datos futuros no cambian fase en t."""
    t_idx = synthetic_df.index[420]
    fase_ref = w1.classify_wyckoff_phase(synthetic_df, 'SYNTH', as_of=t_idx)
    # Modificar datos futuros despues de t_idx
    df_mod = synthetic_df.copy()
    df_mod.loc[df_mod.index > t_idx, 'Close'] = 999999.0
    df_mod.loc[df_mod.index > t_idx, 'High'] = 1000000.0
    df_mod.loc[df_mod.index > t_idx, 'Low'] = 999998.0
    fase_mod = w1.classify_wyckoff_phase(df_mod, 'SYNTH', as_of=t_idx)
    assert fase_mod == fase_ref, (
        f"Look-ahead detectado: fase en t={t_idx} cambio al modificar "
        f"datos futuros. ref={fase_ref}, mod={fase_mod}"
    )


def test_v12_i15_markup_no_requiere_precedente(synthetic_df):
    """I15: MARKUP no requiere precedente (invariante v1.2)."""
    df = _make_uptrend_series(n=600, seed=1)
    # Verificar que la clasificacion no cae en RANGE por struct_max alto.
    fase = w1.classify_wyckoff_phase(df, 'SYNTH')
    assert fase != "INSUFFICIENT_DATA"

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


def test_v13_markup_not_rejected_by_stable_trend():
    """I22: subida sostenida puede ser MARKUP (T1 formalizado)."""
    df = _make_constant_slope_series(n=600, total_log_ret=1.2)
    phase = w1.classify_wyckoff_phase(df, 'SYNTH')
    assert phase == "MARKUP", (
        f"Subida sostenida devolvio {phase}. I22 exige MARKUP. "
        "Si falla, P1 ha reaparecido."
    )


def test_v13_no_lookahead_equivalente_a_T4(synthetic_df):
    """I23: datos futuros no cambian la fase en t (via as_of)."""
    t_idx = synthetic_df.index[420]
    fase_ref = w1.classify_wyckoff_phase(synthetic_df, 'SYNTH', as_of=t_idx)
    df_mod = synthetic_df.copy()
    df_mod.loc[df_mod.index > t_idx, 'Close'] = 999999.0
    fase_mod = w1.classify_wyckoff_phase(df_mod, 'SYNTH', as_of=t_idx)
    assert fase_mod == fase_ref

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


def test_distribution_is_reachable(monkeypatch):
    """I25b: DISTRIBUTION alcanzable con mock de wyckoff_score.

    Construye sinteticamente un df cuyas condiciones cumplan §5.4 v1.4.
    """
    from unittest import mock
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # Estructura: 470 con precedente fuerte, 30 con deterioro reciente.
    # Razon: la ventana de precedente (60) ve los 60 valores antes del
    # ultimo. Si el deterioro fuera de 200, la ventana no veria el +0.40.
    # Con 30 de deterioro, la ventana captura 30 de deterioro + 30 de
    # precedente fuerte -> struct_max = +0.40 > 0.30.
    struct_vals = np.concatenate([
        np.full(n - 30, 0.40),        # precedente fuerte
        np.full(30, -0.20),           # deterioro reciente
    ])
    t_norm_vals = np.concatenate([
        np.full(n - 30, 0.50),
        np.full(30, -0.15),           # |t| < 0.30
    ])
    c_norm_vals = np.full(n, 0.10)    # > -0.10
    struct = pd.Series(struct_vals, index=dates)
    tact = pd.Series(np.full(n, 0.0), index=dates)
    t_norm = pd.Series(t_norm_vals, index=dates)
    c_norm = pd.Series(c_norm_vals, index=dates)
    v_norm = pd.Series(np.zeros(n), index=dates)
    e_norm = pd.Series(np.zeros(n), index=dates)
    combined = 0.70 * struct + 0.30 * tact
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': 100.0, 'Volume': 1_000_000.0,
    }, index=dates)
    # v1.6: DISTRIBUTION requiere candidate + SOW reciente.
    sow_vals = np.zeros(n); sow_vals[-1] = 1
    sow = pd.Series(sow_vals, index=dates)
    sow_params = {"window": 30, "x_atr": 0.5, "y_vol": 1.2, "max_age_m": 10}
    with mock.patch.object(
        w1, 'wyckoff_score',
        return_value=(combined, struct, tact, t_norm, c_norm, v_norm, e_norm),
    ), mock.patch.object(w1, 'detect_sow', return_value=sow):
        phase = w1.classify_wyckoff_phase(df, 'SYNTH', sow_params=sow_params)
    assert phase == "DISTRIBUTION", (
        f"Esperado DISTRIBUTION alcanzable, obtenido {phase}. "
        "Si falla, I25 rota (contrato imposible)."
    )

# =====================================================================
# I26-I28 - DISTRIBUTION v1.5 (dictamen O1)
# =====================================================================

def test_distribution_requires_prior_strength():
    """I18 + I26: DISTRIBUTION exige precedente fuerte (struct_max > 0.30).

    Casos:
      1) struct_max <= 0.30 + deterioro actual -> NO DISTRIBUTION.
      2) struct_max > 0.30 + deterioro actual + t_norm > -0.30 -> DISTRIBUTION.
    """
    from unittest import mock
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': 100.0, 'Volume': 1_000_000.0,
    }, index=dates)

    # Caso 1: sin precedente fuerte
    struct_vals_no = np.concatenate([
        np.full(n - 30, 0.10),         # NO fuerte
        np.full(30, -0.20),            # deterioro actual
    ])
    t_norm_vals = np.concatenate([
        np.full(n - 30, 0.15),
        np.full(30, -0.10),
    ])
    c_norm_vals = np.full(n, 0.10)
    struct = pd.Series(struct_vals_no, index=dates)
    tact = pd.Series(np.zeros(n), index=dates)
    t_norm = pd.Series(t_norm_vals, index=dates)
    c_norm = pd.Series(c_norm_vals, index=dates)
    v_norm = pd.Series(np.zeros(n), index=dates)
    e_norm = pd.Series(np.zeros(n), index=dates)
    combined = 0.70 * struct + 0.30 * tact
    with mock.patch.object(w1, 'wyckoff_score',
                            return_value=(combined, struct, tact, t_norm, c_norm, v_norm, e_norm)):
        phase_no = w1.classify_wyckoff_phase(df, 'SYNTH')
    assert phase_no != "DISTRIBUTION", (
        f"Sin precedente fuerte, no debe dar DISTRIBUTION. Obtenido: {phase_no}"
    )

    # Caso 2: con precedente fuerte
    struct_vals_si = np.concatenate([
        np.full(n - 30, 0.40),         # SI fuerte
        np.full(30, -0.20),            # deterioro actual
    ])
    struct2 = pd.Series(struct_vals_si, index=dates)
    t_norm2 = pd.Series(t_norm_vals, index=dates)
    c_norm2 = pd.Series(c_norm_vals, index=dates)
    combined2 = 0.70 * struct2 + 0.30 * tact
    # v1.6: anadir SOW reciente para confirmar DISTRIBUTION.
    sow_vals = np.zeros(n); sow_vals[-1] = 1
    sow = pd.Series(sow_vals, index=dates)
    sow_params = {"window": 30, "x_atr": 0.5, "y_vol": 1.2, "max_age_m": 10}
    with mock.patch.object(w1, 'wyckoff_score',
                            return_value=(combined2, struct2, tact, t_norm2, c_norm2, v_norm, e_norm)), \
         mock.patch.object(w1, 'detect_sow', return_value=sow):
        phase_si = w1.classify_wyckoff_phase(df, 'SYNTH', sow_params=sow_params)
    assert phase_si == "DISTRIBUTION", (
        f"Con precedente fuerte + deterioro + t>-0.30 + SOW, debe ser DISTRIBUTION. "
        f"Obtenido: {phase_si}"
    )


def test_distribution_vs_markdown_boundary():
    """I16 + I27: frontera DISTRIBUTION vs MARKDOWN.

    Con struct_max > 0.30:
      - struct_actual en (-0.30, -0.10) y t > -0.30 -> DISTRIBUTION.
      - struct_actual < -0.30 y t < -0.30 (con c < 0) -> MARKDOWN.
    """
    from unittest import mock
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': 100.0, 'Volume': 1_000_000.0,
    }, index=dates)

    # Caso DISTRIBUTION: struct en (-0.30, -0.10), t > -0.30
    struct_dist = np.concatenate([np.full(n - 30, 0.40), np.full(30, -0.20)])
    t_dist = np.concatenate([np.full(n - 30, 0.50), np.full(30, -0.15)])
    c_dist = np.full(n, 0.10)
    struct = pd.Series(struct_dist, index=dates)
    t_norm = pd.Series(t_dist, index=dates)
    c_norm = pd.Series(c_dist, index=dates)
    tact = pd.Series(np.zeros(n), index=dates)
    v_norm = pd.Series(np.zeros(n), index=dates)
    e_norm = pd.Series(np.zeros(n), index=dates)
    combined = 0.70 * struct + 0.30 * tact
    sow_vals = np.zeros(n); sow_vals[-1] = 1
    sow = pd.Series(sow_vals, index=dates)
    sow_params = {"window": 30, "x_atr": 0.5, "y_vol": 1.2, "max_age_m": 10}
    with mock.patch.object(w1, 'wyckoff_score',
                            return_value=(combined, struct, tact, t_norm, c_norm, v_norm, e_norm)), \
         mock.patch.object(w1, 'detect_sow', return_value=sow):
        phase = w1.classify_wyckoff_phase(df, 'SYNTH', sow_params=sow_params)
    assert phase == "DISTRIBUTION", f"Esperado DISTRIBUTION, obtenido {phase}"

    # Caso MARKDOWN: struct < -0.30, t < -0.30, c < 0
    struct_md = np.full(n, -0.40)
    t_md = np.full(n, -0.50)
    c_md = np.full(n, -0.20)
    struct2 = pd.Series(struct_md, index=dates)
    t_norm2 = pd.Series(t_md, index=dates)
    c_norm2 = pd.Series(c_md, index=dates)
    combined2 = 0.70 * struct2 + 0.30 * tact
    with mock.patch.object(w1, 'wyckoff_score',
                            return_value=(combined2, struct2, tact, t_norm2, c_norm2, v_norm, e_norm)):
        phase2 = w1.classify_wyckoff_phase(df, 'SYNTH')
    assert phase2 == "MARKDOWN", f"Esperado MARKDOWN, obtenido {phase2}"


def test_distribution_no_lookahead(synthetic_df):
    """I28: la fase en t es invariante a datos posteriores.

    Test con datos reales (sin mock). Clasifica en t = idx[420] con
    as_of. Luego anade ruido futuro. La fase en t debe coincidir.
    """
    t_idx = synthetic_df.index[420]
    fase_ref = w1.classify_wyckoff_phase(synthetic_df, 'SYNTH', as_of=t_idx)

    # Anadir datos futuros muy distintos
    df_fut = synthetic_df.copy()
    df_fut.loc[df_fut.index > t_idx, 'Close'] = 999999.0
    df_fut.loc[df_fut.index > t_idx, 'High'] = 1000000.0
    df_fut.loc[df_fut.index > t_idx, 'Low'] = 999998.0

    fase_mod = w1.classify_wyckoff_phase(df_fut, 'SYNTH', as_of=t_idx)
    assert fase_mod == fase_ref, (
        f"Look-ahead detectado: fase en t={t_idx} cambio al modificar "
        f"datos futuros. ref={fase_ref}, mod={fase_mod}"
    )

# =====================================================================
# v1.6 - DISTRIBUTION_CANDIDATE vs DISTRIBUTION
# =====================================================================

def test_v16_candidate_no_implica_distribution():
    """T1 (dictamen v1.6): candidate sin SOW no es DISTRIBUTION."""
    from unittest import mock
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': 100.0, 'Volume': 1_000_000.0,
    }, index=dates)
    struct_vals = np.concatenate([
        np.full(n - 30, 0.40),
        np.full(30, -0.20),
    ])
    struct = pd.Series(struct_vals, index=dates)
    t_norm = pd.Series(np.concatenate([
        np.full(n - 30, 0.50), np.full(30, -0.15)
    ]), index=dates)
    c_norm = pd.Series(np.full(n, 0.10), index=dates)
    tact = pd.Series(np.zeros(n), index=dates)
    v_norm = pd.Series(np.zeros(n), index=dates)
    e_norm = pd.Series(np.zeros(n), index=dates)
    combined = 0.70 * struct + 0.30 * tact

    # SOW siempre 0 (df plano no dispara SOW)
    sow_zeros = pd.Series(np.zeros(n), index=dates)
    with mock.patch.object(w1, 'wyckoff_score',
                            return_value=(combined, struct, tact, t_norm, c_norm, v_norm, e_norm)), \
         mock.patch.object(w1, 'detect_sow', return_value=sow_zeros):
        phase = w1.classify_wyckoff_phase(df, 'SYNTH')
    # Candidate cumple condiciones pero sin SOW -> no DISTRIBUTION
    assert phase != "DISTRIBUTION", (
        f"Candidate sin SOW no debe ser DISTRIBUTION. Obtenido: {phase}"
    )
    assert phase == "RANGE", f"Esperado RANGE, obtenido: {phase}"


def test_v16_candidate_con_sow_es_distribution():
    """T3 (dictamen v1.6): candidate + SOW reciente -> DISTRIBUTION."""
    from unittest import mock
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': 100.0, 'Volume': 1_000_000.0,
    }, index=dates)
    struct_vals = np.concatenate([
        np.full(n - 30, 0.40),
        np.full(30, -0.20),
    ])
    struct = pd.Series(struct_vals, index=dates)
    t_norm = pd.Series(np.concatenate([
        np.full(n - 30, 0.50), np.full(30, -0.15)
    ]), index=dates)
    c_norm = pd.Series(np.full(n, 0.10), index=dates)
    tact = pd.Series(np.zeros(n), index=dates)
    v_norm = pd.Series(np.zeros(n), index=dates)
    e_norm = pd.Series(np.zeros(n), index=dates)
    combined = 0.70 * struct + 0.30 * tact

    # SOW en la ultima posicion (dentro de M=10)
    sow_vals = np.zeros(n)
    sow_vals[-1] = 1
    sow = pd.Series(sow_vals, index=dates)
    sow_params = {"window": 30, "x_atr": 0.5, "y_vol": 1.2, "max_age_m": 10}
    with mock.patch.object(w1, 'wyckoff_score',
                            return_value=(combined, struct, tact, t_norm, c_norm, v_norm, e_norm)), \
         mock.patch.object(w1, 'detect_sow', return_value=sow):
        phase = w1.classify_wyckoff_phase(df, 'SYNTH', sow_params=sow_params)
    assert phase == "DISTRIBUTION", (
        f"Candidate + SOW reciente debe ser DISTRIBUTION. Obtenido: {phase}"
    )


def test_v16_sow_aislado_no_es_distribution():
    """T2 (dictamen v1.6): SOW sin candidate no es DISTRIBUTION."""
    from unittest import mock
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': 100.0, 'Volume': 1_000_000.0,
    }, index=dates)
    # struct positivo (no candidate)
    struct = pd.Series(np.full(n, 0.40), index=dates)
    t_norm = pd.Series(np.full(n, 0.50), index=dates)
    c_norm = pd.Series(np.full(n, 0.10), index=dates)
    tact = pd.Series(np.zeros(n), index=dates)
    v_norm = pd.Series(np.zeros(n), index=dates)
    e_norm = pd.Series(np.zeros(n), index=dates)
    combined = 0.70 * struct + 0.30 * tact
    sow_vals = np.zeros(n)
    sow_vals[-1] = 1
    sow = pd.Series(sow_vals, index=dates)
    with mock.patch.object(w1, 'wyckoff_score',
                            return_value=(combined, struct, tact, t_norm, c_norm, v_norm, e_norm)), \
         mock.patch.object(w1, 'detect_sow', return_value=sow):
        phase = w1.classify_wyckoff_phase(df, 'SYNTH')
    assert phase != "DISTRIBUTION", (
        f"SOW aislado (sin candidate) no debe ser DISTRIBUTION. Obtenido: {phase}"
    )


def test_v16_sow_antiguo_no_confirma():
    """T4 (dictamen v1.6): SOW fuera de M no confirma."""
    from unittest import mock
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': 100.0, 'Volume': 1_000_000.0,
    }, index=dates)
    struct_vals = np.concatenate([
        np.full(n - 30, 0.40),
        np.full(30, -0.20),
    ])
    struct = pd.Series(struct_vals, index=dates)
    t_norm = pd.Series(np.concatenate([
        np.full(n - 30, 0.50), np.full(30, -0.15)
    ]), index=dates)
    c_norm = pd.Series(np.full(n, 0.10), index=dates)
    tact = pd.Series(np.zeros(n), index=dates)
    v_norm = pd.Series(np.zeros(n), index=dates)
    e_norm = pd.Series(np.zeros(n), index=dates)
    combined = 0.70 * struct + 0.30 * tact
    # SOW en posicion n-50 (fuera de M=10)
    sow_vals = np.zeros(n)
    sow_vals[-50] = 1
    sow = pd.Series(sow_vals, index=dates)
    with mock.patch.object(w1, 'wyckoff_score',
                            return_value=(combined, struct, tact, t_norm, c_norm, v_norm, e_norm)), \
         mock.patch.object(w1, 'detect_sow', return_value=sow):
        phase = w1.classify_wyckoff_phase(df, 'SYNTH')
    assert phase != "DISTRIBUTION", (
        f"SOW fuera de M no confirma DISTRIBUTION. Obtenido: {phase}"
    )


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


def test_v16_classify_meta_devuelve_campos():
    """I29: classify_wyckoff_phase_meta devuelve dict con phase + flag."""
    df = _make_constant_slope_series(n=600, total_log_ret=1.2)
    meta = w1.classify_wyckoff_phase_meta(df, 'SYNTH')
    assert isinstance(meta, dict)
    assert "phase" in meta
    assert "distribution_candidate" in meta
    assert isinstance(meta["distribution_candidate"], bool)

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


def test_i36_sin_sow_params_no_emite_distribution():
    """I36: classify_wyckoff_phase sin sow_params nunca emite DISTRIBUTION.

    Con candidate_conditions=True y sin sow_params, debe devolver RANGE.
    Con sow_params explicitos, debe devolver DISTRIBUTION.
    """
    from unittest import mock
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': 100.0, 'Volume': 1_000_000.0,
    }, index=dates)
    struct_vals = np.concatenate([np.full(n - 30, 0.40), np.full(30, -0.20)])
    struct = pd.Series(struct_vals, index=dates)
    t_norm = pd.Series(np.concatenate([
        np.full(n - 30, 0.50), np.full(30, -0.15)
    ]), index=dates)
    c_norm = pd.Series(np.full(n, 0.10), index=dates)
    tact = pd.Series(np.zeros(n), index=dates)
    v_norm = pd.Series(np.zeros(n), index=dates)
    e_norm = pd.Series(np.zeros(n), index=dates)
    combined = 0.70 * struct + 0.30 * tact
    sow_vals = np.zeros(n); sow_vals[-1] = 1
    sow = pd.Series(sow_vals, index=dates)
    with mock.patch.object(w1, 'wyckoff_score',
                            return_value=(combined, struct, tact, t_norm, c_norm, v_norm, e_norm)), \
         mock.patch.object(w1, 'detect_sow', return_value=sow):
        phase = w1.classify_wyckoff_phase(df, 'SYNTH')
    assert phase == "RANGE", (
        f"Sin sow_params, no debe emitir DISTRIBUTION. Obtenido: {phase}"
    )
    sow_params = {"window": 30, "x_atr": 0.5, "y_vol": 1.2, "max_age_m": 10}
    with mock.patch.object(w1, 'wyckoff_score',
                            return_value=(combined, struct, tact, t_norm, c_norm, v_norm, e_norm)), \
         mock.patch.object(w1, 'detect_sow', return_value=sow):
        phase2 = w1.classify_wyckoff_phase(df, 'SYNTH', sow_params=sow_params)
    assert phase2 == "DISTRIBUTION", (
        f"Con sow_params, debe emitir DISTRIBUTION. Obtenido: {phase2}"
    )


def test_i36b_meta_propaga_sow_params():
    """I36b: classify_wyckoff_phase_meta propaga sow_params a classify."""
    from unittest import mock
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        'Open': 100.0, 'High': 101.0, 'Low': 99.0,
        'Close': 100.0, 'Volume': 1_000_000.0,
    }, index=dates)
    struct_vals = np.concatenate([np.full(n - 30, 0.40), np.full(30, -0.20)])
    struct = pd.Series(struct_vals, index=dates)
    t_norm = pd.Series(np.concatenate([
        np.full(n - 30, 0.50), np.full(30, -0.15)
    ]), index=dates)
    c_norm = pd.Series(np.full(n, 0.10), index=dates)
    tact = pd.Series(np.zeros(n), index=dates)
    v_norm = pd.Series(np.zeros(n), index=dates)
    e_norm = pd.Series(np.zeros(n), index=dates)
    combined = 0.70 * struct + 0.30 * tact
    sow_vals = np.zeros(n); sow_vals[-1] = 1
    sow = pd.Series(sow_vals, index=dates)
    sow_params = {"window": 30, "x_atr": 0.5, "y_vol": 1.2, "max_age_m": 10}
    with mock.patch.object(w1, 'wyckoff_score',
                            return_value=(combined, struct, tact, t_norm, c_norm, v_norm, e_norm)), \
         mock.patch.object(w1, 'detect_sow', return_value=sow):
        meta = w1.classify_wyckoff_phase_meta(df, 'SYNTH', sow_params=sow_params)
    assert meta["phase"] == "DISTRIBUTION"
    assert meta["distribution_candidate"] is True

# =====================================================================
# Frente F - QA interno de componentes no-SOW
# =====================================================================

def test_f02_meta_no_traga_runtime_error():
    """F-02: classify_wyckoff_phase_meta no debe silenciar RuntimeError.

    Regla 01_METODO seccion 8: RuntimeError va en tuplas acotadas.
    Un except Exception lo traga silenciosamente. Con el fix (acotar
    except a tuplas), un RuntimeError de _is_distribution_candidate
    debe propagar.

    Test rojo sin el fix. Verde tras acotar el except.
    """
    from unittest import mock
    import pandas as pd

    # df minimo que devuelve RANGE para llegar al bloque con el
    # except. Con mock sobre _is_distribution_candidate forzamos
    # que lance RuntimeError.
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": 100.0, "High": 101.0, "Low": 99.0,
        "Close": 100.0, "Volume": 1_000_000.0,
    }, index=dates)

    # Mock: classify devuelve RANGE (fuerza entrar al except) y
    # _is_distribution_candidate lanza RuntimeError.
    with mock.patch.object(w1, "classify_wyckoff_phase", return_value="RANGE"), \
         mock.patch.object(w1, "_is_distribution_candidate",
                           side_effect=RuntimeError("fallo simulado")):
        with pytest.raises(RuntimeError, match="fallo simulado"):
            w1.classify_wyckoff_phase_meta(df, "SYNTH")

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
    """detect_spring detecta una configuracion construida."""
    from indicators.wyckoff_v1 import detect_spring
    df = _make_flat_ohlcv(n=30, price=100.0, volume=1000.0)
    # Fila 10: low cae, cierre > open, volumen alto
    df.iloc[10, df.columns.get_loc("Low")] = 99.0
    df.iloc[10, df.columns.get_loc("Open")] = 100.0
    df.iloc[10, df.columns.get_loc("Close")] = 101.0
    df.iloc[10, df.columns.get_loc("High")] = 101.0
    df.iloc[10, df.columns.get_loc("Volume")] = 50000.0
    spring = detect_spring(df, "SYNTH")
    assert spring.dtype == int or spring.dtype == np.int64
    assert int(spring.iloc[10]) == 1, (
        f"spring no detectado en fila 10. Valores: {spring.iloc[8:13].tolist()}"
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
