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
            fases.append(f"ERROR")
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
