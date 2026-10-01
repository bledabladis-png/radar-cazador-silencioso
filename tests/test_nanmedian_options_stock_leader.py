"""Tests de regresion: NaN en _zscore_last_in_window y score_mad.

Mismo patron que el bug de index_leaders::robust_intra (2f956ed, 2026-10-01):
np.median propaga NaN silenciosamente. Fix: filtrar NaN antes de calcular.

Nota sobre stock_leader: rolling(10).median() con 1 NaN en la ventana
devuelve NaN (min_periods=10, 9 no-NaN). El fix de score_mad no altera
score_median; solo evita que mad propague NaN. Los tests lo reflejan.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators.options import _zscore_last_in_window


# --- options.py ---

def test_zscore_ventana_con_nan_no_propaga():
    """Ventana con 1 NaN no debe devolver NaN."""
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    z = _zscore_last_in_window(s)
    assert not pd.isna(z)
    assert np.isfinite(z)


def test_zscore_ventana_sin_nan_comportamiento_igual():
    """Sin NaN, el fix no cambia el resultado."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = _zscore_last_in_window(s)
    median = s.median()
    mad = np.median(np.abs(s - median))
    expected = (s.iloc[-1] - median) / (1.4826 * mad)
    assert abs(z - expected) < 1e-12


def test_zscore_ventana_todos_nan_devuelve_0():
    """Ventana todos NaN: valid < 2 -> 0.0."""
    s = pd.Series([np.nan, np.nan, np.nan, np.nan])
    z = _zscore_last_in_window(s)
    assert z == 0.0


def test_zscore_ventana_mad_cero_devuelve_0():
    """Sin dispersion (todos iguales): mad == 0 -> 0.0."""
    s = pd.Series([3.0, 3.0, 3.0, 3.0])
    z = _zscore_last_in_window(s)
    assert z == 0.0


def test_zscore_ultimo_nan_devuelve_0():
    """Ultima observacion NaN -> 0.0."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
    z = _zscore_last_in_window(s)
    assert z == 0.0


# --- stock_leader.py: logica de _mad_filtrado ---

def _mad_filtrado(x):
    """Replica exacta del helper en stock_leader.py."""
    v = x[~np.isnan(x)]
    if len(v) < 2:
        return 0.0
    m = np.median(v)
    return float(np.median(np.abs(v - m)))


def test_mad_filtrado_con_nan_no_propaga():
    """mad con 1 NaN debe ser finito."""
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    mad = _mad_filtrado(x)
    assert not pd.isna(mad)
    assert np.isfinite(mad)


def test_mad_filtrado_sin_nan_comportamiento_igual():
    """Sin NaN, mad identico al calculo original."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    mad_original = np.median(np.abs(x - np.median(x)))
    mad_nuevo = _mad_filtrado(x)
    assert abs(mad_original - mad_nuevo) < 1e-12


def test_mad_filtrado_todos_nan_devuelve_0():
    x = np.array([np.nan, np.nan, np.nan, np.nan])
    assert _mad_filtrado(x) == 0.0


def test_mad_filtrado_un_solo_valor_devuelve_0():
    x = np.array([5.0, np.nan, np.nan, np.nan])
    assert _mad_filtrado(x) == 0.0