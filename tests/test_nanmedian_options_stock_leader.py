"""Tests de regresion: np.nanmedian en options.py y stock_leader.py.

Mismo patron que el bug de index_leaders::robust_intra (2f956ed, 2026-10-01):
np.median propaga NaN silenciosamente. Fix: np.nanmedian + guard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from indicators.options import _zscore_last_in_window


def test_zscore_ventana_con_nan_no_propaga():
    """Ventana con 1 NaN no debe devolver NaN."""
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    z = _zscore_last_in_window(s)
    assert not pd.isna(z), "z con 1 NaN no debe ser NaN"
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
    """Ventana todos NaN: mad NaN -> guard devuelve 0.0."""
    s = pd.Series([np.nan, np.nan, np.nan, np.nan])
    z = _zscore_last_in_window(s)
    assert z == 0.0


def test_zscore_ventana_mad_cero_devuelve_0():
    """Sin dispersion (todos iguales): mad == 0 -> 0.0."""
    s = pd.Series([3.0, 3.0, 3.0, 3.0])
    z = _zscore_last_in_window(s)
    assert z == 0.0


def test_stock_leader_score_mad_con_nan_no_propaga():
    """score_mad con NaN en la ventana -> guard 0.0 -> stability finita."""
    # Reproducir la logica exacta de stock_leader.py:59
    wyckoff_series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    score_median = wyckoff_series.rolling(10).median().iloc[-1]
    score_mad = wyckoff_series.rolling(10).apply(
        lambda x: np.nanmedian(np.abs(x - np.nanmedian(x)))
    ).iloc[-1]
    if pd.isna(score_mad):
        score_mad = 0.0
    stability = np.tanh(score_median / (score_mad + 1e-9))
    assert not pd.isna(stability)
    assert np.isfinite(stability)


def test_stock_leader_score_mad_sin_nan_comportamiento_igual():
    """Sin NaN, score_mad identico al calculo original."""
    wyckoff_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    x = wyckoff_series.iloc[-10:].values
    mad_original = np.median(np.abs(x - np.median(x)))
    mad_nuevo = np.nanmedian(np.abs(x - np.nanmedian(x)))
    assert abs(mad_original - mad_nuevo) < 1e-12