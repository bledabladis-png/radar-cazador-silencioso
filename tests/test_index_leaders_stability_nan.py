# -*- coding: utf-8 -*-
"""Regresion D-02: index_leaders stability propaga NaN via mad.

Bug detectado en auditoria S-01 (2026-10-03). Mismo patron que D-01
(darkpool_scoring) y que el fix previo de robust_intra en este mismo
fichero (2026-10-01: QQQ top20 con SPCX wyckoff_score NaN).

Sintoma: compute_index_leaders() usa inline
    np.median(np.abs(x - np.median(x)))
sobre rolling(10).apply. Con 1 NaN en la ventana, np.median propaga
NaN -> score_mad = NaN -> stability = tanh(x/NaN) = NaN. Sin guard.

El hermano stock_leader.py:66-69 ya tiene el fix (funcion _mad_filtrado
+ guard pd.isna(score_mad)). index_leaders quedo fuera.

Este test verifica el contrato del fix: expone _mad_filtrado como
funcion de modulo y comprueba que ignora NaN.
"""
import numpy as np

from indicators.index_leaders import _mad_filtrado


def test_mad_filtrado_1_nan_no_propaga():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    m = _mad_filtrado(x)
    assert np.isfinite(m), f"mad debe ser finito, es {m}"
    # 9 validos: [1,2,4,5,6,7,8,9,10], mediana=6, |x-6| mediana=2
    assert abs(m - 2.0) < 1e-9


def test_mad_filtrado_sin_nan():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = _mad_filtrado(x)
    # mediana=3, |x-3|=[2,1,0,1,2], mediana=1
    assert abs(m - 1.0) < 1e-9


def test_mad_filtrado_todos_nan():
    x = np.array([np.nan, np.nan, np.nan])
    assert _mad_filtrado(x) == 0.0


def test_mad_filtrado_un_solo_valido():
    x = np.array([5.0, np.nan, np.nan])
    assert _mad_filtrado(x) == 0.0


def test_stability_finita_con_nan():
    """Reproduce la cadena completa: median+mad+tanh."""
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    med = np.nanmedian(x)
    mad = _mad_filtrado(x)
    stability = np.tanh(med / (mad + 1e-9))
    assert np.isfinite(stability)
