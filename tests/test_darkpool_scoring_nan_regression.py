# -*- coding: utf-8 -*-
"""Regresion D-01: darkpool_scoring.robust_zscore propaga NaN via mad.

Bug detectado en auditoria S-01 (2026-10-03). Mismo patron que el fix
de index_leaders (commit 2f956ed) y que los fixes ya aplicados en
options.py:44-56 y fls.py:29-40. darkpool_scoring.py quedo fuera.

Contrato de los tres hermanos (misma semantica, distinta implementacion):
    robust_zscore(series) -> pd.Series indexada, con:
        - los NaN originales permanecen NaN
        - los valores no-NaN obtienen z-score calculado sobre la ventana
          de valores no-NaN
        - mad == 0 -> todos ceros (mismo indice)
        - serie vacia -> serie vacia

Sin fix: np.median(np.abs(series - median)) con 1 NaN -> mad = NaN
-> z de TODA la serie = NaN -> classify_darkpool cae en "Sin historial
suficiente" enmascarando el problema.
"""
import numpy as np
import pandas as pd

from indicators.darkpool_scoring import robust_zscore, _compute_z_for_window


def test_robust_zscore_nan_en_ventana_no_propaga():
    """D-01: 1 NaN en la ventana no debe anular los z-scores no-NaN."""
    s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    z = robust_zscore(s)
    # Los NaN originales siguen NaN
    assert pd.isna(z.iloc[2]), "el NaN original debe seguir siendo NaN"
    # Los no-NaN deben ser finitos
    no_nan = z.dropna()
    assert len(no_nan) == 4, f"esperados 4 valores finitos, hay {len(no_nan)}"
    assert np.all(np.isfinite(no_nan)), "z no-NaN debe ser finito"
    # Cálculo manual: mediana([1,2,4,5]) = 3.0, mad = 1.5
    expected_last = (5.0 - 3.0) / (1.4826 * 1.5)
    assert abs(z.iloc[-1] - expected_last) < 1e-9


def test_robust_zscore_sin_nan_calculo_canonico():
    """Sin NaN, el resultado coincide con el calculo manual."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = robust_zscore(s)
    median = 3.0
    mad = 1.0  # mediana(|x-3|) sobre [2,1,0,1,2] = 1.0
    expected = (s - median) / (1.4826 * mad)
    pd.testing.assert_series_equal(z, expected, check_names=False)


def test_robust_zscore_mad_cero_devuelve_ceros():
    """MAD == 0 -> todos ceros, mismo indice (contrato A3.3-01)."""
    s = pd.Series([5.0, 5.0, 5.0, 5.0])
    z = robust_zscore(s)
    assert len(z) == len(s)
    assert (z == 0).all()


def test_robust_zscore_serie_vacia():
    """Serie vacia -> serie vacia (contrato K-DT3-RUNTIMEWARN)."""
    s = pd.Series([], dtype=float)
    z = robust_zscore(s)
    assert len(z) == 0


def test_robust_zscore_todos_nan():
    """Todos NaN -> todos NaN, mismo indice."""
    s = pd.Series([np.nan, np.nan, np.nan])
    z = robust_zscore(s)
    assert len(z) == 3
    assert z.isna().all()


def test_compute_z_for_window_con_hueco_no_devuelve_nan():
    """El z final de la ventana debe ser finito si el ultimo no-NaN existe."""
    # 60 filas, 1 NaN en el medio. window=52.
    n = 60
    ratio = np.linspace(0.5, 1.5, n)
    ratio[25] = np.nan
    hist = pd.DataFrame({"ratio": ratio})
    z, mom, pct, state = _compute_z_for_window(hist, 52)
    assert np.isfinite(z), f"z={z} debe ser finito con 1 hueco; state={state}"
    assert state != "Sin historial suficiente", (
        "con 59 valores validos en ventana 52, el estado no debe ser "
        "Sin historial suficiente"
    )
