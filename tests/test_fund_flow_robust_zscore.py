"""Tests del contrato fund_flow_robust_zscore (A5-76 / F6-05).

Contrato aprobado por auditor externo 2026-09-27.
Los 10 casos obligatorios cubren:
  1. Calculo manual de ventana pequena.
  2. MAD > 0, distribucion normal.
  3. MAD == 0 & s == median -> 0.
  4. MAD == 0 & s != median -> NaN.
  5. Ultimo valor NaN -> NaN.
  6. Ventana insuficiente -> NaN.
  7. Outlier superior -> clip +5.
  8. Outlier inferior -> clip -5.
  9. Serie vacia sin warnings.
  10. Sin ffill: NaN intermedio no se imputa.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from data.providers._fund_flow_utils import fund_flow_robust_zscore


# ============================================================
# 1. Calculo manual de ventana pequena
# ============================================================
def test_ventana_pequena_calculo_literal():
    """window=5. Valores: [10,12,14,13,11]. Calculo a mano.

    median = 12
    residuos = |[10-12, 12-12, 14-12, 13-12, 11-12]| = [2, 0, 2, 1, 1]
    MAD = median([0, 1, 1, 2, 2]) = 1
    z = (11 - 12) / (1.4826 * 1) = -1 / 1.4826 = -0.674486...
    """
    s = pd.Series([10.0, 12.0, 14.0, 13.0, 11.0])
    z = fund_flow_robust_zscore(s, window=5, min_periods=5)
    expected = -1.0 / 1.4826
    assert z.iloc[-1] == pytest.approx(expected, abs=1e-6)


# ============================================================
# 2. MAD > 0, distribucion normal
# ============================================================
def test_mad_positivo_z_dentro_de_rango():
    """Serie con dispersion moderada: z dentro de [-5, +5]."""
    s = pd.Series([10.0, 11.5, 12.0, 11.8, 12.3, 11.9, 12.1, 12.4,
                   12.0, 11.7, 12.2, 11.9, 12.1, 12.3, 12.0])
    z = fund_flow_robust_zscore(s, window=10, min_periods=5)
    last = z.iloc[-1]
    assert pd.notna(last)
    assert -5.0 <= last <= 5.0


# ============================================================
# 3. MAD == 0 & s == median
# ============================================================
def test_mad_cero_s_igual_median_devuelve_cero():
    """Serie constante: sin desviacion, z = 0."""
    s = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
    z = fund_flow_robust_zscore(s, window=5, min_periods=5)
    assert z.iloc[-1] == 0.0


# ============================================================
# 4. MAD == 0 & s != median -> NaN
# ============================================================
def test_mad_cero_s_distinto_median_devuelve_nan():
    """Serie plana, ultimo valor distinto. Sin escala: z no definido."""
    s = pd.Series([10.0, 10.0, 10.0, 10.0, 11.0])
    z = fund_flow_robust_zscore(s, window=5, min_periods=5)
    assert pd.isna(z.iloc[-1])


# ============================================================
# 5. Ultimo valor NaN -> NaN
# ============================================================
def test_ultimo_valor_nan_devuelve_nan():
    """No se rescata el ultimo no-NaN. NaN en fila actual -> NaN."""
    s = pd.Series([10.0, 12.0, 14.0, 13.0, np.nan])
    z = fund_flow_robust_zscore(s, window=5, min_periods=5)
    assert pd.isna(z.iloc[-1])


# ============================================================
# 6. Ventana insuficiente -> NaN
# ============================================================
def test_ventana_insuficiente_devuelve_nan():
    """Serie con menos valores validos que min_periods: NaN.

    window=20, min_periods=10. Serie de 5 valores. La ventana aun
    no contiene 10 valores validos: el resultado debe ser NaN.
    """
    s = pd.Series([10.0, 12.0, 14.0, 13.0, 11.0])
    z = fund_flow_robust_zscore(s, window=20, min_periods=10)
    assert pd.isna(z.iloc[-1])


# ============================================================
# 7. Outlier superior -> clip +5
# ============================================================
def test_outlier_superior_clip_a_mas_5():
    """Outlier muy por encima de la mediana: z bruto > 5 -> clip 5.0.

    values = [1, 2, 3, 4, 100]
    median = 3
    residuos = [2, 1, 0, 1, 97]
    MAD = median([0, 1, 1, 2, 97]) = 1
    z = (100 - 3) / (1.4826 * 1) ~ 65.4 -> clip 5.0
    """
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    z = fund_flow_robust_zscore(s, window=5, min_periods=5)
    assert z.iloc[-1] == 5.0


# ============================================================
# 8. Outlier inferior -> clip -5
# ============================================================
def test_outlier_inferior_clip_a_menos_5():
    """Outlier muy por debajo de la mediana: z bruto < -5 -> clip -5.0."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, -100.0])
    z = fund_flow_robust_zscore(s, window=5, min_periods=5)
    assert z.iloc[-1] == -5.0


# ============================================================
# 9. Serie vacia sin warnings
# ============================================================
def test_serie_vacia_sin_warnings():
    """Serie vacia: no debe emitir RuntimeWarning de numpy/pandas."""
    s = pd.Series([], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        z = fund_flow_robust_zscore(s, window=5, min_periods=5)
    assert len(z) == 0


# ============================================================
# 10. Sin ffill: NaN intermedio no se imputa
# ============================================================
def test_nan_intermedio_no_se_imputa():
    """NaN en medio de la serie: ventana insuficiente si cae por debajo de min_p.

    En [10, 12, NaN, 14, 16] con min_periods=5 solo hay 4 validos.
    El ultimo valor (16) es valido pero la ventana < min_periods.
    """
    s = pd.Series([10.0, 12.0, np.nan, 14.0, 16.0])
    z = fund_flow_robust_zscore(s, window=5, min_periods=5)
    assert pd.isna(z.iloc[-1])

# ============================================================
# Tests de fund_flow_robust_zscore_with_regime
# ============================================================
from data.providers._fund_flow_utils import (
    fund_flow_robust_zscore_with_regime,
    REGIME_NORMAL,
    REGIME_SATURATED,
    REGIME_MAD0_SAME,
    REGIME_MAD0_NAN,
    REGIME_INSUFFICIENT,
)


def test_regime_normal():
    s = pd.Series([10.0, 12.0, 14.0, 13.0, 11.0])
    z, r = fund_flow_robust_zscore_with_regime(s, window=5, min_periods=5)
    assert r.iloc[-1] == REGIME_NORMAL
    assert z.iloc[-1] == pytest.approx(-1.0 / 1.4826, abs=1e-6)


def test_regime_mad0_same():
    s = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
    z, r = fund_flow_robust_zscore_with_regime(s, window=5, min_periods=5)
    assert r.iloc[-1] == REGIME_MAD0_SAME
    assert z.iloc[-1] == 0.0


def test_regime_mad0_nan():
    s = pd.Series([10.0, 10.0, 10.0, 10.0, 11.0])
    z, r = fund_flow_robust_zscore_with_regime(s, window=5, min_periods=5)
    assert r.iloc[-1] == REGIME_MAD0_NAN
    assert pd.isna(z.iloc[-1])


def test_regime_saturated_superior():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    z, r = fund_flow_robust_zscore_with_regime(s, window=5, min_periods=5)
    assert r.iloc[-1] == REGIME_SATURATED
    assert z.iloc[-1] == 5.0


def test_regime_saturated_inferior():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, -100.0])
    z, r = fund_flow_robust_zscore_with_regime(s, window=5, min_periods=5)
    assert r.iloc[-1] == REGIME_SATURATED
    assert z.iloc[-1] == -5.0


def test_regime_insufficient_ventana_corta():
    s = pd.Series([10.0, 11.0, 12.0])
    z, r = fund_flow_robust_zscore_with_regime(s, window=20, min_periods=10)
    assert r.iloc[-1] == REGIME_INSUFFICIENT
    assert pd.isna(z.iloc[-1])


def test_regime_insufficient_ultimo_nan():
    s = pd.Series([10.0, 12.0, 14.0, 13.0, np.nan])
    z, r = fund_flow_robust_zscore_with_regime(s, window=5, min_periods=5)
    assert r.iloc[-1] == REGIME_INSUFFICIENT
    assert pd.isna(z.iloc[-1])


def test_regime_z_coincide_con_funcion_sin_regime():
    """El z devuelto por _with_regime coincide con la funcion simple."""
    s = pd.Series([10.0, 11.5, 12.0, 11.8, 12.3, 11.9, 12.1, 12.4,
                   12.0, 11.7, 12.2, 11.9, 12.1, 12.3, 12.0])
    z_simple = fund_flow_robust_zscore(s, window=10, min_periods=5)
    z_regime, _ = fund_flow_robust_zscore_with_regime(s, window=10, min_periods=5)
    pd.testing.assert_series_equal(z_simple, z_regime)