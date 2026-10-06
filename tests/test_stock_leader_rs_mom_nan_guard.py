# -*- coding: utf-8 -*-
"""Tests de regresion rs_mom NaN por huecos residuales (2026-10-06).

Bug: `rs = close / price_etf` puede tener NaN residuales por festivos
USA (Labor Day, Thanksgiving, MLK, Presidents, etc.) presentes en el
indice comun. `np.log(rs).diff(20).iloc[-1]` propaga el NaN al ultimo
valor y `rs_mom` sale NaN para toda la serie de un sector.

Impacto observado en CI 2026-10-06: rs_mom NaN en los 20 tickers de
analisis_lideres.csv. Consecuencia: n_valid_momentum=0 en
sector_concentration.csv, seccion 'Momentum de Precio - Sectores'
vacia, y f-string imprimia 'nan%' en lugar de 'N/D'.
"""
import numpy as np
import pandas as pd


def test_rs_mom_sin_dropna_propaga_nan():
    """Reproduce el bug: NaN en posicion -21 -> rs_mom NaN."""
    rs = pd.Series([1.0 + i * 0.01 for i in range(30)])
    rs.iloc[-21] = np.nan
    rs_mom_old = np.log(rs).diff(20).iloc[-1]
    assert pd.isna(rs_mom_old), "Bug no reproducido: se esperaba NaN"


def test_rs_mom_con_dropna_y_guard_evita_nan():
    """Fix: dropna() + guard len>=21 -> rs_mom con valor."""
    rs = pd.Series([1.0 + i * 0.01 for i in range(30)])
    rs.iloc[-21] = np.nan
    rs_clean = rs.dropna()
    rs_mom_new = np.log(rs_clean).diff(20).iloc[-1] if len(rs_clean) >= 21 else np.nan
    assert pd.notna(rs_mom_new), "Fix no funciona: se esperaba valor"


def test_rs_mom_serie_corta_devuelve_nan_por_guard():
    """Serie con menos de 21 puntos -> NaN por guard, no por excepcion."""
    rs = pd.Series([1.0 + i * 0.01 for i in range(15)])
    rs_mom = np.log(rs).diff(20).iloc[-1] if len(rs) >= 21 else np.nan
    assert pd.isna(rs_mom)


def test_fmt_num_nan_devuelve_nd():
    """Verificacion del contrato de _fmt_num: NaN -> 'N/D'."""
    from src.report.helpers import _fmt_num
    assert _fmt_num(float('nan'), '{:.2%}') == 'N/D'
    assert _fmt_num(float('nan'), '{:.2f}') == 'N/D'
