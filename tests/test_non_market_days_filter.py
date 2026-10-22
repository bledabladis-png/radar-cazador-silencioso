# -*- coding: utf-8 -*-
"""Test: _nan_usa_on_non_market_days filtra US_EQUITY en festivos NYSE.

Bug: Yahoo devuelve closes sinteticos para tickers USA en festivos
NYSE (2021-12-31, observacion New Year). Filtro aplicado en cada run.
"""

import pandas as pd

from src.utils import _nan_usa_on_non_market_days


def test_filtra_us_en_festivo_preserva_eu():
    """2021-12-31: XLB (US) -> NaN, FRE.DE (EU) -> intacto."""
    idx = pd.DatetimeIndex(['2021-12-30', '2021-12-31'])
    df = pd.DataFrame({
        ('Close', 'XLB'): [100.0, 100.0],
        ('Close', 'FRE.DE'): [200.0, 200.0],
    }, index=idx)
    out, info = _nan_usa_on_non_market_days(df)
    # 2021-12-30 bursatil: nada tocado
    assert out.loc[idx[0], ('Close', 'XLB')] == 100.0
    assert out.loc[idx[0], ('Close', 'FRE.DE')] == 200.0
    # 2021-12-31 no bursatil: XLB US -> NaN
    assert pd.isna(out.loc[idx[1], ('Close', 'XLB')])
    # 2021-12-31 no bursatil: FRE.DE EU -> intacto
    assert out.loc[idx[1], ('Close', 'FRE.DE')] == 200.0
    assert info['n_cells'] >= 1
    assert info['n_dates'] == 1


def test_idempotente():
    """Segunda pasada no cambia nada."""
    idx = pd.DatetimeIndex(['2021-12-30', '2021-12-31'])
    df = pd.DataFrame({
        ('Close', 'XLB'): [100.0, 100.0],
    }, index=idx)
    out1, info1 = _nan_usa_on_non_market_days(df)
    assert info1['n_cells'] == 1
    out2, info2 = _nan_usa_on_non_market_days(out1)
    assert info2['n_cells'] == 0


def test_no_modifica_input_original():
    """El input no se muta (funcion pura sobre copia)."""
    idx = pd.DatetimeIndex(['2021-12-31'])
    df = pd.DataFrame({
        ('Close', 'XLB'): [100.0],
    }, index=idx)
    _ = _nan_usa_on_non_market_days(df)
    assert df.loc[idx[0], ('Close', 'XLB')] == 100.0


def test_sin_fechas_no_bursatiles_noop():
    """Sin fechas no-bursatiles, no-op."""
    idx = pd.DatetimeIndex(['2021-12-30'])
    df = pd.DataFrame({
        ('Close', 'XLB'): [100.0],
    }, index=idx)
    out, info = _nan_usa_on_non_market_days(df)
    assert info['n_cells'] == 0
    assert out.loc[idx[0], ('Close', 'XLB')] == 100.0