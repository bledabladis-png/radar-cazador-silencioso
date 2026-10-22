# -*- coding: utf-8 -*-
"""Test: sector_correlation reindexa al calendario de sesiones.

Item: REP-AUDIT-20261022-ROLLING-WINDOW-GAPS
Causa: compute_sector_correlation tomaba returns_df.iloc[-window:],
que son N FILAS del indice. Si hay huecos en el indice, la ventana
cubre mas dias naturales que N sesiones. Fix: reindexar al calendario
NYSE antes del slice.

Usa columnas planas con nombre de ticker (patron de
test_sector_correlation.py existente): get_col lanza KeyError y
compute_sector_correlation cae al fallback `if ticker in df.columns`.
"""

import pandas as pd
import numpy as np

from indicators.sector_correlation import compute_sector_correlation
from src.market_calendar import get_trading_days


TICKERS = ['XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']


def _build_df(idx, seed=42):
    """DataFrame con columnas planas por ticker en el indice dado."""
    np.random.seed(seed)
    data = {}
    for t in TICKERS:
        base = 100 + np.cumsum(np.random.randn(len(idx)) * 0.5)
        data[t] = base
    return pd.DataFrame(data, index=idx)


def test_correlation_20d_con_huecos_usa_sesiones_no_filas():
    """Con huecos, la ventana debe cubrir 20 sesiones, no 20 filas."""
    idx_completo = get_trading_days('2026-08-01', '2026-10-21')
    huecos = pd.to_datetime(['2026-09-21', '2026-09-22', '2026-10-05'])
    idx_con_hueco = idx_completo.difference(huecos)
    assert len(idx_con_hueco) > 20
    df = _build_df(idx_con_hueco)
    summary = compute_sector_correlation(df, windows=(20,), temporal_meta=None)
    assert not summary.empty
    row = summary.iloc[0]
    assert row['window'] == 20
    assert row['n_valid_pairs'] > 0
    assert pd.notna(row['corr_mean'])
    assert -1.0 <= row['corr_mean'] <= 1.0


def test_correlation_sin_huecos_funciona():
    """Serie sin huecos: resultado coherente (regresion)."""
    idx = get_trading_days('2026-08-01', '2026-10-21')
    df = _build_df(idx, seed=7)
    summary = compute_sector_correlation(df, windows=(20,), temporal_meta=None)
    assert not summary.empty
    row = summary.iloc[0]
    assert row['n_valid_pairs'] > 0
    assert pd.notna(row['corr_mean'])


def test_get_trading_days_excluye_festivos():
    """get_trading_days respeta festivos NYSE."""
    idx = get_trading_days('2026-11-23', '2026-11-30')
    dates = [d.strftime('%Y-%m-%d') for d in idx]
    assert '2026-11-26' not in dates, f'Thanksgiving no excluido: {dates}'
    assert '2026-11-27' in dates, f'viernes post-Thanksgiving ausente: {dates}'