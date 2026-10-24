# -*- coding: utf-8 -*-
"""Tests de fixes 2026-10-22 en wyckoff_v1.py.

B3: max_age_m default 30 (contrato v1.9).
B5: spring/SOS con min_periods=EVENT_WINDOW.
B2: decision documentada (WONT_FIX razonado en cabecera del modulo).
B4: WONT_FIX razonado (fillna(Close) es contrato v1.8, protegido por
    test_regresion_i2_nan_internos_no_degenera).
"""

import numpy as np
import pandas as pd

from indicators.wyckoff_v1 import (
    detect_spring,
    detect_sos,
    detect_sow,
    build_ticker_df,
    wyckoff_score,
    EVENT_WINDOW,
)
from src.market_calendar import get_trading_days


def _synth(n=100, seed=42):
    np.random.seed(seed)
    sessions = get_trading_days('2022-01-01', '2025-12-31')[:n]
    close = 100 + np.cumsum(np.random.randn(len(sessions)) * 0.5)
    return pd.DataFrame({
        'Open': close - 0.1,
        'High': close + 0.5,
        'Low': close - 0.5,
        'Close': close,
        'Volume': np.random.randint(100000, 1000000, len(sessions)).astype(float),
    }, index=sessions)


def _to_multi(df, ticker='T'):
    out = pd.DataFrame({(col, ticker): df[col] for col in df.columns})
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def test_b5_sos_no_dispara_sin_baseline():
    """B5: SOS no dispara antes de EVENT_WINDOW sesiones."""
    df = _synth(EVENT_WINDOW + 30)
    sos = detect_sos(df, 'SYNTH')
    assert sos.iloc[:EVENT_WINDOW].sum() == 0


def test_b5_spring_no_dispara_sin_baseline():
    """B5: spring no dispara antes de EVENT_WINDOW sesiones."""
    df = _synth(EVENT_WINDOW + 30)
    spring = detect_spring(df, 'SYNTH')
    assert spring.iloc[:EVENT_WINDOW].sum() == 0


def test_b4_wont_fix_contrato_v1_8():
    """B4 (WONT_FIX): build_ticker_df rellena H/L/O con Close, Volume con 0.

    Contrato v1.8. Si este test se cambia, revisar tambien
    test_regresion_i2_nan_internos_no_degenera (bug K-INDEX-RANGE-01).
    """
    df = _synth(20)
    df.loc[df.index[5], 'Open'] = np.nan
    df.loc[df.index[10], 'Volume'] = np.nan
    mi = _to_multi(df)
    built = build_ticker_df(mi, 'T')
    assert built['Open'].notna().all(), 'contrato v1.8: Open fillna(Close)'
    assert built['Volume'].notna().all(), 'contrato v1.8: Volume fillna(0)'
    assert built.loc[built.index[5], 'Open'] == built.loc[built.index[5], 'Close']
    assert built.loc[built.index[10], 'Volume'] == 0.0


def test_b3_detect_sow_firma_estable():
    """B3: detect_sow ejecuta sin max_age_m."""
    df = _synth(200, seed=7)
    sow = detect_sow(df, 'SYNTH', window=20, x_atr=0.25, y_vol=1.10)
    assert len(sow) == len(df)


def test_b2_trend_tolera_1_gap():
    """B2 (WONT_FIX documentado): con dropna, 1 gap no rompe el score."""
    df = _synth(500, seed=42)
    df_huecos = df.drop([df.index[100]])
    mi = _to_multi(df_huecos)
    combined, struct, _, _, _, _, _ = wyckoff_score(mi, 'T')
    n_valid = struct.dropna().shape[0]
    assert n_valid > 100, f'B2 dropna: solo {n_valid} validas de {len(mi)}'


def test_b2_documentacion_wont_fix():
    """B2: la decision esta documentada en la cabecera del modulo."""
    from indicators import wyckoff_v1
    src = open(wyckoff_v1.__file__, encoding='utf-8').read()
    assert 'P7-ROLLING-WINDOW-GAPS (2026-10-22): decision documentada' in src