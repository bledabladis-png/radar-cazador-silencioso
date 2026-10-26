# -*- coding: utf-8 -*-
"""Tests DICT3-H05: columna bar_flags en build_ticker_df (contrato v1.8+).

Anade auditoria de calidad por barra SIN cambiar comportamiento de los
detectores ni del scoring. Los consumidores acceden a Open/High/Low/
Close/Volume por nombre y no leen bar_flags.
"""
import numpy as np
import pandas as pd

import indicators.wyckoff_v1 as w1


def _synth(n=30, seed=1):
    rng = np.random.default_rng(seed)
    base = 100 + rng.standard_normal(n).cumsum()
    return pd.DataFrame({
        'Open': base + rng.standard_normal(n) * 0.5,
        'High': base + abs(rng.standard_normal(n)),
        'Low': base - abs(rng.standard_normal(n)),
        'Close': base,
        'Volume': rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=pd.date_range('2025-01-01', periods=n, freq='B'))


def _one(df, ticker='T'):
    mi = df.copy()
    mi.columns = pd.MultiIndex.from_product([mi.columns, [ticker]])
    return w1.build_ticker_df(mi, ticker)


def test_ok_bar():
    out = _one(_synth())
    assert (out['bar_flags'] == 'OK').all()


def test_missing_ohlc():
    df = _synth()
    df.loc[df.index[5], 'Open'] = np.nan
    out = _one(df)
    assert out.loc[out.index[5], 'bar_flags'] == 'MISSING_OHLC'
    rest = out.drop(index=out.index[5])
    assert (rest['bar_flags'] == 'OK').all()


def test_zero_volume():
    df = _synth()
    df.loc[df.index[3], 'Volume'] = 0.0
    df.loc[df.index[7], 'Volume'] = np.nan
    out = _one(df)
    assert out.loc[out.index[3], 'bar_flags'] == 'ZERO_VOLUME'
    assert out.loc[out.index[7], 'bar_flags'] == 'ZERO_VOLUME'


def test_no_range():
    df = _synth()
    df.loc[df.index[10], 'High'] = df.loc[df.index[10], 'Close']
    df.loc[df.index[10], 'Low'] = df.loc[df.index[10], 'Close']
    out = _one(df)
    assert out.loc[out.index[10], 'bar_flags'] == 'NO_RANGE'


def test_combinacion_missing_no_range():
    df = _synth()
    idx = df.index[12]
    df.loc[idx, 'Open'] = np.nan
    df.loc[idx, 'High'] = df.loc[idx, 'Close']
    df.loc[idx, 'Low'] = df.loc[idx, 'Close']
    out = _one(df)
    assert out.loc[idx, 'bar_flags'] == 'MISSING_OHLC;NO_RANGE'


def test_detectores_no_cambian_output():
    """detect_spring y detect_sow no leen bar_flags: outputs identicos."""
    df = _synth(300, seed=7)
    out_con = _one(df)
    out_sin = out_con.drop(columns=['bar_flags'])
    spring_con = w1.detect_spring(out_con, 'T')
    spring_sin = w1.detect_spring(out_sin, 'T')
    pd.testing.assert_series_equal(spring_con, spring_sin)
    sow_con = w1.detect_sow(out_con, 'T', window=20, x_atr=0.25, y_vol=1.10)
    sow_sin = w1.detect_sow(out_sin, 'T', window=20, x_atr=0.25, y_vol=1.10)
    pd.testing.assert_series_equal(sow_con, sow_sin)


def test_wyckoff_score_no_cambia_output():
    """wyckoff_score no lee bar_flags: tupla identica con/sin la columna."""
    df = _synth(500, seed=42)
    out_con = _one(df)
    out_sin = out_con.drop(columns=['bar_flags'])
    score_con = w1.wyckoff_score(out_con, 'T')
    score_sin = w1.wyckoff_score(out_sin, 'T')
    assert len(score_con) == len(score_sin)
    for a, b in zip(score_con, score_sin):
        if hasattr(a, 'equals'):
            pd.testing.assert_series_equal(a, b)
        else:
            assert a == b, 'wyckoff_score diverge con bar_flags'
