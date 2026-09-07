import pandas as pd
import numpy as np
from indicators.evidence_matrix import (
    _sign_from_value, _sign_from_pct_above, _wyckoff_sign, compute_evidence_matrix
)

def test_sign_from_value_nan():
    assert pd.isna(_sign_from_value(np.nan))

def test_sign_from_value_positive():
    assert _sign_from_value(1.0) == 1

def test_sign_from_value_negative():
    assert _sign_from_value(-0.5) == -1

def test_sign_from_value_zero():
    assert _sign_from_value(0) == 0

def test_sign_from_pct_above():
    assert _sign_from_pct_above(51.0) == 1
    assert _sign_from_pct_above(50.0) == 0
    assert _sign_from_pct_above(49.0) == -1
    assert pd.isna(_sign_from_pct_above(np.nan))

def test_wyckoff_sign():
    row = {
        'pct_accumulation': 30,
        'pct_markup': 25,
        'pct_distribution': 20,
        'pct_markdown': 25
    }
    assert _wyckoff_sign(row) == 1
    row2 = {'pct_accumulation': 10, 'pct_markup': 10, 'pct_distribution': 40, 'pct_markdown': 40}
    assert _wyckoff_sign(row2) == -1
    row3 = {'pct_accumulation': 20, 'pct_markup': 20, 'pct_distribution': 20, 'pct_markdown': 20}
    assert _wyckoff_sign(row3) == 0

def test_compute_matrix_insufficient():
    # Crear DataFrames mínimos
    sectors = ['XLK', 'XLF']
    breadth = pd.DataFrame({
        'sector': sectors,
        'pct_above_ema50': [60.0, np.nan],
        'n_valid_ema200': [95, 95],
        'n_total': [100, 100],
    })
    flow = pd.DataFrame({
        'sector': sectors,
        'price_ret_20d': [0.02, -0.01],
        'flow_20d_sum': [np.nan, np.nan],
    })
    conc = pd.DataFrame({
        'sector': sectors,
        'flow_median': [np.nan, np.nan],
        'coverage_flow': [0.0, 0.0],
    })
    wyckoff = pd.DataFrame({
        'sector': sectors,
        'pct_accumulation': [40, 40],
        'pct_markup': [30, 30],
        'pct_distribution': [15, 15],
        'pct_markdown': [15, 15],
        'coverage_wyckoff': [90.0, 90.0],
    })
    df = compute_evidence_matrix(breadth, conc, flow, wyckoff, volatility_regime='LOW', liquidity_regime='ABUNDANTE')
    assert df is not None
    assert len(df) == 2
    assert 'EVIDENCIA INSUFICIENTE' in df['alignment_reading'].values
    # Asegurar que NaN no es neutral
    assert df.loc[0, 'primary_flow_evidence'] is pd.NA