import pandas as pd
import numpy as np
from indicators.evidence_matrix import compute_evidence_matrix

def _make_dfs(sectors, price, breadth, flow, proxy, wyckoff):
    n = len(sectors)
    return (pd.DataFrame({'sector': sectors, 'pct_above_ema50': breadth, 'n_valid_ema50': [10]*n, 'n_total': [10]*n}),
            pd.DataFrame({'sector': sectors, 'price_ret_20d': price, 'flow_20d_sum': flow, 'n_obs_20d': [20]*n}),
            pd.DataFrame({'sector': sectors, 'flow_median': proxy, 'coverage_flow': [80.0]*n}),
            pd.DataFrame({'sector': sectors, 'pct_accumulation': [40]*n, 'pct_markup': [30]*n, 'pct_distribution': [15]*n, 'pct_markdown': [15]*n, 'coverage_wyckoff': [80.0]*n}))

def test_n_valid_menor_3_insuficiente():
    sectors = ['XLK','XLF','XLV']
    b, fc, conc, wy = _make_dfs(sectors, [np.nan, np.nan, 0.01], [50.0, 60.0, 70.0], [np.nan, np.nan, 1.0], [np.nan, np.nan, np.nan], None)
    df = compute_evidence_matrix(b, conc, fc, wy)
    assert df.iloc[0]['alignment_reading'] == 'EVIDENCIA INSUFICIENTE'
    assert df.iloc[0]['n_sector_evidence_valid'] == 2  # corregido

def test_n_valid_3_positivo():
    sectors = ['XLK']
    b, fc, conc, wy = _make_dfs(sectors, [0.02], [60.0], [1.0], [0.5], None)
    df = compute_evidence_matrix(b, conc, fc, wy)
    assert df.iloc[0]['alignment_reading'] == 'EVIDENCIA PREDOMINANTEMENTE FAVORABLE'
    assert df.iloc[0]['n_sector_evidence_valid'] == 5

def test_nan_no_neutral():
    sectors = ['XLK']
    b, fc, conc, wy = _make_dfs(sectors, [np.nan], [60.0], [1.0], [np.nan], None)
    df = compute_evidence_matrix(b, conc, fc, wy)
    assert df.iloc[0]['n_sector_evidence_valid'] == 3
    assert df.iloc[0]['n_sector_evidence_neutral'] == 0