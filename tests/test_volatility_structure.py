import pandas as pd
import numpy as np
from indicators.volatility_structure import compute_volatility_structure

def _make_df(n=100):
    idx = pd.date_range('2026-01-01', periods=n, freq='D')
    data = {
        '^VIX': np.random.normal(20, 3, n),
        '^VIX3M': np.random.normal(22, 3, n),
    }
    return pd.DataFrame(data, index=idx)

def test_volatility_structure_basic():
    df = _make_df()
    out = compute_volatility_structure(df)
    assert not out.empty
    assert 'vix_level' in out.columns
    assert 'vix_percentile_20d' in out.columns
    assert 'term_structure_ratio' in out.columns

def test_volatility_structure_no_vix3m():
    df = _make_df()
    df = df.drop(columns='^VIX3M')
    out = compute_volatility_structure(df)
    assert not out.empty
    assert pd.isna(out['term_structure_ratio'].iloc[0])
    assert out['term_structure_reading'].iloc[0] == 'N/D'