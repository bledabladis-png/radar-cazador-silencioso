import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from src.utils import robust_zscore, tanh_normalize, sigmoid, get_col, clean_oil_prices

def test_robust_zscore_normal():
    s = pd.Series([1,2,3,4,5,6,7,8,9,10] * 20)
    z = robust_zscore(s, window=60)
    assert not z.isna().all()
    assert -5 <= z.iloc[-1] <= 5

def test_robust_zscore_with_outliers():
    np.random.seed(42)
    s = pd.Series(np.random.randn(120).tolist() + [8.0])
    z = robust_zscore(s, window=60)
    assert z.iloc[-1] > 2

def test_tanh_normalize():
    s = pd.Series(np.random.randn(200))
    t = tanh_normalize(s)
    # Solo los ultimos valores validos (sin NaN)
    valid = t.dropna()
    assert len(valid) > 50
    assert valid.iloc[-20:].between(-1, 1).all()

def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert sigmoid(2) > 0.8

def test_get_col_multindex():
    df = pd.DataFrame({'A_Close': [1,2,3]})
    assert get_col(df, 'A', 'Close').tolist() == [1,2,3]

def test_clean_oil_prices():
    cols = pd.MultiIndex.from_tuples([('Close', 'CL=F'), ('Close', 'BZ=F')])
    df = pd.DataFrame({('Close', 'CL=F'): [-1.0, 50.0], ('Close', 'BZ=F'): [70.0, 75.0]}, index=[0,1])
    df_clean = clean_oil_prices(df)
    assert (df_clean[('Close', 'CL=F')] > 0).all()
