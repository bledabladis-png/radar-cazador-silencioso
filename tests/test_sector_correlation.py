import pandas as pd
import numpy as np
from indicators.sector_correlation import compute_sector_correlation

def _make_df(n=70):
    idx = pd.date_range('2026-01-01', periods=n, freq='D')
    data = {ticker: np.random.normal(0, 0.01, n) for ticker in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']}
    return pd.DataFrame(data, index=idx)

def test_compute_sector_correlation_basic():
    df = _make_df()
    matrix, summary = compute_sector_correlation(df)
    assert not summary.empty
    assert not matrix.empty
    # Debe contener ventanas 20 y 60
    assert set(summary['window']) == {20, 60}
    # n_valid_pairs debe ser <= 55
    assert (summary['n_valid_pairs'] <= 55).all()

def test_correlation_positive_perfect():
    idx = pd.date_range('2026-01-01', periods=30, freq='D')
    data = {t: np.linspace(0, 0.1, 30) for t in ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']}
    df = pd.DataFrame(data, index=idx)
    matrix, summary = compute_sector_correlation(df, windows=(20,))
    # Todas las correlaciones deben ser 1
    assert (matrix['correlation'] > 0.999).all()