import pandas as pd
from indicators.sector_concentration import safe_quantile, compute_sector_concentration

def test_safe_quantile_min_obs():
    s = pd.Series([1,2,3,4])
    assert pd.isna(safe_quantile(s, 0.5))
    s = pd.Series([1,2,3,4,5])
    assert safe_quantile(s, 0.5) == 3.0

def test_compute_concentration_synthetic():
    # full_metrics_df con 5 tickers, sector XLK
    metrics = pd.DataFrame({
        'ticker': ['A','B','C','D','E'],
        'sector': ['XLK']*5,
        'rs': [1.1, 0.9, 1.2, 0.8, 1.0],
        'rs_mom': [0.01, -0.02, 0.03, 0.00, 0.02],
        'flow_proxy_z': [0.5, 0.2, -0.1, 0.3, 0.0],
        'wyckoff_score': [0.8, 0.6, 0.7, 0.5, 0.9],
        'wls': [1.0, 0.7, 1.2, 0.6, 0.9],
    })
    # df_stocks con Close para generar ret20
    idx = pd.date_range('2026-01-01', periods=30, freq='D')
    df_stocks = pd.DataFrame({
        ('A','Close'): [10 + i*0.1 for i in range(30)],
        ('B','Close'): [10 + i*0.2 for i in range(30)],
        ('C','Close'): [10 + i*0.3 for i in range(30)],
        ('D','Close'): [10 + i*0.05 for i in range(30)],
        ('E','Close'): [10 + i*0.15 for i in range(30)],
    }, index=idx)
    df_stocks.columns = pd.MultiIndex.from_tuples(df_stocks.columns)
    holdings = pd.DataFrame({'etf': ['XLK']*5, 'ticker': ['A','B','C','D','E']})
    result = compute_sector_concentration(df_stocks, holdings, metrics, reference_date=df_stocks.index[-1])
    assert not result.empty
    assert len(result) == 1
    assert result.iloc[0]['sector'] == 'XLK'
    assert pd.notna(result.iloc[0]['top1_positive_return_concentration'])
    assert pd.notna(result.iloc[0]['rs_median'])


def test_compute_concentration_sin_reference_date_valueerror():
    """FU-008 (2026-09-13): reference_date obligatorio."""
    metrics = pd.DataFrame({
        'ticker': ['A'], 'sector': ['XLK'],
        'rs': [1.0], 'rs_mom': [0.01], 'flow_proxy_z': [0.1],
        'wyckoff_score': [0.5], 'wls': [1.0],
    })
    idx = pd.date_range('2026-01-01', periods=30, freq='D')
    df_stocks = pd.DataFrame({('A','Close'): [10 + i*0.1 for i in range(30)]}, index=idx)
    df_stocks.columns = pd.MultiIndex.from_tuples(df_stocks.columns)
    holdings = pd.DataFrame({'etf': ['XLK'], 'ticker': ['A']})
    import pytest
    with pytest.raises(ValueError, match='reference_date'):
        compute_sector_concentration(df_stocks, holdings, metrics)
