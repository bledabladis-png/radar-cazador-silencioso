import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from data.validator import validate_market_data, validate_macro_manual

def test_validate_market_data_ok():
    pd.MultiIndex.from_tuples([('Close', 'AAPL')])
    df = pd.DataFrame({('Close', 'AAPL'): [100, 101, 102]*30}, index=range(90))
    valid, issues = validate_market_data(df)
    assert 'AAPL' in valid
    assert len(issues) == 0

def test_validate_market_data_nan():
    pd.MultiIndex.from_tuples([('Close', 'BAD')])
    df = pd.DataFrame({('Close', 'BAD'): [100] + [np.nan]*89}, index=range(90))
    valid, issues = validate_market_data(df)
    assert 'BAD' not in valid

def test_validate_macro_manual_ok():
    df = pd.DataFrame({'date': ['2020-01-01', '2020-02-01'], 'CPI': [2.0, 2.1]})
    ok, issues = validate_macro_manual(df)
    assert ok

def test_validate_macro_manual_no_date():
    df = pd.DataFrame({'CPI': [2.0]})
    ok, issues = validate_macro_manual(df)
    assert not ok
