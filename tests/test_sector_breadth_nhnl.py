import pandas as pd
import numpy as np
from indicators.sector_breadth import compute_sector_breadth

def _make_single_ticker(high_values):
    idx = pd.date_range('2025-01-01', periods=300, freq='D')
    data = {
        ('AAA','Close'): [100 + i*0.1 for i in range(300)],
        ('AAA','High'): high_values,
        ('AAA','Low'): [99]*300,
        ('AAA','Open'): [100]*300,
        ('AAA','Volume'): [1000]*300,
    }
    df_stocks = pd.DataFrame(data, index=idx)
    df_stocks.columns = pd.MultiIndex.from_tuples(df_stocks.columns)
    market_data = {('XLK','Close'): [100 + i*0.05 for i in range(300)]}
    df_market = pd.DataFrame(market_data, index=idx)
    df_market.columns = pd.MultiIndex.from_tuples(df_market.columns)
    holdings = pd.DataFrame({'etf':['XLK'], 'ticker':['AAA']})
    return df_market, df_stocks, holdings

def test_no_new_high_when_current_below_prev_max():
    # Máximo previo 102, actual 100 => no NH
    df_market, df_stocks, holdings = _make_single_ticker([102]*299 + [100])
    result = compute_sector_breadth(df_market, df_stocks, holdings)
    assert result.iloc[0]['new_highs'] == 0

def test_new_high_when_exceeds_prev_max():
    # Máximo previo 101, actual 200 => NH
    df_market, df_stocks, holdings = _make_single_ticker([101]*299 + [200])
    result = compute_sector_breadth(df_market, df_stocks, holdings)
    assert result.iloc[0]['new_highs'] == 1