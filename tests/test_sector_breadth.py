import pandas as pd
from indicators.sector_breadth import compute_sector_breadth

def _make_data():
    idx = pd.date_range('2025-01-01', periods=300, freq='D')
    # DataFrame de stocks con MultiIndex (ticker, campo)
    fields = ['Open','High','Low','Close','Volume']
    data = {}
    for t in ['AAA','BBB']:
        data[(t,'Close')] = [100 + i*0.1 for i in range(300)]
        data[(t,'High')] = [101]*300
        data[(t,'Low')] = [99]*300
        data[(t,'Open')] = [100]*300
        data[(t,'Volume')] = [1000]*300
    df_stocks = pd.DataFrame(data, index=idx)
    df_stocks.columns = pd.MultiIndex.from_tuples(df_stocks.columns)
    # DataFrame de mercado con MultiIndex (ticker, campo)
    market_data = {('XLK','Close'): [100 + i*0.05 for i in range(300)]}
    df_market = pd.DataFrame(market_data, index=idx)
    df_market.columns = pd.MultiIndex.from_tuples(df_market.columns)
    holdings = pd.DataFrame({'etf':['XLK','XLK'], 'ticker':['AAA','BBB']})
    return df_market, df_stocks, holdings

def test_as_of_date_no_lookahead():
    df_market, df_stocks, holdings = _make_data()
    cutoff = df_market.index[200]
    hist = compute_sector_breadth(df_market, df_stocks, holdings, as_of_date=cutoff)
    assert hist.iloc[0]['date'] == cutoff.normalize()
    full = compute_sector_breadth(df_market, df_stocks, holdings)
    assert full.iloc[0]['date'] != cutoff.normalize()

def test_advance_pct_nan():
    idx = pd.date_range('2025-01-01', periods=300, freq='D')
    data = {}
    for t in ['AAA','BBB']:
        data[(t,'Close')] = [100]*300  # sin cambios
        data[(t,'High')] = [101]*300
        data[(t,'Low')] = [99]*300
        data[(t,'Open')] = [100]*300
        data[(t,'Volume')] = [1000]*300
    df_stocks = pd.DataFrame(data, index=idx)
    df_stocks.columns = pd.MultiIndex.from_tuples(df_stocks.columns)
    market_data = {('XLK','Close'): [100]*300}
    df_market = pd.DataFrame(market_data, index=idx)
    df_market.columns = pd.MultiIndex.from_tuples(df_market.columns)
    holdings = pd.DataFrame({'etf':['XLK','XLK'], 'ticker':['AAA','BBB']})
    result = compute_sector_breadth(df_market, df_stocks, holdings)
    assert pd.isna(result.iloc[0]['advance_pct'])

def test_n_valid_momentum_positive():
    idx = pd.date_range('2025-01-01', periods=300, freq='D')
    data = {}
    for t in ['AAA','BBB']:
        data[(t,'Close')] = [100 + i*0.1 for i in range(300)]
        data[(t,'High')] = [101]*300
        data[(t,'Low')] = [99]*300
        data[(t,'Open')] = [100]*300
        data[(t,'Volume')] = [1000]*300
    df_stocks = pd.DataFrame(data, index=idx)
    df_stocks.columns = pd.MultiIndex.from_tuples(df_stocks.columns)
    market_data = {('XLK','Close'): [100 + i*0.05 for i in range(300)]}
    df_market = pd.DataFrame(market_data, index=idx)
    df_market.columns = pd.MultiIndex.from_tuples(df_market.columns)
    holdings = pd.DataFrame({'etf':['XLK','XLK'], 'ticker':['AAA','BBB']})
    result = compute_sector_breadth(df_market, df_stocks, holdings)
    assert result.iloc[0]['n_valid_momentum'] == 2