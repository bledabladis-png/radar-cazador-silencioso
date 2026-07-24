import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from regimes.financial_conditions import compute_liquidity_score

def make_df(tickers_dict):
    cols = []
    data = {}
    for t in tickers_dict:
        cols.append(('Close', t))
        data[('Close', t)] = tickers_dict[t]
    return pd.DataFrame(data, index=range(len(data[cols[0]])))

def test_liquidity_crisis():
    n = 120
    vix = [15]*20 + [20]*20 + [35]*40 + [40]*40
    df = make_df({
        '^VIX': vix,
        'HYG': [80]*n,
        'LQD': [120]*n,
        'DX-Y.NYB': [100]*n,
        '^TNX': [4]*n,
        '^FVX': [4]*n,
    })
    score, regime, conf = compute_liquidity_score(df)
    assert regime == 'CRISIS'
    assert conf > 0.5

def test_liquidity_returns_valid_types():
    n = 200
    np.random.seed(123)
    noise = lambda: np.random.randn(n).cumsum() * 0.1
    df = make_df({
        '^VIX': 20 + noise(),
        'HYG': 80 + noise(),
        'LQD': 120 + noise(),
        'DX-Y.NYB': 100 + noise(),
        '^TNX': 3 + noise(),
        '^FVX': 2.8 + noise(),
    })
    score, regime, conf = compute_liquidity_score(df)
    assert isinstance(regime, str)
    assert isinstance(conf, float)
    assert not np.isnan(conf)
