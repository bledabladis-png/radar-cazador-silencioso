import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from regimes.financial_conditions import compute_financial_conditions

def make_df(tickers_dict):
    cols = []
    data = {}
    for t in tickers_dict:
        cols.append(('Close', t))
        data[('Close', t)] = tickers_dict[t]
    return pd.DataFrame(data, index=range(len(data[cols[0]])))

def test_liquidity_crisis():
    n = 120
    # Escenario de estrés realista que produce HIGH_STRESS en v4.3
    vix = [15 + i*(25/119) for i in range(n)]
    hyg = [80 - i*(20/119) for i in range(n)]
    lqd = [120]*n
    dxy = [100 + i*(10/119) for i in range(n)]
    tnx = [2 - i*(1/119) for i in range(n)]
    fvx = [3 - i*(0.5/119) for i in range(n)]
    df = make_df({
        '^VIX': vix,
        'HYG': hyg,
        'LQD': lqd,
        'DX-Y.NYB': dxy,
        '^TNX': tnx,
        '^FVX': fvx,
    })
    score, regime, conf = compute_financial_conditions(df)
    assert regime == 'HIGH_STRESS'
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
    score, regime, conf = compute_financial_conditions(df)
    assert isinstance(regime, str)
    assert isinstance(conf, float)
    assert not np.isnan(conf)
