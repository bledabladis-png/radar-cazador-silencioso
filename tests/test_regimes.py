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

    # C19: confidence = disagreement entre componentes via rango.
    # No imponemos un threshold arbitrario; verificamos que el valor
    # coincide exactamente con la formula C19 aplicada a los
    # componentes del ultimo dia.
    from src.utils import robust_zscore, get_col
    from config.settings import CONFIDENCE_RANGE_DIVISOR

    components = {}
    try:
        vix_s = get_col(df, '^VIX', 'Close')
        components['vix'] = -np.tanh(robust_zscore(vix_s, 60)).iloc[-1]
    except KeyError:
        pass
    try:
        hyg_s = get_col(df, 'HYG', 'Close')
        lqd_s = get_col(df, 'LQD', 'Close')
        components['credit'] = np.tanh(robust_zscore(hyg_s/lqd_s, 60)).iloc[-1]
    except KeyError:
        pass
    try:
        dxy_s = get_col(df, 'DX-Y.NYB', 'Close')
        components['dollar'] = -np.tanh(robust_zscore(dxy_s.pct_change(fill_method=None), 60)).iloc[-1]
    except KeyError:
        pass
    try:
        tnx_s = get_col(df, '^TNX', 'Close')
        fvx_s = get_col(df, '^FVX', 'Close')
        components['curve'] = np.tanh(robust_zscore(tnx_s-fvx_s, 60)).iloc[-1]
    except KeyError:
        pass

    vals = [v for v in components.values() if not np.isnan(v)]
    if len(vals) >= 2:
        rng = max(vals) - min(vals)
        expected = max(0.0, min(1.0, 1.0 - rng / CONFIDENCE_RANGE_DIVISOR))
        assert abs(conf - expected) < 1e-9, f'conf={conf} != expected={expected}'

    # Invariante
    assert 0.0 <= conf <= 1.0

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
