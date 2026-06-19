import pandas as pd
import numpy as np
from src.utils import robust_zscore, get_col

def compute_liquidity_score(df):
    scores = pd.DataFrame(index=df.index)
    
    # VIX
    try:
        vix = get_col(df, '^VIX', 'Close')
        scores['vix'] = -np.tanh(robust_zscore(vix, 60))
    except KeyError:
        pass
    
    # Credito (HYG/LQD)
    try:
        hyg = get_col(df, 'HYG', 'Close')
        lqd = get_col(df, 'LQD', 'Close')
        ratio = hyg / lqd
        scores['credit'] = -np.tanh(robust_zscore(ratio, 60))
    except KeyError:
        pass
    
    # Dolar
    try:
        dxy = get_col(df, 'DX-Y.NYB', 'Close')
        scores['dollar'] = -np.tanh(robust_zscore(dxy.pct_change(fill_method=None), 60))
    except KeyError:
        pass
    
    # Curva 10Y-2Y
    try:
        tnx = get_col(df, '^TNX', 'Close')
        fvx = get_col(df, '^FVX', 'Close')
        curve = tnx - fvx
        scores['curve'] = np.tanh(robust_zscore(curve, 120))
    except KeyError:
        pass
    
    weights = {'vix': 0.4, 'credit': 0.3, 'dollar': 0.2, 'curve': 0.1}
    available = [c for c in weights if c in scores.columns]
    w_sum = sum(weights[c] for c in available)
    if w_sum == 0:
        return pd.Series(0, index=df.index), 'NEUTRAL', 1.0
    
    liquidity_score = sum(scores[c] * weights[c] / w_sum for c in available)
    confidence = (1 - scores[available].std(axis=1).fillna(0) / 2).clip(0, 1)
    last = liquidity_score.iloc[-1] if not liquidity_score.empty else 0
    if last > 0.3:
        regime = 'ABUNDANTE'
    elif last > 0:
        regime = 'NEUTRAL'
    elif last > -0.3:
        regime = 'ESTRECHA'
    else:
        regime = 'CRISIS'
    
    return liquidity_score, regime, confidence.iloc[-1] if not confidence.empty else 0.5
