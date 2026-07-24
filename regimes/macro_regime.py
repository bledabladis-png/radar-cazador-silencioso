import pandas as pd
import numpy as np
from scores.macro_scores import compute_macro_signals, compute_macro_score
from config.weights import LEVEL_WEIGHTS

def compute_macro_regime(df_market, df_macro_manual, liquidity_score, vol_score):
    # Obtener senhales y score desde la capa de Scores
    all_signals = compute_macro_signals(df_market, df_macro_manual, liquidity_score, vol_score)
    macro_score = compute_macro_score(all_signals)

    last = macro_score.iloc[-1]
    last_volatility = all_signals['volatility'].iloc[-1]
    last_credit = all_signals['credit'].iloc[-1]
    last_market_strength = all_signals['market_strength'].iloc[-1]
    last_inflation = all_signals.get('inflation', pd.Series(0)).iloc[-1]
    last_liquidity = all_signals['liquidity'].iloc[-1]
    last_curve = all_signals['curve'].iloc[-1]

    # Reglas de precedencia
    if last_volatility < -2.0:
        regime = 'LIQUIDITY CRISIS'
    elif last_volatility < -1.5 and last_credit < -0.5:
        regime = 'LIQUIDITY CRISIS'
    elif last < -0.4 and last_market_strength < -0.5:
        regime = 'RECESSION'
    elif last_inflation > 0.3 and last_market_strength < 0:
        regime = 'INFLATION SHOCK'
    elif last < -0.2 and last_inflation > 0.3:
        regime = 'STAGFLATION'
    elif last > 0.2 and last_market_strength > 0.2 and last_inflation < 0 and last_volatility < 0:
        regime = 'GOLDILOCKS'
    elif last > 0.05 and last_market_strength > 0 and last_liquidity > -0.1 and last_volatility < 0 and last_curve > 0:
        regime = 'EXPANSION'
    elif last > 0.2 and last_inflation > 0.2:
        regime = 'LATE EXPANSION'
    elif last > 0.00 and last_market_strength > 0:
        regime = 'RECOVERY'
    elif last > 0 and last_inflation < -0.5:
        regime = 'DEFLATION'
    elif last < -0.2 and last_market_strength < 0:
        regime = 'SLOWDOWN'
    else:
        regime = 'MIXED'

    conf = 0.5
    return macro_score, regime, conf, all_signals
