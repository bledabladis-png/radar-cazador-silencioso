import pandas as pd
import numpy as np
import os
from scores.macro_scores import compute_macro_signals, compute_macro_score

def compute_macro_regime(df_market, df_macro_manual=None, liquidity_score=None, vol_regime_score=None, previous_regime=None, real_liquidity_score=None):
    # Obtener señales y score desde la capa de Scores
    all_signals = compute_macro_signals(df_market, df_macro_manual, liquidity_score, vol_regime_score, real_liquidity_score)
    macro_score = compute_macro_score(all_signals)

    # --- Confianza ---
    available_cols = [c for c in all_signals.columns if not all_signals[c].isna().all()]
    if available_cols:
        consistency = 1 - all_signals[available_cols].std(axis=1).fillna(0)
    else:
        consistency = pd.Series(0.5, index=all_signals.index)

    max_signals = 11
    coverage = len(available_cols) / max_signals
    confidence = (consistency * coverage).clip(0, 1)

    # --- Clasificación del régimen ---
    last = macro_score.iloc[-1]
    last_market_strength = all_signals['market_strength'].iloc[-1] if 'market_strength' in all_signals.columns else 0
    last_inflation = all_signals['inflation'].iloc[-1] if 'inflation' in all_signals.columns else 0
    last_liquidity = all_signals['liquidity'].iloc[-1] if 'liquidity' in all_signals.columns else 0
    last_volatility = all_signals['volatility'].iloc[-1] if 'volatility' in all_signals.columns else 0
    last_curve = all_signals['curve'].iloc[-1] if 'curve' in all_signals.columns else 0
    last_credit = all_signals['credit'].iloc[-1] if 'credit' in all_signals.columns else 0

    # Percentiles históricos
    if 'volatility' in all_signals.columns:
        vol_hist = all_signals['volatility'].dropna()
        vol_pct = (vol_hist < last_volatility).mean() if len(vol_hist) > 500 else 0.5
    else:
        vol_pct = 0.5
    if 'inflation' in all_signals.columns:
        inf_hist = all_signals['inflation'].dropna()
        inf_pct = (inf_hist < last_inflation).mean() if len(inf_hist) > 500 else 0.5
    else:
        inf_pct = 0.5

    regime = 'MIXED'
    if last_volatility < -2.0 or (last_volatility < -1.5 and vol_pct < 0.05):
        regime = 'LIQUIDITY CRISIS'
    if last_volatility < -2.0 and last < 0.1:
        regime = 'LIQUIDITY CRISIS'
    elif last_volatility < -1.5 and last_credit < -0.5:
        regime = 'LIQUIDITY CRISIS'
    elif last < -0.4 and last_market_strength < -0.5:
        regime = 'RECESSION'
    elif last_inflation > 0.3 and last_market_strength < 0:
        regime = 'INFLATION SHOCK'
    elif last < -0.2 and last_inflation > 0.3:
        regime = 'STAGFLATION'
    elif last > 0.05 and last_market_strength > 0 and last_liquidity > -0.1 and last_volatility < 0 and last_curve > 0:
        regime = 'EXPANSION'
    elif last > 0.2 and last_inflation > 0.2:
        regime = 'LATE EXPANSION'
    elif last > 0.2 and last_market_strength > 0.2 and last_inflation < 0 and last_liquidity > 0 and last_volatility < 0:
        regime = 'GOLDILOCKS'
    elif last < -0.2 and last_market_strength < 0:
        regime = 'SLOWDOWN'
    elif last > 0.00 and last_market_strength > 0:
        regime = 'RECOVERY'
    elif last > 0 and last_inflation < -0.5:
        regime = 'DEFLATION'

    conf = confidence.iloc[-1] if not confidence.empty else 0.5
    return macro_score, regime, conf, all_signals
