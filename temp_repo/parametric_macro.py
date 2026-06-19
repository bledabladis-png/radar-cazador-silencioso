import pandas as pd
import numpy as np
import os
from config.weights import LEVEL_WEIGHTS, CRITICAL_WEIGHTS, IMPORTANT_WEIGHTS, CONTEXTUAL_WEIGHTS
from indicators.momentum import momentum_score, normalize_momentum
from indicators.credit import credit_spread_signal
from indicators.macro_fundamental import fundamental_signals
from src.utils import robust_zscore, tanh_normalize, get_col

def compute_macro_regime_parametric(df_market, df_macro_manual=None, liquidity_score=None, vol_regime_score=None, previous_regime=None, z_window=60, mix_weight=0.5):
    market_signals = {}

    # --- Crecimiento ---
    equity_tickers = ['^GSPC', '^NDX', '^RUT', '^STOXX50E', 'EEM']
    returns = {}
    for t in equity_tickers:
        try:
            close = get_col(df_market, t, 'Close')
            returns[t] = close.pct_change(fill_method=None)
        except KeyError:
            pass
    if returns:
        growth_ret = pd.DataFrame(returns).mean(axis=1)
        market_signals['growth'] = normalize_momentum(momentum_score(growth_ret, 63))

    # --- Curva ---
    try:
        tnx = get_col(df_market, '^TNX', 'Close')
        fvx = get_col(df_market, '^FVX', 'Close')
        tyx = get_col(df_market, '^TYX', 'Close')
        spread = tnx - fvx
        level = (tnx + fvx + tyx) / 3
        curve_level = tanh_normalize(level)
        curve_slope = tanh_normalize(spread.diff(20))
        curve_velocity = tanh_normalize(spread.diff(20).diff(20))
        market_signals['curve'] = 0.33 * curve_level + 0.33 * curve_slope + 0.34 * curve_velocity
    except KeyError:
        pass

    # --- Crédito ---
    try:
        hyg = get_col(df_market, 'HYG', 'Close')
        lqd = get_col(df_market, 'LQD', 'Close')
        ratio = hyg / lqd
        z = robust_zscore(ratio, window=z_window)
        market_signals['credit'] = np.tanh(z)
    except KeyError:
        market_signals['credit'] = pd.Series(0, index=df_market.index)

    # --- Volatilidad ---
    if vol_regime_score is not None:
        market_signals['volatility'] = -vol_regime_score
    else:
        try:
            vix = get_col(df_market, '^VIX', 'Close')
            market_signals['volatility'] = -tanh_normalize(vix)
        except KeyError:
            pass

    # --- Liquidez ---
    if liquidity_score is not None:
        market_signals['liquidity'] = liquidity_score
    else:
        market_signals['liquidity'] = pd.Series(0, index=df_market.index)

    # --- Dólar ---
    try:
        dxy = get_col(df_market, 'DX-Y.NYB', 'Close')
        market_signals['dollar'] = -tanh_normalize(dxy.pct_change(fill_method=None).rolling(20).mean())
    except KeyError:
        pass

    # --- Materias primas ---
    commo_tickers = ['^SPGSCI', 'GC=F', 'HG=F', 'CL=F']
    commo_ret = []
    for t in commo_tickers:
        try:
            close = get_col(df_market, t, 'Close')
            commo_ret.append(close.pct_change(fill_method=None))
        except KeyError:
            pass
    if commo_ret:
        commo_avg = pd.concat(commo_ret, axis=1).mean(axis=1)
        market_signals['commodities'] = normalize_momentum(momentum_score(commo_avg, 63))
    else:
        market_signals['commodities'] = pd.Series(0, index=df_market.index)

    # --- Breadth ---
    from indicators.breadth import compute_breadth
    breadth_20, breadth_50, breadth_200, nh, nl = compute_breadth(df_market)
    market_signals['breadth'] = 0.4 * tanh_normalize(breadth_20) + \
                                0.3 * tanh_normalize(breadth_50) + \
                                0.2 * tanh_normalize(breadth_200) + \
                                0.1 * tanh_normalize(nh - nl)

    # --- Señales fundamentales ---
    fundamental_sigs = None
    if df_macro_manual is not None and not df_macro_manual.empty:
        fundamental_sigs = fundamental_signals(df_macro_manual)

    all_signals = pd.DataFrame(market_signals)
    if fundamental_sigs is not None and not fundamental_sigs.empty:
        all_signals = all_signals.join(fundamental_sigs, how='left').ffill().bfill()

    # --- Pesos ---
    def weighted_score(df, keys, weights):
        available = [k for k in keys if k in df.columns]
        if not available:
            return pd.Series(0, index=df.index)
        w = {k: weights[k] for k in available}
        w_sum = sum(w.values())
        return sum(df[k] * w[k] / w_sum for k in available)

    critical_score = weighted_score(all_signals, ['curve', 'credit', 'volatility', 'liquidity'], CRITICAL_WEIGHTS)
    important_score = weighted_score(all_signals, ['dollar', 'commodities', 'breadth'], IMPORTANT_WEIGHTS)
    contextual_score = weighted_score(all_signals, ['growth'], CONTEXTUAL_WEIGHTS)

    macro_score = (
        LEVEL_WEIGHTS['critical'] * critical_score +
        LEVEL_WEIGHTS['important'] * important_score +
        LEVEL_WEIGHTS['contextual'] * contextual_score
    )

    if fundamental_sigs is not None and not fundamental_sigs.empty:
        fund_mean = fundamental_sigs.mean(axis=1).reindex(macro_score.index).ffill().bfill()
        macro_score = mix_weight * macro_score + (1 - mix_weight) * fund_mean

    macro_score = macro_score.rolling(2, min_periods=1).mean()

    # --- Confianza ---
    available_cols = [c for c in all_signals.columns if not all_signals[c].isna().all()]
    if available_cols:
        consistency = 1 - all_signals[available_cols].std(axis=1).fillna(0)
    else:
        consistency = pd.Series(0.5, index=all_signals.index)
    max_signals = 11
    coverage = len(available_cols) / max_signals
    confidence = (consistency * coverage).clip(0, 1)

    # --- Clasificación ---
    last = macro_score.iloc[-1]
    last_market_strength = all_signals['growth'].iloc[-1] if 'growth' in all_signals.columns else 0
    last_inflation = all_signals['inflation'].iloc[-1] if 'inflation' in all_signals.columns else 0
    last_liquidity = market_signals['liquidity'].iloc[-1] if 'liquidity' in market_signals else 0
    last_volatility = market_signals['volatility'].iloc[-1] if 'volatility' in market_signals else 0
    last_curve = market_signals['curve'].iloc[-1] if 'curve' in market_signals else 0
    last_credit = market_signals['credit'].iloc[-1] if 'credit' in market_signals else 0

    # Percentiles
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
    if last_volatility < -2.5 or (last_volatility < -1.5 and vol_pct < 0.05):
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
    elif last > 0.2 and last_market_strength > 0 and last_liquidity > -0.1 and last_volatility < 0 and last_curve > 0:
        regime = 'EXPANSION'
    elif last > 0.2 and last_inflation > 0.2:
        regime = 'LATE EXPANSION'
    elif last > 0.2 and last_market_strength > 0.2 and last_inflation < 0 and last_liquidity > 0 and last_volatility < 0:
        regime = 'GOLDILOCKS'
    elif last < -0.2 and last_market_strength < 0:
        regime = 'SLOWDOWN'
    elif last > 0 and last_market_strength > 0.1:
        regime = 'RECOVERY'
    elif last > 0 and last_inflation < -0.5:
        regime = 'DEFLATION'

    conf = confidence.iloc[-1] if not confidence.empty else 0.5
    return macro_score, regime, conf, all_signals
