import pandas as pd
from config.weights import LEVEL_WEIGHTS, CRITICAL_WEIGHTS, IMPORTANT_WEIGHTS, CONTEXTUAL_WEIGHTS
from indicators.momentum import momentum_score, normalize_momentum
from indicators.credit import credit_risk_signal
from indicators.macro_fundamental import fundamental_signals
from src.utils import tanh_normalize, get_col

def compute_macro_signals(df_market, df_macro_manual=None, liquidity_score=None, vol_regime_score=None, real_liquidity_score=None):
    market_signals = {}

    # --- Crecimiento ---
    equity_tickers = ['^GSPC', '^NDX', '^RUT', '^STOXX50E', 'EEM', 'EWJ']
    returns = {}
    for t in equity_tickers:
        try:
            close = get_col(df_market, t, 'Close')
            returns[t] = close.pct_change(fill_method=None)
        except KeyError:
            pass
    if returns:
        growth_ret = pd.DataFrame(returns).mean(axis=1)
        market_signals['market_strength'] = normalize_momentum(momentum_score(growth_ret, 63))

    # --- Curva ---
    try:
        bil = get_col(df_market, 'BIL', 'Close')
        ief = get_col(df_market, 'IEF', 'Close')
        tlt = get_col(df_market, 'TLT', 'Close')
        level = (bil + ief + tlt) / 3
        curve_level = tanh_normalize(level)
        spread = ief / bil
        curve_slope = tanh_normalize(spread.diff(20))
        curve_velocity = tanh_normalize(spread.diff(20).diff(20))
        market_signals['curve'] = 0.33 * curve_level + 0.33 * curve_slope + 0.34 * curve_velocity
    except KeyError:
        pass

    # --- Crédito ---
    try:
        hyg = get_col(df_market, 'HYG', 'Close')
        lqd = get_col(df_market, 'LQD', 'Close')
        ief = get_col(df_market, 'IEF', 'Close')
        credit_spread = tanh_normalize(hyg / lqd)
        duration_spread = tanh_normalize(lqd / ief)
        market_signals['credit'] = 0.6 * credit_spread + 0.4 * duration_spread
    except KeyError:
        market_signals['credit'] = credit_risk_signal(df_market)

    # --- Volatilidad ---
    try:
        vix = get_col(df_market, '^VIX', 'Close')
        vix3m = get_col(df_market, '^VIX3M', 'Close')
        vix_level = tanh_normalize(vix)
        vix_term = tanh_normalize(vix3m - vix)
        vol_signal = 0.7 * vix_level + 0.3 * vix_term
        market_signals['volatility'] = -vol_signal
    except KeyError:
        if vol_regime_score is not None:
            market_signals['volatility'] = -vol_regime_score
        else:
            try:
                vix = get_col(df_market, '^VIX', 'Close')
                market_signals['volatility'] = -tanh_normalize(vix)
            except KeyError:
                pass

    # --- Liquidez (estrés financiero) ---
    if liquidity_score is not None:
        market_signals['liquidity'] = liquidity_score
    else:
        market_signals['liquidity'] = pd.Series(0, index=df_market.index)

    # --- Liquidez real (FRED) ---
    if real_liquidity_score is not None:
        market_signals['real_liquidity'] = real_liquidity_score.reindex(df_market.index).ffill()
    else:
        market_signals['real_liquidity'] = pd.Series(0, index=df_market.index)

    # --- Dólar ---
    try:
        dxy = get_col(df_market, 'DX-Y.NYB', 'Close')
        market_signals['dollar'] = -tanh_normalize(dxy.pct_change(fill_method=None).rolling(20).mean())
    except KeyError:
        pass

    # --- Materias primas ---
    industrial_tickers = ['^SPGSCI', 'HG=F', 'CL=F', 'NG=F']
    industrial_ret = []
    for t in industrial_tickers:
        try:
            close = get_col(df_market, t, 'Close')
            industrial_ret.append(close.pct_change(fill_method=None))
        except KeyError:
            pass
    if industrial_ret:
        industrial_avg = pd.concat(industrial_ret, axis=1).mean(axis=1)
        industrial_signal = normalize_momentum(momentum_score(industrial_avg, 63))
    else:
        industrial_signal = pd.Series(0, index=df_market.index)
    try:
        gold_close = get_col(df_market, 'GC=F', 'Close')
        gold_ret = gold_close.pct_change(fill_method=None)
        defensive_signal = normalize_momentum(momentum_score(gold_ret, 63))
    except KeyError:
        defensive_signal = pd.Series(0, index=df_market.index)
    market_signals['commodities'] = 0.5 * industrial_signal + 0.5 * defensive_signal

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

    return all_signals

def compute_macro_score(all_signals):
    def weighted_score(df, keys, weights):
        available = [k for k in keys if k in df.columns]
        if not available:
            return pd.Series(0, index=df.index)
        w = {k: weights[k] for k in available}
        w_sum = sum(w.values())
        # Rellenar NaN con 0 para evitar que una señal invalide todo el score
        return sum(df[k].fillna(0) * w[k] / w_sum for k in available)

    critical_score = weighted_score(all_signals, ['curve', 'credit', 'volatility', 'liquidity', 'real_liquidity'], CRITICAL_WEIGHTS)
    important_score = weighted_score(all_signals, ['dollar', 'commodities', 'breadth'], IMPORTANT_WEIGHTS)
    contextual_score = weighted_score(all_signals, ['market_strength'], CONTEXTUAL_WEIGHTS)

    macro_score = (
        LEVEL_WEIGHTS['critical'] * critical_score +
        LEVEL_WEIGHTS['important'] * important_score +
        LEVEL_WEIGHTS['contextual'] * contextual_score
    )

    # Mezcla con fundamentales si existen
    fundamental_sigs = [c for c in all_signals.columns if c in ['inflation', 'employment', 'activity']]
    if fundamental_sigs:
        fund_mean = all_signals[fundamental_sigs].mean(axis=1).fillna(0)
        macro_score = 0.5 * macro_score + 0.5 * fund_mean

    macro_score = macro_score.rolling(2, min_periods=1).mean()
    return macro_score

def compute_macro_regime(df_market, df_macro_manual, liquidity_score, vol_score):
    # Obtener señales y score desde las funciones internas
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
