with open('regimes/macro_regime.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = """    # --- Volatilidad ---
    if vol_regime_score is not None:
        market_signals['volatility'] = -vol_regime_score
    else:
        try:
            vix = get_col(df_market, '^VIX', 'Close')
            market_signals['volatility'] = -tanh_normalize(vix)
        except KeyError:
            pass"""

new_block = """    # --- Volatilidad ---
    try:
        vix = get_col(df_market, '^VIX', 'Close')
        vix3m = get_col(df_market, '^VIX3M', 'Close')
        vix_level = tanh_normalize(vix)
        vix_term = tanh_normalize(vix3m - vix)  # estructura temporal
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
                pass"""

content = content.replace(old_block, new_block)
with open('regimes/macro_regime.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Volatilidad actualizada.')
