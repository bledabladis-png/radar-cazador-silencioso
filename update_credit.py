with open('regimes/macro_regime.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = """    # --- Crédito ---
    market_signals['credit'] = credit_spread_signal(df_market)"""

new_block = """    # --- Crédito ---
    try:
        hyg = get_col(df_market, 'HYG', 'Close')
        lqd = get_col(df_market, 'LQD', 'Close')
        ief = get_col(df_market, '^IEF', 'Close') if '^IEF' in [c[1] for c in df_market.columns] else None
        credit_spread = tanh_normalize(hyg / lqd)
        if ief is not None:
            duration_spread = tanh_normalize(lqd / ief)
            market_signals['credit'] = 0.6 * credit_spread + 0.4 * duration_spread
        else:
            market_signals['credit'] = credit_spread
    except KeyError:
        market_signals['credit'] = credit_spread_signal(df_market)"""

content = content.replace(old_block, new_block)
with open('regimes/macro_regime.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Crédito actualizado.')
