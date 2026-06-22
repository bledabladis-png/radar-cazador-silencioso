with open('regimes/macro_regime.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = """    commo_tickers = ['^SPGSCI', 'GC=F', 'HG=F', 'CL=F']
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
        market_signals['commodities'] = pd.Series(0, index=df_market.index)"""

new_block = """    # Commodities industriales (cobre, petróleo, gas natural)
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

    # Commodities defensivas (oro)
    try:
        gold_close = get_col(df_market, 'GC=F', 'Close')
        gold_ret = gold_close.pct_change(fill_method=None)
        defensive_signal = normalize_momentum(momentum_score(gold_ret, 63))
    except KeyError:
        defensive_signal = pd.Series(0, index=df_market.index)

    market_signals['commodities'] = 0.5 * industrial_signal + 0.5 * defensive_signal"""

content = content.replace(old_block, new_block)
with open('regimes/macro_regime.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Commodities separadas actualizadas.')
