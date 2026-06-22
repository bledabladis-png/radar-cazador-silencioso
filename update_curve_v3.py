with open('regimes/macro_regime.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = """        tnx = get_col(df_market, '^TNX', 'Close')
        fvx = get_col(df_market, '^FVX', 'Close')
        tyx = get_col(df_market, '^TYX', 'Close')
        spread = tnx - fvx
        # Nivel
        level = (tnx + fvx + tyx) / 3
        curve_level = tanh_normalize(level)
        # Pendiente (cambio en el spread)
        curve_slope = tanh_normalize(spread.diff(20))
        # Velocidad (cambio en la pendiente)
        curve_velocity = tanh_normalize(spread.diff(20).diff(20))
        # Combinación
        market_signals['curve'] = 0.33 * curve_level + 0.33 * curve_slope + 0.34 * curve_velocity"""

new_block = """        bil = get_col(df_market, 'BIL', 'Close')
        ief = get_col(df_market, 'IEF', 'Close')
        tlt = get_col(df_market, 'TLT', 'Close')
        # Nivel (media de los tres ETFs)
        level = (bil + ief + tlt) / 3
        curve_level = tanh_normalize(level)
        # Pendiente (ratio IEF/BIL normalizado)
        spread = ief / bil
        curve_slope = tanh_normalize(spread.diff(20))
        # Velocidad
        curve_velocity = tanh_normalize(spread.diff(20).diff(20))
        # Combinación
        market_signals['curve'] = 0.33 * curve_level + 0.33 * curve_slope + 0.34 * curve_velocity"""

content = content.replace(old_block, new_block)
with open('regimes/macro_regime.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Curva actualizada a ETFs (BIL, IEF, TLT).')
