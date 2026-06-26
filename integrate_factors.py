with open('scores/macro_scores.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Añadir la nueva función justo antes de compute_macro_score
new_func = '''
def compute_factor_momentum(df_market):
    """Señal de rotación de factores (Value, Momentum, Quality)."""
    try:
        vlue = get_col(df_market, 'VLUE', 'Close')
        mtum = get_col(df_market, 'MTUM', 'Close')
        qual = get_col(df_market, 'QUAL', 'Close')
        # Retornos 63d / volatilidad (mismo método que market_strength)
        vlue_mom = momentum_score(vlue.pct_change(fill_method=None), 63)
        mtum_mom = momentum_score(mtum.pct_change(fill_method=None), 63)
        qual_mom = momentum_score(qual.pct_change(fill_method=None), 63)
        # Puntuaciones robustas normalizadas
        vlue_score = tanh_normalize(vlue_mom)
        mtum_score = tanh_normalize(mtum_mom)
        qual_score = tanh_normalize(qual_mom)
        # Combinación con pesos fijos
        return 0.33 * vlue_score + 0.33 * mtum_score + 0.34 * qual_score
    except KeyError:
        return pd.Series(0, index=df_market.index)
'''

# Insertar la función antes de compute_macro_score
old = 'def compute_macro_score'
content = content.replace(old, new_func + '\n' + old)

# Añadir factor_momentum a la lista de señales importantes en compute_macro_score
old = "important_score = weighted_score(all_signals, ['dollar', 'commodities', 'breadth'], IMPORTANT_WEIGHTS)"
new = "important_score = weighted_score(all_signals, ['dollar', 'commodities', 'breadth', 'factor_momentum'], IMPORTANT_WEIGHTS)"
content = content.replace(old, new)

# Añadir factor_momentum al DataFrame de señales (dentro de compute_macro_signals)
old = "market_signals['breadth'] = 0.4 * tanh_normalize(breadth_20) + \\"
insert = "    market_signals['factor_momentum'] = compute_factor_momentum(df_market)\n    "
content = content.replace(old, insert + old)

with open('scores/macro_scores.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Factor momentum integrado.')
