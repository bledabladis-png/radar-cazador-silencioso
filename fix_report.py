with open('src/report_generator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Añadir parámetros en la firma de la función
old = 'leader_lines=None, breadth_values=None,'
new = 'leader_lines=None, breadth_values=None, real_liquidity_regime=None, real_liquidity_conf=None,'
content = content.replace(old, new)

# Añadir línea de liquidez real en el reporte
old = 'lines.append(f"- **Cond. Financieras:** {liquidity_regime} (Score: {liquidity_score.iloc[-1]:.2f}, Confianza: {liq_conf:.0%})\\n")'
new = old + '\n    if real_liquidity_regime is not None:\n        lines.append(f"- **Liquidez Real (FRED):** {real_liquidity_regime} (Confianza: {real_liquidity_conf:.0%})\\n")'
content = content.replace(old, new)

with open('src/report_generator.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Liquidez real añadida a report_generator.py.')
