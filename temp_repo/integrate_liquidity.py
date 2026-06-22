with open('run.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if 'print(f"  Liquidez: {liquidity_regime} (conf: {liq_conf:.0%})")' in line:
        new_lines.append('    print("Calculando liquidez real (FRED)...")\n')
        new_lines.append('    real_liq_score, real_liq_regime, real_liq_conf = compute_real_liquidity()\n')
        new_lines.append('    if real_liq_score is not None:\n')
        new_lines.append('        print(f"  Liquidez real: {real_liq_regime} (conf: {real_liq_conf:.0%})")\n')
        new_lines.append('    else:\n')
        new_lines.append('        print("  Liquidez real: no disponible (sin datos FRED)")\n')
        new_lines.append('        real_liq_score = None\n')
        new_lines.append('        real_liq_regime = "N/A"\n')
        new_lines.append('        real_liq_conf = 0.0\n')

with open('run.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Liquidez real integrada en run.py.')
