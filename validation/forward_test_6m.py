# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS, SECTOR_NAMES
from regimes.tactical_engine import compute_tactical_score
from regimes.structural_engine import compute_structural_score
from indicators.persistence import compute_persistence
from indicators.wyckoff import wyckoff_structure_core
from indicators.slpm_v12 import evaluate_slpm_v12
from indicators.stock_leader import generate_leader_section

# Configuración
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=180)
print(f"Forward Test: {START_DATE.date()} -> {END_DATE.date()}")

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
data = router.get_market_data(sectors, period='1y')

# Filtrar datos al período de interés
mask = (data.index >= pd.Timestamp(START_DATE)) & (data.index <= pd.Timestamp(END_DATE))
period_data = data.loc[mask]
fridays = period_data[period_data.index.dayofweek == 4].index  # solo viernes

print(f"Semanas a evaluar: {len(fridays)}")

registro = []
for fecha in fridays:
    # Tomar datos hasta esta fecha
    df_hasta = data.loc[:fecha]
    if len(df_hasta) < 200:
        continue

    # Scores táctico y estructural para cada sector
    tactical = {}
    structural = {}
    wyckoff_phases = {}
    for s in sectors:
        try:
            tactical[s] = compute_tactical_score(df_hasta, s)
            structural[s] = compute_structural_score(df_hasta, s)
            wyckoff_phases[s] = wyckoff_structure_core(df_hasta, s)
        except:
            tactical[s] = 0.0
            structural[s] = 0.0
            wyckoff_phases[s] = 'N/A'

    # Ranking sectorial simplificado
    scores = {}
    for s in sectors:
        scores[s] = 0.50*tactical[s] + 0.50*structural[s]
    ranking = pd.Series(scores).sort_values(ascending=False)
    top1_etf = ranking.index[0]
    top1_name = SECTOR_NAMES.get(top1_etf, top1_etf)
    top1_phase = wyckoff_phases.get(top1_etf, 'N/A')
    top3 = [(SECTOR_NAMES.get(s, s), f'{scores[s]:.2f}', wyckoff_phases.get(s, ''))
            for s in ranking.head(3).index]

    # Persistence del sector #1
    pers = compute_persistence(pd.Series(scores)) if len(scores) >= 12 else None

    # SLPM (simplificado: sin líderes individuales, solo con scores)
    sector_results = {'ranking': [(top1_etf, top1_name, scores[top1_etf], top1_phase)]}
    slpm = evaluate_slpm_v12(df_hasta, sector_results, [], 0.0,
                             tactical_scores=tactical, structural_scores=structural,
                             sector_persistence={top1_etf: pers})
    slpm_state = slpm.get('state', '?') if slpm else '?'

    registro.append({
        'fecha': fecha.strftime('%Y-%m-%d'),
        'sector_lider': top1_etf,
        'nombre_lider': top1_name,
        'score_lider': f'{scores[top1_etf]:.3f}',
        'fase_wyckoff': top1_phase,
        'slpm_estado': slpm_state,
        'top3': ' | '.join([f'{n} ({ph})' for n, sc, ph in top3])
    })

# Guardar CSV
df_reg = pd.DataFrame(registro)
df_reg.to_csv('outputs/audit/forward_test_6m.csv', index=False)
print(f"\nRegistro guardado: {len(df_reg)} semanas")

# Generar informe Markdown
with open('outputs/audit/forward_test_6m.md', 'w', encoding='utf-8') as f:
    f.write('# Forward Test - Evolución del Radar (6 meses)\n\n')
    f.write(f'Período: {START_DATE.date()} → {END_DATE.date()}\n\n')
    f.write('| Fecha | Sector Líder | Score | Fase Wyckoff | SLPM | Top 3 |\n')
    f.write('|-------|-------------|-------|--------------|------|-------|\n')
    for _, row in df_reg.iterrows():
        f.write(f"| {row['fecha']} | {row['nombre_lider']} ({row['sector_lider']}) | {row['score_lider']} | {row['fase_wyckoff']} | {row['slpm_estado']} | {row['top3']} |\n")
    f.write('\n## Notas del gestor\n\n')
    f.write('*Espacio para documentar eventos de mercado observados en cada fecha.*\n')

print("Informe Markdown generado: outputs/audit/forward_test_6m.md")
