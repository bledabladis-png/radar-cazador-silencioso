# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
from datetime import datetime, timedelta
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS, SECTOR_NAMES
from regimes.tactical_engine import compute_tactical_score
from regimes.structural_engine import compute_structural_score
from indicators.persistence import compute_persistence
from indicators.wyckoff import wyckoff_structure_core
from indicators.slpm_v12 import evaluate_slpm_v12

# Fecha de ejecución (viernes actual o último disponible)
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=7)  # última semana

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
data = router.get_market_data(sectors + ['^GSPC'], period='1y')

# Último viernes con datos
mask = (data.index >= pd.Timestamp(START_DATE)) & (data.index <= pd.Timestamp(END_DATE))
period_data = data.loc[mask]
fridays = period_data[period_data.index.dayofweek == 4].index
if len(fridays) == 0:
    print("No hay viernes en la última semana. Abortando.")
    sys.exit(0)
fecha = fridays[-1]

df_hasta = data.loc[:fecha]
if len(df_hasta) < 200:
    print("Datos insuficientes (<200 sesiones). Abortando.")
    sys.exit(0)

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

persistence = {}
for s in sectors:
    try:
        p = compute_persistence(pd.Series(tactical[s]))
        persistence[s] = p if p is not None else 0.5
    except:
        persistence[s] = 0.5

scores = {s: 0.50*tactical[s] + 0.50*structural[s] for s in sectors}
ranking = pd.Series(scores).sort_values(ascending=False)
top1_etf = ranking.index[0]
top1_name = SECTOR_NAMES.get(top1_etf, top1_etf)
top1_phase = wyckoff_phases.get(top1_etf, 'N/A')
top3 = [(SECTOR_NAMES.get(s, s), f'{scores[s]:.2f}', wyckoff_phases.get(s, '')) for s in ranking.head(3).index]

sector_results = {'ranking': [(top1_etf, top1_name, scores[top1_etf], top1_phase)]}
slpm = evaluate_slpm_v12(df_hasta, sector_results, [], 0.0,
                         tactical_scores=tactical, structural_scores=structural,
                         sector_persistence=persistence)
slpm_state = slpm.get('state', '?') if slpm else '?'

nuevo_registro = {
    'fecha': fecha.strftime('%Y-%m-%d'),
    'sector_lider': top1_etf,
    'nombre_lider': top1_name,
    'score_lider': f'{scores[top1_etf]:.3f}',
    'fase_wyckoff': top1_phase,
    'slpm_estado': slpm_state,
    'top3': ' | '.join([f'{n} ({ph})' for n, sc, ph in top3])
}

# Cargar histórico existente o crear nuevo
hist_path = 'outputs/history/forward_test_historico.csv'
if os.path.exists(hist_path):
    historico = pd.read_csv(hist_path)
    # Evitar duplicados
    if fecha.strftime('%Y-%m-%d') in historico['fecha'].values:
        print(f"Fecha {fecha.date()} ya existe en el histórico. Abortando.")
        sys.exit(0)
    historico = pd.concat([historico, pd.DataFrame([nuevo_registro])], ignore_index=True)
else:
    historico = pd.DataFrame([nuevo_registro])

historico.to_csv(hist_path, index=False)
print(f"Registro añadido: {fecha.date()}")

# Regenerar informe Markdown completo
with open('outputs/audit/forward_test_6m.md', 'w', encoding='utf-8') as f:
    f.write('# Forward Test - Evolución del Radar (6 meses)\n\n')
    if len(historico) > 0:
        f.write(f'Período: {historico.iloc[0]["fecha"]} → {historico.iloc[-1]["fecha"]}\n\n')
    f.write('| Fecha | Sector Líder | Score | Fase Wyckoff | SLPM | Top 3 |\n')
    f.write('|-------|-------------|-------|--------------|------|-------|\n')
    for _, row in historico.iterrows():
        f.write(f'| {row["fecha"]} | {row["nombre_lider"]} ({row["sector_lider"]}) | {row["score_lider"]} | {row["fase_wyckoff"]} | {row["slpm_estado"]} | {row["top3"]} |\n')
    f.write('\n## Notas del gestor\n\n')
    f.write('*Espacio para documentar eventos de mercado observados en cada fecha.*\n')

print("Informe Markdown regenerado con todas las semanas.")
