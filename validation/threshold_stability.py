# -*- coding: utf-8 -*-
# validation/threshold_stability.py
# Fase 2: Sensibilidad de los umbrales Wyckoff (MARKUP >0.30, DISTRIBUTION <-0.30)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
import numpy as np
from data.providers.router import DataRouter
from config.tickers import MARKET_TICKERS
from indicators.wyckoff import wyckoff_score

router = DataRouter()
sectors = MARKET_TICKERS['sectors']
data = router.get_market_data(sectors, period='3y')

print('=== SENSIBILIDAD DE UMBRALES WYCKOFF ===')
print('Se perturban los umbrales ±0.05 y se mide el % de cambios de fase.\n')

thresholds_original = {'MARKUP': 0.30, 'DISTRIBUTION': -0.30}
perturbations = [0.05, -0.05]

for pert in perturbations:
    changes_per_sector = []
    total_changes = 0
    total_obs = 0
    for sector in sectors:
        scores = []
        for fecha in data.index[::5]:
            df_hasta = data.loc[:fecha]
            if len(df_hasta) < 200:
                continue
            try:
                score = wyckoff_score(df_hasta, sector)[0].iloc[-1]
                scores.append((fecha, score))
            except:
                continue
        if len(scores) < 30:
            continue
        df_sec = pd.DataFrame(scores, columns=['fecha','score'])
        # Clasificación original
        def classify_orig(s):
            if s > 0.30: return 'MARKUP'
            elif s > 0: return 'ACCUMULATION'
            elif s > -0.30: return 'RANGE'
            else: return 'DISTRIBUTION'
        # Clasificación perturbada
        def classify_pert(s, delta):
            if s > 0.30 + delta: return 'MARKUP'
            elif s > 0: return 'ACCUMULATION'
            elif s > -0.30 + delta: return 'RANGE'
            else: return 'DISTRIBUTION'
        orig = df_sec['score'].apply(classify_orig)
        pert_cls = df_sec['score'].apply(lambda x: classify_pert(x, pert))
        changes = (orig != pert_cls).sum()
        total_changes += changes
        total_obs += len(df_sec)
        changes_per_sector.append((sector, changes, len(df_sec)))
    
    pct = total_changes / total_obs * 100 if total_obs > 0 else 0
    print(f'Perturbación {pert:+.2f}: {total_changes} cambios de {total_obs} observaciones ({pct:.2f}%)')
    for sec, ch, n in changes_per_sector[:3]:  # primeros 3 sectores
        print(f'  {sec}: {ch}/{n}')
    print()

# Guardar resumen
with open('outputs/audit/threshold_stability_summary.md', 'w', encoding='utf-8') as f:
    f.write('# Sensibilidad de umbrales Wyckoff\n\n')
    for pert in perturbations:
        f.write(f'- Perturbación {pert:+.2f}: ver consola\n')
print('Resumen parcial guardado.')
