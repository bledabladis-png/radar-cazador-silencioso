# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import pandas as pd
from indicators.state_machine import classify_leadership_state

hist = pd.read_csv('outputs/slpm_history.csv')

NEUTRAL_TACTICAL = 0.0
NEUTRAL_STRUCTURAL = 0.0
NEUTRAL_LIS = 0.0
NEUTRAL_BREADTH = 0.5
NEUTRAL_PERSISTENCE = 0.5

results = {}
# Componentes que existen en el CSV y podemos anular
components_in_csv = {
    'tactical': 'tactical_score',
    'structural': 'structural_score',
    'breadth': 'leader_breadth'
}

for comp, col in components_in_csv.items():
    altered = hist.copy()
    # Anular el componente
    if comp == 'tactical':
        altered[col] = NEUTRAL_TACTICAL
    elif comp == 'structural':
        altered[col] = NEUTRAL_STRUCTURAL
    elif comp == 'breadth':
        altered[col] = NEUTRAL_BREADTH

    new_states = []
    for _, row in altered.iterrows():
        state = classify_leadership_state(
            row['tactical_score'], row['structural_score'],
            NEUTRAL_LIS, row['leader_breadth'], NEUTRAL_PERSISTENCE, coverage=1.0
        )['state']
        new_states.append(state)
    altered['new_state'] = new_states
    changes = (altered['state'] != altered['new_state']).mean()
    results[comp] = changes

print("Ablacion del SLPM (% de cambios de estado al eliminar componente):")
for comp, pct in results.items():
    print(f"  Sin {comp:12s}: {pct:.1%}")
