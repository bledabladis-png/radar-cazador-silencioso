# -*- coding: utf-8 -*-
"""
Ablacion de la State Machine actual del SLPM.
Compara BASE vs variantes con un componente neutralizado.
No compara contra estados historicos, porque la logica cambio.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import pandas as pd
from indicators.state_machine import classify_leadership_state

hist = pd.read_csv('outputs/history/slpm_history.csv')

# Valores neutros
NEUTRAL_TACTICAL = 0.0
NEUTRAL_STRUCTURAL = 0.0
NEUTRAL_BREADTH = 0.5
NEUTRAL_PERSISTENCE = 0.5
COVERAGE = 1.0

components = {
    'tactical': ('tactical_score', NEUTRAL_TACTICAL),
    'structural': ('structural_score', NEUTRAL_STRUCTURAL),
    'breadth': ('leader_breadth', NEUTRAL_BREADTH),
    'persistence': ('_persistence_', NEUTRAL_PERSISTENCE),
}

# Generar BASE actual
base_states = []
for _, row in hist.iterrows():
    state = classify_leadership_state(
        row['tactical_score'],
        row['structural_score'],
        row['leader_breadth'],
        NEUTRAL_PERSISTENCE,
        coverage=COVERAGE
    )['state']
    base_states.append(state)

results = []

for comp_name, (col, neutral) in components.items():
    changed_states = []
    transitions = []
    for i, (_, row) in enumerate(hist.iterrows()):
        if comp_name == 'persistence':
            # persistence no está en el CSV, se usa neutro
            t = row['tactical_score']
            s = row['structural_score']
            b = row['leader_breadth']
            p = neutral
        else:
            t = row['tactical_score'] if col != 'tactical_score' else neutral
            s = row['structural_score'] if col != 'structural_score' else neutral
            b = row['leader_breadth'] if col != 'leader_breadth' else neutral
            p = NEUTRAL_PERSISTENCE

        ablation_state = classify_leadership_state(t, s, b, p, coverage=COVERAGE)['state']
        base_state = base_states[i]
        changed = base_state != ablation_state
        changed_states.append(changed)
        if changed:
            transitions.append((base_state, ablation_state))

    change_rate = sum(changed_states) / len(changed_states)
    results.append({
        'component': comp_name,
        'changes': int(sum(changed_states)),
        'total': len(changed_states),
        'change_rate': change_rate,
        'dominant_transitions': pd.Series(transitions).value_counts().head(5).to_dict() if transitions else {}
    })

print("\nAblacion de la State Machine actual (BASE vs variante):\n")
for r in results:
    print(f"  Sin {r['component']:12s}: {r['change_rate']:.1%} ({r['changes']}/{r['total']})")
    if r['dominant_transitions']:
        print(f"    Transiciones dominantes:")
        for tr, cnt in r['dominant_transitions'].items():
            print(f"      {tr[0]} -> {tr[1]}  ({cnt})")
    print()
