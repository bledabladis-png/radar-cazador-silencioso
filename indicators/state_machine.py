# -*- coding: utf-8 -*-
"""
state_machine.py -- State Machine centralizada para SLPM v1.2
Unica fuente de verdad para la clasificacion de liderazgo.
LIS queda excluido de la decision: Breadth es el unico factor de lideres.
"""
import numpy as np

THRESHOLDS = {
    'structural_min_confirmed': 0.20,
    'structural_min_emerging': 0.20,
    'structural_max_decay': -0.20,
    'structural_max_lost': -0.40,
    'breadth_max_decay': 0.35,
}

def classify_leadership_state(structural_score, leader_breadth, coverage=1.0):
    """Clasifica el estado de liderazgo del sector #1 usando la State Machine.
    Estados: CONFIRMED, EMERGING, TACTICAL_CORRECTION, STRUCTURAL_DECAY,
    LOST, UNRESOLVED.
    Umbrales: S>0.20, Breadth>0.35. Tactical y Persistence no son decisorios."""
    t = THRESHOLDS

    if coverage < 0.30:
        return {'state': 'UNRESOLVED', 'reason': 'Cobertura de líderes insuficiente (<30%).', 'data_quality': 'LOW'}

    if (structural_score <= t['structural_max_lost'] and
        leader_breadth <= t['breadth_max_decay']):
        return {'state': 'LOST', 'reason': 'Deterioro estructural extremo con baja amplitud de lideres.', 'data_quality': 'HIGH' if coverage >= 0.6 else 'MEDIUM'}

    if (structural_score <= t['structural_max_decay'] and
        leader_breadth <= t['breadth_max_decay']):
        return {'state': 'STRUCTURAL_DECAY', 'reason': 'Deterioro estructural con baja amplitud de lideres.', 'data_quality': 'HIGH' if coverage >= 0.6 else 'MEDIUM'}

    return {'state': 'UNRESOLVED', 'reason': 'Ninguna condicion de estado se cumple. Senhales mixtas o insuficientes.', 'data_quality': 'MEDIUM' if coverage >= 0.3 else 'LOW'}

def get_opportunity_quadrant(state):
    mapping = {
        'CONFIRMED': 'Structural Strength',
        'EMERGING': 'Structural Strength',
        'TACTICAL_CORRECTION': 'Tactical Correction',
        'STRUCTURAL_DECAY': 'Structural Weakness',
        'LOST': 'Structural Weakness',
        'UNRESOLVED': 'Transition'
    }
    return mapping.get(state, 'Transition')

def validate_state(state, structural_score, leader_breadth):
    errors = []
    t = THRESHOLDS

    if state == 'CONFIRMED':
        if not (structural_score > t['structural_min_confirmed']):
            errors.append(f"CONFIRMED requiere Structural > {t['structural_min_confirmed']}, pero es {structural_score:.2f}")
        if not (leader_breadth > 0.35):
            errors.append(f"CONFIRMED requiere Breadth > 0.35, pero es {leader_breadth:.2f}")

    elif state == 'STRUCTURAL_DECAY':
        if not (structural_score <= t['structural_max_decay']):
            errors.append(f"STRUCTURAL_DECAY requiere Structural <= {t['structural_max_decay']}, pero es {structural_score:.2f}")
        if not (leader_breadth <= t['breadth_max_decay']):
            errors.append(f"STRUCTURAL_DECAY requiere Breadth <= {t['breadth_max_decay']}, pero es {leader_breadth:.2f}")

    return errors
