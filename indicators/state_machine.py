# -*- coding: utf-8 -*-
"""
state_machine.py -- State Machine centralizada para SLPM v1.2
Unica fuente de verdad para la clasificacion de liderazgo.
"""
import numpy as np

THRESHOLDS = {
    'structural_min_confirmed': 0.20,
    'structural_min_emerging': 0.20,
    'structural_max_decay': -0.20,
    'structural_max_lost': -0.40,
    'tactical_max_correction': -0.20,
    'lis_min_confirmed': 0.30,
    'lis_max_decay': -0.20,
    'breadth_max_decay': 0.35,
    'persistence_min_confirmed': 0.50,
    'persistence_max_emerging': 0.50,
}

def classify_leadership_state(tactical_score, structural_score, lis, leader_breadth, persistence, coverage=1.0):
    t = THRESHOLDS
    
    # Regla de cobertura: si hay pocos líderes, forzar UNRESOLVED
    if coverage < 0.30:
        return {'state': 'UNRESOLVED', 'reason': 'Cobertura de líderes insuficiente (<30%).', 'data_quality': 'LOW'}
    if (structural_score <= t['structural_max_lost'] and
        lis <= t['lis_max_decay'] and
        leader_breadth <= t['breadth_max_decay']):
        return {'state': 'LOST', 'reason': 'Deterioro estructural extremo con baja integridad de lideres.', 'data_quality': 'HIGH' if coverage >= 0.6 else 'MEDIUM'}
    
    if (structural_score <= t['structural_max_decay'] and
        lis <= t['lis_max_decay'] and
        leader_breadth <= t['breadth_max_decay']):
        return {'state': 'STRUCTURAL_DECAY', 'reason': 'Deterioro estructural con baja integridad de lideres.', 'data_quality': 'HIGH' if coverage >= 0.6 else 'MEDIUM'}
    
    if (structural_score > t['structural_min_confirmed'] and
        tactical_score < t['tactical_max_correction'] and
        lis > t['lis_min_confirmed']):
        return {'state': 'TACTICAL_CORRECTION', 'reason': 'Estructura solida con debilidad tactica temporal.', 'data_quality': 'HIGH' if coverage >= 0.6 else 'MEDIUM'}
    
    if (structural_score > t['structural_min_confirmed'] and
        persistence >= t['persistence_min_confirmed'] and
        lis > t['lis_min_confirmed']):
        return {'state': 'CONFIRMED', 'reason': 'Liderazgo estructural confirmado con alta persistencia.', 'data_quality': 'HIGH' if coverage >= 0.6 else 'MEDIUM'}
    
    if (structural_score > t['structural_min_emerging'] and
        persistence < t['persistence_max_emerging'] and
        lis > 0.10):
        return {'state': 'EMERGING', 'reason': 'Estructura emergiendo pero sin historial suficiente.', 'data_quality': 'HIGH' if coverage >= 0.6 else 'MEDIUM'}
    
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

def validate_state(state, tactical_score, structural_score, lis, leader_breadth, persistence):
    errors = []
    t = THRESHOLDS
    
    if state == 'TACTICAL_CORRECTION':
        if not (structural_score > t['structural_min_confirmed']):
            errors.append(f"TACTICAL_CORRECTION requiere Structural > {t['structural_min_confirmed']}, pero es {structural_score:.2f}")
        if not (tactical_score < t['tactical_max_correction']):
            errors.append(f"TACTICAL_CORRECTION requiere Tactical < {t['tactical_max_correction']}, pero es {tactical_score:.2f}")
        if not (lis > t['lis_min_confirmed']):
            errors.append(f"TACTICAL_CORRECTION requiere LIS > {t['lis_min_confirmed']}, pero es {lis:.2f}")
    
    elif state == 'CONFIRMED':
        if not (structural_score > t['structural_min_confirmed']):
            errors.append(f"CONFIRMED requiere Structural > {t['structural_min_confirmed']}, pero es {structural_score:.2f}")
        if not (persistence >= t['persistence_min_confirmed']):
            errors.append(f"CONFIRMED requiere Persistence >= {t['persistence_min_confirmed']:.0%}, pero es {persistence:.0%}")
        if not (lis > t['lis_min_confirmed']):
            errors.append(f"CONFIRMED requiere LIS > {t['lis_min_confirmed']}, pero es {lis:.2f}")
    
    elif state == 'STRUCTURAL_DECAY':
        if not (structural_score <= t['structural_max_decay']):
            errors.append(f"STRUCTURAL_DECAY requiere Structural <= {t['structural_max_decay']}, pero es {structural_score:.2f}")
        if not (lis <= t['lis_max_decay']):
            errors.append(f"STRUCTURAL_DECAY requiere LIS <= {t['lis_max_decay']}, pero es {lis:.2f}")
        if not (leader_breadth <= t['breadth_max_decay']):
            errors.append(f"STRUCTURAL_DECAY requiere Breadth <= {t['breadth_max_decay']}, pero es {leader_breadth:.2f}")
    
    return errors

