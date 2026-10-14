# -*- coding: utf-8 -*-
"""
state_machine.py -- State Machine centralizada para SLPM v1.2
Unica fuente de verdad para la clasificacion de liderazgo.
LIS queda excluido de la decision: Breadth es el unico factor de lideres.
"""

import math

THRESHOLDS = {
    'structural_min_confirmed': 0.20,
    'structural_min_emerging': 0.20,
    'structural_max_decay': -0.20,
    'structural_max_lost': -0.40,
    'breadth_max_decay': 0.35,
    'persistence_min_confirmed': 0.50,
    'persistence_max_emerging': 0.50,
}

def classify_leadership_state(structural_score, leader_breadth, persistence, coverage=1.0):
    """Clasifica el estado de liderazgo del sector #1 usando la State Machine.
    Estados: CONFIRMED, EMERGING, TACTICAL_CORRECTION, STRUCTURAL_DECAY,
    LOST, UNRESOLVED.
    Umbrales: S>0.20, Breadth>0.35, Persistence>0.50. Tactical no es decisorio.

    N1 ronda 2 (2026-10-14, dictamen auditor C2):
      - Guard defensivo: si leader_breadth no es finito, devuelve
        UNRESOLVED con reason_code=INSUFFICIENT_DATA_BREADTH.
        NO infiere la causa (puede venir de RS, momentum, flow o
        wyckoff); solo declara dato no disponible.
      - reason_code en todos los returns para distinguir datos
        insuficientes de senales mixtas.
    """
    t = THRESHOLDS

    # Guard defensivo: no depender de que NaN > x sea False.
    try:
        _breadth_finite = math.isfinite(float(leader_breadth))
    except (TypeError, ValueError):
        _breadth_finite = False
    if not _breadth_finite:
        return {
            'state': 'UNRESOLVED',
            'reason': 'Breadth no disponible (dato no finito).',
            'reason_code': 'INSUFFICIENT_DATA_BREADTH',
            'data_quality': 'LOW',
        }

    if coverage < 0.30:
        return {
            'state': 'UNRESOLVED',
            'reason': 'Cobertura de lideres insuficiente (<30%).',
            'reason_code': 'INSUFFICIENT_COVERAGE',
            'data_quality': 'LOW',
        }

    if (structural_score <= t['structural_max_lost'] and
        leader_breadth <= t['breadth_max_decay']):
        return {
            'state': 'LOST',
            'reason': 'Deterioro estructural extremo con baja amplitud de lideres.',
            'reason_code': 'STRUCTURAL_LOST',
            'data_quality': 'HIGH' if coverage >= 0.6 else 'MEDIUM',
        }

    if (structural_score <= t['structural_max_decay'] and
        leader_breadth <= t['breadth_max_decay']):
        return {
            'state': 'STRUCTURAL_DECAY',
            'reason': 'Deterioro estructural con baja amplitud de lideres.',
            'reason_code': 'STRUCTURAL_DECAY',
            'data_quality': 'HIGH' if coverage >= 0.6 else 'MEDIUM',
        }

    if (structural_score > t['structural_min_confirmed'] and
        persistence >= t['persistence_min_confirmed'] and
        leader_breadth > 0.35):
        return {
            'state': 'CONFIRMED',
            'reason': 'Liderazgo estructural confirmado con alta persistencia.',
            'reason_code': 'CONFIRMED',
            'data_quality': 'HIGH' if coverage >= 0.6 else 'MEDIUM',
        }

    if (structural_score > t['structural_min_emerging'] and
        persistence < t['persistence_max_emerging'] and
        leader_breadth > 0.35):
        return {
            'state': 'EMERGING',
            'reason': 'Estructura emergiendo pero sin historial suficiente.',
            'reason_code': 'EMERGING',
            'data_quality': 'HIGH' if coverage >= 0.6 else 'MEDIUM',
        }

    return {
        'state': 'UNRESOLVED',
        'reason': 'Ninguna condicion de estado se cumple. Senhales mixtas o insuficientes.',
        'reason_code': 'MIXED_OR_INSUFFICIENT_SIGNALS',
        'data_quality': 'MEDIUM' if coverage >= 0.3 else 'LOW',
    }

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

def validate_state(state, structural_score, leader_breadth, persistence):
    errors = []
    t = THRESHOLDS

    if state == 'CONFIRMED':
        if not (structural_score > t['structural_min_confirmed']):
            errors.append(f"CONFIRMED requiere Structural > {t['structural_min_confirmed']}, pero es {structural_score:.2f}")
        if not (persistence >= t['persistence_min_confirmed']):
            errors.append(f"CONFIRMED requiere Persistence >= {t['persistence_min_confirmed']:.0%}, pero es {persistence:.0%}")
        if not (leader_breadth > 0.35):
            errors.append(f"CONFIRMED requiere Breadth > 0.35, pero es {leader_breadth:.2f}")

    elif state == 'STRUCTURAL_DECAY':
        if not (structural_score <= t['structural_max_decay']):
            errors.append(f"STRUCTURAL_DECAY requiere Structural <= {t['structural_max_decay']}, pero es {structural_score:.2f}")
        if not (leader_breadth <= t['breadth_max_decay']):
            errors.append(f"STRUCTURAL_DECAY requiere Breadth <= {t['breadth_max_decay']}, pero es {leader_breadth:.2f}")

    return errors
