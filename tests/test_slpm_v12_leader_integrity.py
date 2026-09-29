# -*- coding: utf-8 -*-
"""Tests de indicators.slpm_v12.compute_leader_integrity.

Cubre: contrato de retorno, finitud, manejo de flow_proxy_z=0.0
(fix 2026-09-29).
"""
from __future__ import annotations

import math

from indicators.slpm_v12 import compute_leader_integrity

def test_compute_leader_integrity_lista_vacia():
    out = compute_leader_integrity([])
    assert out == {'lis': 0.0, 'n_leaders': 0}

def test_compute_leader_integrity_flow_proxy_z_cero_no_nan():
    """Fix 2026-09-29: m.get('flow_proxy_z') or np.nan convertia 0.0
    en NaN. Un flow z-score exactamente cero es valor legitimo (flujo
    neutro). El LIS debe seguir siendo finito."""
    metrics = [
        {'rs': 1.2, 'rs_momentum': 0.1, 'flow_proxy_z': 0.0,
         'wyckoff_phase': 'MARKUP'},
    ]
    out = compute_leader_integrity(metrics)
    assert out['n_leaders'] == 1
    assert math.isfinite(out['lis'])

def test_compute_leader_integrity_flow_proxy_z_none_si_nan():
    """flow_proxy_z=None -> NaN -> no contribuye (coherente con el
    comportamiento previo al fix)."""
    metrics = [
        {'rs': 1.2, 'rs_momentum': 0.1, 'flow_proxy_z': None,
         'wyckoff_phase': 'MARKUP'},
    ]
    out = compute_leader_integrity(metrics)
    assert out['n_leaders'] == 1
    assert math.isfinite(out['lis'])

def test_compute_leader_integrity_clip_entre_menos_uno_y_uno():
    """El retorno siempre va clipado a [-1, +1]."""
    metrics = [
        {'rs': 5.0, 'rs_momentum': 5.0, 'flow_proxy_z': 5.0,
         'wyckoff_phase': 'MARKUP'},
    ]
    out = compute_leader_integrity(metrics)
    assert -1.0 <= out['lis'] <= 1.0