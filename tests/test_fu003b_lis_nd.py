# tests/test_fu003b_lis_nd.py
"""Tests FU-003b (2026-09-15): LIS/Eff Breadth -> N/D con n=0.

Extiende FU-013 (SLPM 0% con n=0) a los campos LIS, Eff Breadth y Cf
de la linea compacta "Scores oficiales" y a la seccion
"Leader Integrity Score (LIS)".
"""
from src.report.slpm import render_slpm_v12


def _make_slpm(n_leaders=0, n_breadth=0, lis_val=0.0):
    return {
        'sector': 'Energy',
        'state': 'UNRESOLVED',
        'state_reason': 'test',
        'opportunity_quadrant': 'Transition',
        'leader_breadth_v2': {
            'n_used': n_breadth,
            'expected_leaders': 5,
            'coverage': 0.0,
            'coverage_warning': False,
            'rs_breadth': 0.0,
            'momentum_breadth': 0.0,
            'flow_breadth': 0.0,
            'wyckoff_breadth': 0.0,
            'composite': 0.0,
            'effective_composite': 0.0,
        },
        'input_scores': {
            'effective_breadth': 0.0,
            'persistence': 1.0,
            'tactical': 0.37,
            'structural': 0.15,
        },
        'flow_divergence_v2': {'composite': None},
        'leader_integrity': {
            'lis': lis_val,
            'n_leaders': n_leaders,
        },
    }


def test_fu003b_lis_section_n0_shows_nd():
    """Con n_leaders=0 -> 'LIS:** N/D (n=0)'."""
    lines = render_slpm_v12(_make_slpm(n_leaders=0, n_breadth=0))
    joined = ''.join(lines)
    assert "- **LIS:** N/D (n=0)" in joined
    assert "- **LIS:** +0.00 (n=0)" not in joined


def test_fu003b_scores_oficiales_n0_shows_nd():
    """Con n=0 -> 'LIS=N/D' y 'Eff Breadth=N/D' en la linea compacta."""
    lines = render_slpm_v12(_make_slpm(n_leaders=0, n_breadth=0))
    joined = ''.join(lines)
    assert "LIS=N/D" in joined
    assert "Eff Breadth=N/D" in joined
    assert "LIS=+0.00" not in joined
    assert "Eff Breadth=0.00" not in joined


def test_fu003b_lis_section_n5_numeric():
    """Con n_leaders=5 -> valor numerico (no N/D)."""
    lines = render_slpm_v12(_make_slpm(n_leaders=5, n_breadth=5, lis_val=0.42))
    joined = ''.join(lines)
    assert "- **LIS:** +0.42 (n=5)" in joined
    assert "LIS=+0.42" in joined


def test_fu003b_breadth_n0_but_lis_n5():
    """Caso mixto: LIS con datos, Breadth sin datos."""
    lines = render_slpm_v12(_make_slpm(n_leaders=5, n_breadth=0, lis_val=0.42))
    joined = ''.join(lines)
    assert "LIS=+0.42" in joined
    assert "Eff Breadth=N/D" in joined
