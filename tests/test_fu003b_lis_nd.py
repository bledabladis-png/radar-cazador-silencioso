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
            'rs_breadth_status': 'OK' if n_breadth > 0 else 'NO_LEADERS',
            'leader_breadth_status': 'OK' if n_breadth > 0 else 'NO_LEADERS',
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


# ============================================================
# N1 ronda 2 (2026-10-14): render del state_reason_code
# ============================================================
def test_render_muestra_motivo_cuando_reason_code_presente():
    """Si state_reason_code != '' y != 'UNKNOWN', se renderiza."""
    slpm = {
        'sector': 'Energy',
        'state': 'UNRESOLVED',
        'state_reason': 'Breadth no disponible.',
        'state_reason_code': 'INSUFFICIENT_DATA_RS',
        'opportunity_quadrant': 'Transition',
        'leader_breadth_v2': {
            'n_used': 3, 'expected_leaders': 5, 'coverage': 0.6,
            'coverage_warning': False,
            'rs_breadth': float('nan'),
            'rs_breadth_status': 'INSUFFICIENT_RS_RATIO',
            'rs_breadth_missing': 2,
            'leader_breadth_status': 'INSUFFICIENT_RS_RATIO',
            'momentum_breadth': 0.8,
            'flow_breadth': 0.6,
            'wyckoff_breadth': 1.0,
            'composite': float('nan'),
            'effective_composite': float('nan'),
        },
        'input_scores': {'effective_breadth': 0.0, 'persistence': 0.5,
                         'tactical': 0.2, 'structural': 0.1},
        'flow_divergence_v2': {'composite': None},
        'leader_integrity': {'lis': 0.0, 'n_leaders': 3},
    }
    lines = render_slpm_v12(slpm)
    joined = chr(10).join(lines)
    assert 'Motivo: INSUFFICIENT_DATA_RS' in joined
    assert 'Breadth no disponible' in joined
    assert 'nan%' not in joined


def test_render_no_muestra_motivo_si_unknown():
    """state_reason_code='UNKNOWN' no se renderiza como motivo."""
    slpm = {
        'sector': 'Energy',
        'state': 'UNRESOLVED',
        'state_reason': 'test',
        'state_reason_code': 'UNKNOWN',
        'opportunity_quadrant': 'Transition',
        'leader_breadth_v2': {
            'n_used': 0, 'expected_leaders': 5, 'coverage': 0.0,
            'coverage_warning': True,
            'rs_breadth': float('nan'),
            'rs_breadth_status': 'NO_LEADERS',
            'leader_breadth_status': 'NO_LEADERS',
            'momentum_breadth': float('nan'),
            'flow_breadth': float('nan'),
            'wyckoff_breadth': float('nan'),
            'composite': float('nan'),
            'effective_composite': float('nan'),
        },
        'input_scores': {'effective_breadth': 0.0, 'persistence': 0.5,
                         'tactical': 0.0, 'structural': 0.0},
        'flow_divergence_v2': {'composite': None},
        'leader_integrity': {'lis': 0.0, 'n_leaders': 0},
    }
    lines = render_slpm_v12(slpm)
    joined = chr(10).join(lines)
    assert 'Motivo: UNKNOWN' not in joined


# ============================================================
# H8 (2026-10-15): "Sector Lider" no debe aparecer cuando el estado
# es UNRESOLVED por falta de datos. En ese caso, "Candidato a lider
# (no confirmado)".
# ============================================================

def _make_slpm_h8(state, reason_code, sector='Technology'):
    return {
        'sector': sector,
        'state': state,
        'state_reason': 'test',
        'state_reason_code': reason_code,
        'opportunity_quadrant': 'Transition',
        'leader_breadth_v2': {
            'n_used': 0, 'expected_leaders': 5, 'coverage': 0.0,
            'coverage_warning': True,
            'rs_breadth': float('nan'),
            'rs_breadth_status': 'NO_LEADERS',
            'leader_breadth_status': 'NO_LEADERS',
            'momentum_breadth': float('nan'),
            'flow_breadth': float('nan'),
            'wyckoff_breadth': float('nan'),
            'composite': float('nan'),
            'effective_composite': float('nan'),
        },
        'input_scores': {'effective_breadth': 0.0, 'persistence': 0.5,
                         'tactical': 0.0, 'structural': 0.0},
        'flow_divergence_v2': {'composite': None},
        'leader_integrity': {'lis': 0.0, 'n_leaders': 0},
    }


def test_h8_sector_lider_confirmado_cuando_estado_normal():
    """Con estado distinto de UNRESOLVED, se muestra 'Sector Lider'."""
    slpm = _make_slpm_h8('CONFIRMED', 'CONFIRMED')
    lines = render_slpm_v12(slpm)
    joined = '\n'.join(lines)
    assert 'Sector Líder' in joined
    assert 'Candidato a líder' not in joined


def test_h8_candidato_cuando_no_leaders():
    slpm = _make_slpm_h8('UNRESOLVED', 'NO_LEADERS')
    lines = render_slpm_v12(slpm)
    joined = '\n'.join(lines)
    assert 'Candidato a líder (no confirmado)' in joined
    assert 'no confirma liderazgo' in joined
    assert 'NO_LEADERS' in joined


def test_h8_candidato_cuando_input_unavailable():
    slpm = _make_slpm_h8('UNRESOLVED', 'INPUT_UNAVAILABLE')
    lines = render_slpm_v12(slpm)
    joined = '\n'.join(lines)
    assert 'Candidato a líder' in joined


def test_h8_candidato_cuando_insufficient_rs():
    slpm = _make_slpm_h8('UNRESOLVED', 'INSUFFICIENT_DATA_RS')
    lines = render_slpm_v12(slpm)
    joined = '\n'.join(lines)
    assert 'Candidato a líder' in joined


def test_h8_candidato_cuando_insufficient_coverage():
    slpm = _make_slpm_h8('UNRESOLVED', 'INSUFFICIENT_COVERAGE')
    lines = render_slpm_v12(slpm)
    joined = '\n'.join(lines)
    assert 'Candidato a líder' in joined


def test_h8_sector_lider_preservado_en_unresolved_por_senales_mixtas():
    """UNRESOLVED por MIXED_OR_INSUFFICIENT_SIGNALS no es insuficiencia
    de datos. En ese caso se mantiene 'Sector Lider'."""
    slpm = _make_slpm_h8('UNRESOLVED', 'MIXED_OR_INSUFFICIENT_SIGNALS')
    lines = render_slpm_v12(slpm)
    joined = '\n'.join(lines)
    assert 'Sector Líder' in joined
    assert 'Candidato a líder' not in joined
