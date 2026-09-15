# tests/test_fu003a_ampliacion.py
"""Tests FU-003a ampliacion: +0 en Rotacion sectorial y Matriz Evidencia."""
import numpy as np
import pandas as pd

from src.report.market_context import render_rotacion_reciente
from src.report.synthesis import render_matriz_evidencia


def test_rotacion_cero_sin_signo():
    df = pd.DataFrame([{
        'sector': 'XLF', 'rank_actual': 5,
        'rank_change_5d': 0, 'rank_change_10d': 3, 'rank_change_20d': -2,
        'lectura_5d': 'Estable', 'lectura_10d': 'Mejora', 'lectura_20d': 'Deterioro',
    }])
    joined = ''.join(render_rotacion_reciente(df))
    assert '| 0 |' in joined
    assert '+0 |' not in joined
    assert '+3 |' in joined
    assert '-2 |' in joined


def test_matriz_evidencia_cero_sin_signo():
    df = pd.DataFrame([{
        'sector': 'XLE',
        'price_evidence': 0, 'breadth_evidence': 1, 'primary_flow_evidence': -1,
        'proxy_flow_evidence': 0, 'wyckoff_evidence': -1, 'credit_evidence': 1,
        'volatility_evidence': -1, 'evidence_quality': 'Baja',
        'alignment_reading': 'MIXTA',
    }])
    joined = ''.join(render_matriz_evidencia(df))
    assert '+0' not in joined
    assert '| +1' in joined
    assert '| -1' in joined


def test_matriz_evidencia_nan_sigue_na():
    """NaN debe seguir como NA, no como N/D."""
    df = pd.DataFrame([{
        'sector': 'XLE',
        'price_evidence': np.nan, 'breadth_evidence': 1, 'primary_flow_evidence': 1,
        'proxy_flow_evidence': 1, 'wyckoff_evidence': 1, 'credit_evidence': 1,
        'volatility_evidence': 1, 'evidence_quality': 'Baja', 'alignment_reading': 'FAV',
    }])
    joined = ''.join(render_matriz_evidencia(df))
    assert 'NA' in joined
