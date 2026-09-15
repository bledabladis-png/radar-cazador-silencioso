# tests/test_fu003a_render.py
"""Tests FU-003a commit 2: % AUM a 2 decimales en renders."""
import pandas as pd

from src.report.flows_international import render_flujo_daxex, render_flujo_iwm


def _make_blackrock_df():
    return pd.DataFrame([{
        'date': pd.Timestamp('2026-09-15'),
        'nav': 100.0,
        'shares_outstanding': 1_000_000,
        'shares_change': 0,
        'estimated_flow_eur': 0.0,
        'flow_pct_assets': 0.0,
        'flow_zscore': 0.0,
    }])


def _make_iwm_df():
    return pd.DataFrame([{
        'date': pd.Timestamp('2026-09-15'),
        'nav': 100.0,
        'shares_outstanding': 1_000_000,
        'shares_change': 0,
        'primary_flow_usd': 0.0,
        'primary_flow_pct': 0.0,
        'primary_flow_z': 0.0,
    }])


def test_daxex_cero_sin_signo_2decimales():
    """FU-003a: cero -> '0.00%' no '+0.000000%'."""
    lines = render_flujo_daxex(_make_blackrock_df())
    joined = ''.join(lines)
    assert '**Flujo % AUM:** 0.00%' in joined
    assert '+0.000000%' not in joined
    assert '0.000000%' not in joined


def test_iwm_cero_sin_signo_2decimales():
    """FU-003a: cero -> '0.00%' en IWM."""
    lines = render_flujo_iwm(_make_iwm_df())
    joined = ''.join(lines)
    assert '**Flujo % AUM:** 0.00%' in joined
    assert '+0.000000%' not in joined


def test_daxex_no_cero_mantiene_signo():
    """Valor no cero -> mantiene '+X.XX%'."""
    df = _make_blackrock_df()
    df.loc[0, 'flow_pct_assets'] = 0.0123  # -> 1.23%
    lines = render_flujo_daxex(df)
    joined = ''.join(lines)
    assert '+1.23%' in joined


def test_daxex_flujo_estimado_cero_sin_signo():
    """FU-003a: cero -> '0.00' en Flujo Estimado."""
    lines = render_flujo_daxex(_make_blackrock_df())
    joined = ''.join(lines)
    assert '**Flujo Estimado (EUR):** 0.00' in joined
    assert '+0.00' not in joined
