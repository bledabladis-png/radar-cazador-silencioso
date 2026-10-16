# -*- coding: utf-8 -*-
"""Tests H11 + H16 (auditoria 2026-10-16)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ============================================================
# H11 - tabla SPDR con dShares
# ============================================================
def test_spdr_incluye_columna_dshares():
    from src.report.etf_flows import render_flujo_spdr
    df = pd.DataFrame([{
        'ticker': 'XLP', 'Date': '2026-10-14',
        'nav': 86.03, 'shares_outstanding': 166_771_809,
        'shares_change': -8_250_000,
        'total_net_assets': 14_347_600_433,
        'primary_flow_usd': -709_758_464.25,
        'primary_flow_pct': -4.95,
        'primary_flow_z': -5.0,
        'primary_flow_z_regime': 'SATURATED',
        'primary_flow_z_raw': -6.787,
    }])
    out = "".join(render_flujo_spdr(df))
    assert "ΔShares" in out, "columna dShares no aparece"
    assert "-8,250,000" in out, "valor dShares no renderiza"
    assert "H11 auditoria" in out, "nota H11 no aparece"


def test_spdr_dshares_nan_va_a_nd():
    from src.report.etf_flows import render_flujo_spdr
    df = pd.DataFrame([{
        'ticker': 'XLE', 'Date': '2026-10-14',
        'nav': 67.60, 'shares_outstanding': 654_348_400,
        'shares_change': float('nan'),
        'total_net_assets': 44_235_116_839,
        'primary_flow_usd': 0.0,
        'primary_flow_pct': 0.0,
        'primary_flow_z': 0.0,
        'primary_flow_z_regime': 'NORMAL',
        'primary_flow_z_raw': 0.0,
    }])
    out = "".join(render_flujo_spdr(df))
    # En la fila de XLE, dShares debe ser N/D
    lineas_xle = [l for l in out.split("\n") if "| XLE |" in l]
    assert len(lineas_xle) == 1
    assert "N/D" in lineas_xle[0]


# ============================================================
# H16 - unidades de percentiles
# ============================================================
def test_volatility_percentiles_en_porcentaje():
    from src.report.volatility_mte import render_estructura_volatilidad
    df = pd.DataFrame([{
        'date': '2026-10-14',
        'vix_level': 16.87,
        'vix_percentile_20d': 1.00,       # escala 0-1
        'vix_percentile_60d': 0.83,       # escala 0-1
        'term_structure_ratio': 1.11,
        'pcr_zscore': 0.99,
        'pcr_percentile_20d': 80.0,       # escala 0-100
        'volatility_reading': 'Volatilidad elevada reciente',
        'term_structure_reading': 'Contango',
    }])
    out = "".join(render_estructura_volatilidad(df))
    # Perc 20d: 1.00 -> 100%
    assert "| 100%" in out or "100% |" in out
    # Perc 60d: 0.83 -> 83%
    assert "83%" in out
    # Perc PCR: 80.0 -> 80%
    assert "80%" in out
    # Nota H16 presente
    assert "H16 auditoria" in out


def test_volatility_percentiles_nan_va_a_nd():
    from src.report.volatility_mte import render_estructura_volatilidad
    df = pd.DataFrame([{
        'date': '2026-10-14',
        'vix_level': 16.87,
        'vix_percentile_20d': float('nan'),
        'vix_percentile_60d': 0.5,
        'term_structure_ratio': 1.11,
        'pcr_zscore': 0.99,
        'pcr_percentile_20d': float('nan'),
        'volatility_reading': 'r',
        'term_structure_reading': 't',
    }])
    out = "".join(render_estructura_volatilidad(df))
    # 2 campos deben mostrar N/D
    assert out.count("N/D") >= 2