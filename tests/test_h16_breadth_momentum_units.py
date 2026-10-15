"""Test H16: la tabla Momentum de amplitud declara unidad (pp) y reglas
de Expansion/Deterioro.

Auditoria externa 2026-10-15, hallazgo H16 (medio): el reporte mostraba
valores como +95.0 y -70.0 sin especificar unidad ni las reglas que
asignan Expansion/Deterioro.
"""
import pandas as pd

from src.report.sector_context import render_momentum_amplitud


def _df():
    return pd.DataFrame([{
        "date": "2026-10-13",
        "sector": "XLU",
        "delta_1d_ema20": 0.0,
        "delta_5d_ema20": 25.0,
        "delta_20d_ema20": 85.0,
        "delta_5d_ema50": 40.0,
        "delta_5d_ema200": 15.0,
        "breadth_expansion_5d": True,
        "breadth_deterioration_5d": False,
    }])


def test_h16_unidad_pp():
    out = "".join(render_momentum_amplitud(_df()))
    assert "PUNTOS PORCENTUALES" in out or "pp" in out.lower()
    assert "40% a 50%" in out or "pp" in out.lower()


def test_h16_regla_expansion():
    out = "".join(render_momentum_amplitud(_df()))
    assert "delta_5d_ema20 >= +5 pp" in out or ">= +5 pp" in out
    assert "pct_above_ema20 > 60%" in out or "> 60%" in out


def test_h16_regla_deterioro():
    out = "".join(render_momentum_amplitud(_df()))
    assert "delta_5d_ema20 <= -5 pp" in out or "<= -5 pp" in out


def test_h16_fuente_declarada():
    out = "".join(render_momentum_amplitud(_df()))
    assert "sector_breadth_momentum.py" in out
