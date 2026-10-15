"""Test H18: la tabla de rendimiento QQQ/SPY aclara que "Desde inicio"
compara periodos no homogeneos.

Auditoria externa 2026-10-15, hallazgo H18 (medio): QQQ y SPY tienen
fechas de lanzamiento distintas (1999-03-10 y 1993-01-22). La columna
"Desde inicio" no es comparable directamente.
"""
import pandas as pd

from src.report.flows_international import render_rendimiento_qqq


def _df():
    return pd.DataFrame([{
        "ticker": "QQQ",
        "ytd": 0.2145, "y1": 0.2673, "y3": 1.0736,
        "y5": 1.1437, "y10": 5.7118, "inception": 16.2995,
        "effectiveDate": "2026-10-13",
    }, {
        "ticker": "SPY",
        "ytd": 0.1498, "y1": 0.2043, "y3": 0.8781,
        "y5": 0.9206, "y10": 3.2220, "inception": 31.3449,
        "effectiveDate": "2026-10-13",
    }])


def test_h18_menciona_fechas_lanzamiento():
    out = "".join(render_rendimiento_qqq(_df()))
    assert "1999-03-10" in out
    assert "1993-01-22" in out


def test_h18_advierte_no_comparable():
    out = "".join(render_rendimiento_qqq(_df()))
    assert "no es valida" in out or "no es comparable" in out.lower()
    assert "periodos NO son homogeneos" in out or "no homogeneos" in out.lower()


def test_h18_recomienda_alternativa():
    out = "".join(render_rendimiento_qqq(_df()))
    assert "10Y" in out
    assert "anualizar" in out.lower()
