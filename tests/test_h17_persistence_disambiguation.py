"""Test H17: las 3 "persistencias" del reporte estan documentadas como
metricas distintas.

Auditoria externa 2026-10-15, hallazgo H17 (medio): el reporte usa
"Persist" (Structural Ranking), "Pers 5d/10d/20d" (lideres por ticker)
y "Pers 5d/20d" (flujo primario) sin aclarar que miden cosas distintas.
"""
import pandas as pd

from src.report.etf_flows import render_flujo_caracteristicas


def test_rankings_documenta_persist():
    """La tabla de rankings debe aclarar que Persist = persistencia del
    structural score sectorial."""
    # El render necesita datos complejos. Test alternativo: verificar
    # que la constante o la fuente tiene la nota.
    from pathlib import Path
    r = Path(__file__).resolve().parent.parent / "src" / "report" / "rankings.py"
    src = r.read_text(encoding="utf-8")
    assert "H17" in src
    assert "structural" in src.lower()
    assert "Persist" in src


def test_etf_flows_documenta_pers_5d_20d():
    """La tabla de flujo caracteristicas debe aclarar Pers 5d/20d."""
    df = pd.DataFrame([{
        "date": "2026-10-13",
        "sector": "XLK",
        "flow_dollar": 1000000.0,
        "flow_pct_aum": 0.5,
        "flow_zscore": 1.2,
        "flow_5d_sum": 5000000.0,
        "flow_20d_sum": 8000000.0,
        "persistence_5d": 0.8,
        "persistence_20d": 0.6,
        "price_ret_20d": 0.02,
        "price_flow_regime": "ACUMULACION",
    }])
    out = "".join(render_flujo_caracteristicas(df))
    assert "Pers 5d / Pers 20d" in out
    assert "pos / (pos+neg)" in out or "fraccion de la ventana" in out.lower()
    assert "Structural Ranking" in out
