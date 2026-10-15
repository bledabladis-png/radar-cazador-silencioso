"""Test H9: la nota de la Matriz de Evidencia cuantifica la
correlacion entre evidencias.

Auditoria externa 2026-10-15, hallazgo H9 (medio): el auditor pedia
cuantificar el efecto de la doble contabilizacion en la Matriz de
Evidencia. Medicion empirica: de los 10 pares posibles, solo
price-breadth tiene correlacion material (r=0.64); el resto |r|<0.3.
"""
import pandas as pd

from src.report.synthesis import render_matriz_evidencia


def _df_ejemplo():
    return pd.DataFrame([{
        "sector": "XLK",
        "price_evidence": 1,
        "breadth_evidence": 1,
        "primary_flow_evidence": 1,
        "proxy_flow_evidence": -1,
        "wyckoff_evidence": 1,
        "credit_evidence": 0,
        "volatility_evidence": 1,
        "evidence_quality": "Baja",
        "alignment_reading": "EVIDENCIA PREDOMINANTEMENTE FAVORABLE",
    }])


def test_h9_cuantifica_correlacion():
    """La nota debe incluir el valor numerico de la correlacion medida."""
    out = render_matriz_evidencia(_df_ejemplo())
    joined = "".join(out)
    assert "H9" in joined
    assert "r=0.64" in joined or "0.64" in joined
    # Debe mencionar que el resto de pares esta por debajo de 0.3
    assert "|r|<0.3" in joined or "0.3" in joined


def test_h9_desmiente_duplicacion_proxy_flow():
    """La nota debe aclarar que proxy_flow NO correlaciona con precio."""
    out = render_matriz_evidencia(_df_ejemplo())
    joined = "".join(out)
    assert "Flow Proxy" in joined
    # Debe indicar que no correlaciona con precio
    assert "no correlaciona" in joined.lower() or "r=0.015" in joined


def test_h9_no_modifica_algoritmo():
    """La nota aclara que no hay justificacion para modificar el algoritmo."""
    out = render_matriz_evidencia(_df_ejemplo())
    joined = "".join(out)
    # No debe prometer cambio de pesos
    assert "reweighting" not in joined.lower()
    assert "pesos" not in joined.lower() or "sin pesos" in joined.lower()
