"""Test H12: la nota de Flujo Primario aclara que es una estimacion
operativa, no flujo economico del mercado secundario.

Auditoria externa 2026-10-15, hallazgo H12 (medio): la etiqueta "Flujo
Primario" puede inducir a interpretar la estimacion como flujo exacto.
"""
import pandas as pd

from src.report.etf_flows import render_flujo_spdr


def _df():
    return pd.DataFrame([{
        "ticker": "XLK", "nav": 250.0, "shares_outstanding": 1_000_000,
        "total_net_assets": 250_000_000, "primary_flow_usd": 5_000_000.0,
        "primary_flow_pct": 2.0, "primary_flow_z": 1.5,
        "primary_flow_z_raw": 1.5, "primary_flow_z_regime": "NORMAL",
        "Date": "2026-10-13",
    }])


def test_h12_nota_estimacion_operativa():
    """La nota debe aclarar el caracter de estimacion operativa."""
    out = "".join(render_flujo_spdr(_df()))
    assert "ESTIMACION OPERATIVA" in out or "estimacion operativa" in out.lower()
    assert "mercado secundario" in out.lower()


def test_h12_nota_no_afirma_flujo_economico():
    """No debe afirmar que el flujo sea observacion exacta del mercado."""
    out = "".join(render_flujo_spdr(_df()))
    # No debe decir "flujo economico" como afirmacion positiva
    assert "flujo economico del activo" not in out.lower() or "no como" in out.lower()
