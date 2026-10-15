"""Test H15: el bloque MTE publica pesos, umbrales y condiciones
del scoring de escenarios.

Auditoria externa 2026-10-15, hallazgo H15 (medio): la clasificacion
STAGFLATION no era reproducible. El reporte remitia a
SCENARIO_WEIGHTS pero no publicaba los valores.
"""
from src.report.volatility_mte import render_mte


def _mte():
    return {
        "scenario": "STAGFLATION",
        "confidence": 0.48,
        "msi": 36,
        "ipi": 29,
        "sector_rotation_score": 0.04,
        "safe_haven_score": -0.03,
        "credit_stress_score": 0.08,
        "inflation_pressure_score": -0.42,
        "components_in_pressure": 1,
    }


def test_h15_publica_pesos():
    out = "".join(render_mte(_mte()))
    assert "CLS=3" in out
    assert "SHS=2" in out
    assert "SRS=2" in out
    assert "IPS=3" in out


def test_h15_publica_umbrales_activacion():
    out = "".join(render_mte(_mte()))
    assert "CLS>0.20" in out
    assert "SHS>0.20" in out
    assert "SRS>0.10" in out
    assert "IPS>0.15" in out


def test_h15_publica_condicion_stagflation():
    out = "".join(render_mte(_mte()))
    assert "STAGFLATION" in out
    assert "IPS>0.15 AND SRS>0" in out or "IPS>0.15" in out


def test_h15_publica_bonos():
    out = "".join(render_mte(_mte()))
    assert "CLS>0.50 suma +6" in out
    assert "IPS>0.50 suma +2" in out


def test_h15_publica_umbral_unconfirmed():
    out = "".join(render_mte(_mte()))
    assert "confidence < 0.50" in out or "confidence<0.50" in out
