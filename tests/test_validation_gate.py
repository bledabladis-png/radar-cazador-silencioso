# -*- coding: utf-8 -*-
"""
Tests del Validation Gate (src/pipeline/validation_gate.py).

Cubren los 4 fixes de 2026-09-12:
  - Check 1: SLPM con validation_errors debe fallar el Gate.
  - Check 2: PCR con pd.NA no debe crashear.
  - Check 5: sin sectores coincidentes debe fallar.
  - Check 5: valores no numericos no deben crashear.
Y un caso feliz que valida que la logica normal sigue OK.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.pipeline.validation_gate import run_validation_gate


def _base_kwargs():
    """Datos sinteticos que pasan el Gate sin errores."""
    return dict(
        slpm_v12_data={
            "state": "UNRESOLVED",
            "opportunity_quadrant": "Transition",
            "sector_etf": "XLK",
            "sector": "Tech",
            "validation_errors": [],
        },
        pcr_data={"total_pcr": 0.85, "last_date": "2026-09-10"},
        darkpool_data={"media_dark_pool": 23.95, "week": "2026-08-18"},
        mte_result={"msi": 35.0, "ipi": 70.0},
        tactical_scores={"XLK": 0.5, "XLE": 0.2},
        structural_scores={"XLK": 0.6, "XLE": 0.3},
    )


def test_gate_caso_feliz():
    """Regresion: con datos normales el Gate pasa."""
    result = run_validation_gate(**_base_kwargs())
    assert result["passed"] is True, f"errors={result['errors']}"
    assert len(result["checks"]) == 10


def test_slpm_errors_fallan_gate():
    """Check 1: si SLPM reporta errores, el Gate debe fallar."""
    kwargs = _base_kwargs()
    kwargs["slpm_v12_data"]["validation_errors"] = ["LIS fuera de rango"]
    result = run_validation_gate(**kwargs)
    assert result["passed"] is False
    assert any("SLPM" in e for e in result["errors"])


def test_check5_sin_sectores_falla():
    """Check 5: sin interseccion tactical/structural, el Gate debe fallar."""
    kwargs = _base_kwargs()
    kwargs["tactical_scores"] = {"XLK": 0.5}
    kwargs["structural_scores"] = {"XLE": 0.3}
    result = run_validation_gate(**kwargs)
    assert result["passed"] is False
    assert any("sin sectores coincidentes" in e for e in result["errors"])


def test_check5_tipo_invalido_no_crashea():
    """Check 5: valores None/no numericos no deben crashear."""
    kwargs = _base_kwargs()
    kwargs["tactical_scores"] = {"XLK": None, "XLE": 0.2}
    kwargs["structural_scores"] = {"XLK": 0.6, "XLE": 0.3}
    result = run_validation_gate(**kwargs)
    # No debe lanzar TypeError; el Gate debe reportar error.
    assert result["passed"] is False
    assert any("no numerico" in e for e in result["errors"])


def test_check2_pdna_no_crashea():
    """Check 2: total_pcr=pd.NA no debe crashear con f-string."""
    kwargs = _base_kwargs()
    kwargs["pcr_data"]["total_pcr"] = pd.NA
    result = run_validation_gate(**kwargs)
    assert result["passed"] is False
    assert any("PCR" in e for e in result["errors"])


def test_check3_nan_falla():
    """Check 3: media_dark_pool=NaN debe fallar."""
    kwargs = _base_kwargs()
    kwargs["darkpool_data"]["media_dark_pool"] = float("nan")
    result = run_validation_gate(**kwargs)
    assert result["passed"] is False
    assert any("Dark Pool" in e for e in result["errors"])


def test_check4_mte_pdna_no_crashea():
    """Check 4: msi=pd.NA no debe crashear."""
    kwargs = _base_kwargs()
    kwargs["mte_result"]["msi"] = pd.NA
    result = run_validation_gate(**kwargs)
    assert result["passed"] is False
    assert any("MTE" in e for e in result["errors"])