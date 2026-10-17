# -*- coding: utf-8 -*-
"""Regresion C1 (2026-10-17): la sintesis no debe confundir
Condiciones Financieras (HIGH_STRESS) con Liquidez Real (FRED, NEUTRAL).

Antes: "Condiciones financieras: HIGH_STRESS — estres elevado en
credito y liquidez". Menciona liquidez cuando el modulo de liquidez
real (FRED, v2) esta en un estado distinto. Son dos dimensiones
independientes. El Resumen de Regimenes las muestra por separado.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.report.synthesis import render_sintesis_senales


def test_high_stress_no_menciona_liquidez():
    """HIGH_STRESS -> texto solo habla de credito, no de liquidez."""
    out = render_sintesis_senales(
        macro_regime="SLOWDOWN",
        slpm_v12_data=None,
        financial_regime="HIGH_STRESS",
        sector_dispersion_data=None,
        breadth_values=None,
        price_flow_divergences=None,
    )
    text = "".join(out)
    assert "HIGH_STRESS" in text
    assert "credito" in text.lower() or "crédito" in text.lower()
    # La clave de la regresion: no atribuir "liquidez" al modulo financiero.
    assert "credito y liquidez" not in text.lower()
    assert "crédito y liquidez" not in text.lower()
    # Debe redirigir a la seccion separada.
    assert "Resumen de Reg" in text  # acepta "Regimenes" o "Regímenes"