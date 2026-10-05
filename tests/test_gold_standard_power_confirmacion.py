# -*- coding: utf-8 -*-
"""Tests B_power con confirmacion (P1.2, dictamen).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.power import (
    B_CONFIRMACION,
    B_SCREENING,
    dimensionar_n_B_con_confirmacion,
)


def _estratos_sinteticos(n_por_estrato=10000, k=9):
    return pd.DataFrame({
        "sector": [f"S{i}" for i in range(k)],
        "periodo": ["P1"] * k,
        "N_h": [n_por_estrato] * k,
    })


def test_constantes_p12():
    assert B_SCREENING == 500
    assert B_CONFIRMACION == 2000


def test_confirmacion_devuelve_fase():
    estratos = _estratos_sinteticos(n_por_estrato=10000, k=9)
    escenarios = [{"pi": 0.30, "se": 0.80, "sp": 0.80}]
    out = dimensionar_n_B_con_confirmacion(
        estratos, escenarios=escenarios,
        n_min=100, n_max=300, paso=100,
        seed=42,
    )
    assert "fase" in out
    assert out["fase"] in (
        "confirmado", "confirmado_tras_ajuste",
        "screening_fallido", "confirmacion_fallida",
    )


def test_confirmacion_con_escenario_facil():
    estratos = _estratos_sinteticos(n_por_estrato=20000, k=9)
    escenarios = [{"pi": 0.30, "se": 0.80, "sp": 0.80}]
    out = dimensionar_n_B_con_confirmacion(
        estratos, escenarios=escenarios,
        n_min=100, n_max=400, paso=100,
        seed=42,
    )
    assert out["fase"] in (
        "confirmado", "confirmado_tras_ajuste", "confirmacion_fallida",
    )



def test_confirmacion_determinista():
    """Mismo seed -> mismo resultado."""
    estratos = _estratos_sinteticos(n_por_estrato=10000, k=9)
    escenarios = [{"pi": 0.30, "se": 0.80, "sp": 0.80}]
    out1 = dimensionar_n_B_con_confirmacion(
        estratos, escenarios=escenarios,
        n_min=100, n_max=300, paso=100,
        seed=42,
    )
    out2 = dimensionar_n_B_con_confirmacion(
        estratos, escenarios=escenarios,
        n_min=100, n_max=300, paso=100,
        seed=42,
    )
    assert out1["fase"] == out2["fase"]
    assert out1["n_B_min"] == out2["n_B_min"]


def test_confirmacion_reporta_B_usados():
    """Si hay fase posterior al screening, B_screening y B_confirmacion presentes."""
    estratos = _estratos_sinteticos(n_por_estrato=10000, k=9)
    escenarios = [{"pi": 0.30, "se": 0.80, "sp": 0.80}]
    out = dimensionar_n_B_con_confirmacion(
        estratos, escenarios=escenarios,
        n_min=100, n_max=300, paso=100,
        seed=42,
    )
    if out["fase"] == "screening_fallido":
        return
    assert out["B_screening"] == 500
    assert out["B_confirmacion"] == 2000
