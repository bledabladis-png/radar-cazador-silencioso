# -*- coding: utf-8 -*-
"""Tests power_v8 (escenarios + poblacion + MC).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.power_v8 import (
    _hamilton,
    escenarios_factibles,
    generar_poblacion_sintetica,
    evaluar_escenario,
)
from scripts.gold_standard.sampling_b_v8 import asignar_n_h


Q_D_REAL = 2559 / 243435


def test_escenarios_factibles_no_vacio():
    e = escenarios_factibles(Q_D_REAL)
    assert len(e) > 0


def test_escenarios_factibles_todos_compatibles():
    e = escenarios_factibles(Q_D_REAL)
    for s in e:
        assert s["pi_Y"] * s["Se"] <= Q_D_REAL + 1e-12
        assert 0.0 <= s["Sp"] <= 1.0


def test_escenarios_factibles_es_12():
    e = escenarios_factibles(Q_D_REAL)
    assert len(e) == 12


def test_poblacion_sintetica_conteos_ok():
    N_h_celda = {"C1": 1000, "C2": 500, "C3": 200, "C4": 300}
    pop = generar_poblacion_sintetica(N_h_celda, 0.008, 0.70, 0.995)
    for celda in ("C1", "C2", "C3", "C4"):
        assert pop[celda]["n_Y1"] + pop[celda]["n_Y0"] == pop[celda]["N"]


def test_poblacion_sintetica_Se_aproximado():
    N_h_celda = {"C1": 10_000, "C2": 5_000, "C3": 200, "C4": 300}
    pop = generar_poblacion_sintetica(N_h_celda, 0.008, 0.70, 0.995)
    n_Y1_D1 = pop["C1"]["n_Y1"] + pop["C2"]["n_Y1"]
    n_D1 = pop["C1"]["N"] + pop["C2"]["N"]
    se_efectivo = n_Y1_D1 / n_D1
    assert abs(se_efectivo - 0.70) < 0.01


def test_hamilton_suma_exacta():
    cuotas = np.array([1.3, 2.7, 0.5, 4.5])
    assert _hamilton(cuotas, 9).sum() == 9


def test_evaluar_escenario_devuelve_estructura():
    estratos = pd.DataFrame({
        "celda": ["C1"] * 4 + ["C2"] * 4 + ["C3"] * 4 + ["C4"] * 4,
        "sector": ["A", "B"] * 8,
        "periodo": ["P1"] * 16,
        "N_h": [100] * 16,
    })
    e = asignar_n_h(estratos, K=20)
    res = evaluar_escenario(0.008, 0.70, 0.995, e, R_MC=10, B=50)
    assert "q95_MC" in res
    assert "pass_estimador" in res
    assert "invalid_rate_max" in res