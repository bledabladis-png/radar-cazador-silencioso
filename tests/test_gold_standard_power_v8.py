# -*- coding: utf-8 -*-
"""Test sanity power_v8. Verifica varianza > 0."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.power_v8 import (
    escenarios_factibles,
    evaluar_escenario,
    _poblacion_por_estrato,
)
from scripts.gold_standard.sampling_b_v8 import asignar_n_h


def _make_estratos_sinteticos():
    """4 celdas x 4 estratos."""
    rows = []
    for celda, N in [("C1", 500), ("C2", 500), ("C3", 5000), ("C4", 50000)]:
        for s in ("A", "B", "C", "D"):
            rows.append({
                "celda": celda, "sector": s, "periodo": "P1",
                "N_h": N,
            })
    return pd.DataFrame(rows)


def test_poblacion_por_estrato_conteos():
    e = _make_estratos_sinteticos()
    q_D = 0.0105
    n_Y1 = _poblacion_por_estrato(e, 0.008, 0.70, 0.995, q_D)
    assert n_Y1.sum() > 0
    assert (n_Y1 >= 0).all()
    N_h = e["N_h"].to_numpy()
    assert (n_Y1 <= N_h).all()


def test_evaluar_escenario_tiene_varianza():
    """Criterio de sanity: q95(se) NO puede ser 0 si hay varianza."""
    e = _make_estratos_sinteticos()
    e = asignar_n_h(e, K=50)
    res = evaluar_escenario(0.008, 0.70, 0.995, e, R_MC=100, B=200, master_seed=42)
    q95 = res["q95_MC"]
    # Si algun estimador tiene varianza > 0 en la simulacion, q95 > 0
    n_positivos = sum(1 for k in ("se", "sp", "ppv", "npv")
                       if np.isfinite(q95[k]) and q95[k] > 0)
    assert n_positivos >= 3, f"q95_MC = {q95}"


def test_evaluar_escenario_estructura():
    e = _make_estratos_sinteticos()
    e = asignar_n_h(e, K=50)
    res = evaluar_escenario(0.008, 0.70, 0.995, e, R_MC=50, B=100, master_seed=42)
    for k in ("q95_MC", "pass_estimador", "invalid_rate_max", "PASS"):
        assert k in res


def test_escenarios_factibles_no_vacio():
    q_D = 2559 / 243435
    esc = escenarios_factibles(q_D)
    assert len(esc) == 12