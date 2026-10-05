# -*- coding: utf-8 -*-
"""Tests simulacion de potencia (dimensionamiento de n_B).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.constants import (
    E_NPV,
    E_PPV,
    E_SE,
    E_SP,
)
from scripts.gold_standard.power import (
    escenarios_default,
    simular_escenario,
    dimensionar_n_B,
    _asignar_n_h,
    _cumple_restricciones,
)


def _estratos_sinteticos(n_por_estrato: int = 500, k: int = 9):
    return pd.DataFrame({
        "sector": [f"S{i}" for i in range(k)],
        "periodo": ["P1"] * k,
        "N_h": [n_por_estrato] * k,
    })


def test_escenarios_default_son_27():
    e = escenarios_default()
    assert len(e) == 27
    pis = {x["pi"] for x in e}
    ses = {x["se"] for x in e}
    sps = {x["sp"] for x in e}
    assert pis == {0.10, 0.20, 0.30}
    assert ses == {0.60, 0.70, 0.80}
    assert sps == {0.60, 0.70, 0.80}


def test_asignar_n_h_minimo_rao_wu():
    estratos = _estratos_sinteticos(n_por_estrato=500, k=9)
    n_h = _asignar_n_h(estratos, n_B=90)
    assert (n_h >= 2).all()
    assert (n_h <= 500).all()


def test_asignar_n_h_proporcional():
    estratos = _estratos_sinteticos(n_por_estrato=1000, k=4)
    n_h = _asignar_n_h(estratos, n_B=100)
    assert n_h.sum() == 100
    assert (n_h == 25).all()


def test_simular_escenario_devuelve_metricas():
    estratos = _estratos_sinteticos(n_por_estrato=2000, k=4)
    res = simular_escenario(
        pi=0.20, se=0.70, sp=0.70,
        estratos=estratos, n_B=200, B=50, seed=42,
    )
    for k in ("se_mean", "sp_mean", "ppv_mean", "npv_mean",
              "se_e", "sp_e", "ppv_e", "npv_e"):
        assert k in res
        assert not np.isnan(res[k])


def test_simular_escenario_determinista():
    estratos = _estratos_sinteticos(n_por_estrato=500, k=4)
    r1 = simular_escenario(0.20, 0.70, 0.70, estratos, 100, B=30, seed=1)
    r2 = simular_escenario(0.20, 0.70, 0.70, estratos, 100, B=30, seed=1)
    assert r1["se_mean"] == r2["se_mean"]
    assert r1["se_e"] == r2["se_e"]


def test_simular_se_cerca_del_valor_verdadero_con_n_grande():
    estratos = _estratos_sinteticos(n_por_estrato=5000, k=9)
    res = simular_escenario(
        pi=0.30, se=0.80, sp=0.80,
        estratos=estratos, n_B=800, B=200, seed=42,
    )
    assert abs(res["se_mean"] - 0.80) < 0.05
    assert abs(res["sp_mean"] - 0.80) < 0.05


def test_error_baja_al_aumentar_n():
    estratos = _estratos_sinteticos(n_por_estrato=5000, k=9)
    r_small = simular_escenario(0.20, 0.70, 0.70, estratos, 100, B=200, seed=1)
    r_large = simular_escenario(0.20, 0.70, 0.70, estratos, 800, B=200, seed=1)
    assert r_large["se_e"] < r_small["se_e"]
    assert r_large["sp_e"] < r_small["sp_e"]


def test_cumple_restricciones_true_para_valores_bajos():
    res = {
        "se_e": E_SE - 0.01,
        "sp_e": E_SP - 0.01,
        "ppv_e": E_PPV - 0.01,
        "npv_e": E_NPV - 0.01,
    }
    assert _cumple_restricciones(res)


def test_cumple_restricciones_false_si_alguno_excede():
    res = {
        "se_e": E_SE + 0.01,
        "sp_e": E_SP - 0.01,
        "ppv_e": E_PPV - 0.01,
        "npv_e": E_NPV - 0.01,
    }
    assert not _cumple_restricciones(res)


def test_dimensionar_n_B_devuelve_resultado():
    estratos = _estratos_sinteticos(n_por_estrato=10000, k=9)
    # Escenarios muy laxos: uno solo con pi alto y Se/Sp altos
    escenarios = [{"pi": 0.30, "se": 0.80, "sp": 0.80}]
    out = dimensionar_n_B(
        estratos,
        escenarios=escenarios,
        n_min=100, n_max=500, paso=100,
        B=100, seed=42,
    )
    assert "n_B_min" in out
    assert "excede_capacidad" in out
    if not out["excede_capacidad"]:
        assert out["n_B_min"] is not None
        assert 100 <= out["n_B_min"] <= 500


def test_dimensionar_n_B_determinista():
    estratos = _estratos_sinteticos(n_por_estrato=5000, k=9)
    escenarios = [{"pi": 0.30, "se": 0.80, "sp": 0.80}]
    o1 = dimensionar_n_B(estratos, escenarios=escenarios,
                         n_min=100, n_max=200, paso=100, B=50, seed=7)
    o2 = dimensionar_n_B(estratos, escenarios=escenarios,
                         n_min=100, n_max=200, paso=100, B=50, seed=7)
    assert o1["n_B_min"] == o2["n_B_min"]
    assert o1["excede_capacidad"] == o2["excede_capacidad"]