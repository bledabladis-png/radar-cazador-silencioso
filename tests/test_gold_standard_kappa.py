# -*- coding: utf-8 -*-
"""Tests Fleiss kappa y Cohen kappa.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.kappa import (
    cohen_kappa,
    encode_labels,
    fleiss_kappa,
    kappa_report,
)


def test_encode_labels_mapea_correctamente():
    s = pd.Series(["SOW", "NO_SOW", "SOW", None, "SOW", ""])
    enc = encode_labels(s, ("SOW", "NO_SOW"))
    assert list(enc) == [0, 1, 0, -1, 0, -1]


def test_encode_labels_ignora_desconocidos():
    s = pd.Series(["SOW", "OTRO", "NO_SOW"])
    enc = encode_labels(s, ("SOW", "NO_SOW"))
    assert list(enc) == [0, -1, 1]


def test_fleiss_acuerdo_perfecto():
    M = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]])
    k = fleiss_kappa(M, n_categorias=2)
    assert abs(k - 1.0) < 1e-9


def test_fleiss_desacuerdo_parcial():
    # Caso computable a mano:
    # Item 1: (0,0,0) -> P_1 = 1
    # Item 2: (0,0,1) -> P_2 = 1/3
    # Item 3: (1,1,0) -> P_3 = 1/3
    # P_obs = 5/9
    # p_0 = 6/9, p_1 = 3/9 -> P_exp = 4/9 + 1/9 = 5/9
    # kappa = 0
    M = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 0]])
    k = fleiss_kappa(M, n_categorias=2)
    assert abs(k - 0.0) < 1e-9


def test_fleiss_sin_categorias_vacio():
    M = np.zeros((0, 3), dtype=int)
    assert np.isnan(fleiss_kappa(M, n_categorias=2))

def test_cohen_acuerdo_perfecto():
    a = np.array([0, 0, 1, 1, 0])
    b = np.array([0, 0, 1, 1, 0])
    assert abs(cohen_kappa(a, b) - 1.0) < 1e-9


def test_cohen_valor_conocido():
    # a = [1,1,1,1,0,0,0,0]
    # b = [1,1,1,0,0,0,0,0]
    # p_o = 7/8 = 0.875
    # p_e = 0.5*0.375 + 0.5*0.625 = 0.5
    # kappa = 0.375/0.5 = 0.75
    a = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    b = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    assert abs(cohen_kappa(a, b) - 0.75) < 1e-9


def test_cohen_longitud_distinta():
    with pytest.raises(ValueError, match="distinta longitud"):
        cohen_kappa(np.array([0, 1]), np.array([0]))


def test_cohen_vacio():
    assert np.isnan(cohen_kappa(np.array([], dtype=int), np.array([], dtype=int)))

def _make_panel_3raters(rows):
    """rows: lista de (a1, a2, a3)."""
    return pd.DataFrame([
        {"anotador_1": r[0], "anotador_2": r[1], "anotador_3": r[2]}
        for r in rows
    ])


def test_kappa_report_fleiss_solo_3_validos():
    df = _make_panel_3raters([
        ("SOW", "SOW", "SOW"),
        ("SOW", "SOW", ""),
        ("NO_SOW", "NO_SOW", "NO_SOW"),
        ("", "SOW", "SOW"),
    ])
    res = kappa_report(df, ("SOW", "NO_SOW"))
    # Fleiss: solo 2 casos con 3 validos
    assert res.fleiss_n == 2
    assert res.n_total == 4


def test_kappa_report_cohen_pairwise():
    df = _make_panel_3raters([
        ("SOW", "SOW", "SOW"),
        ("SOW", "SOW", ""),
        ("NO_SOW", "NO_SOW", "NO_SOW"),
    ])
    res = kappa_report(df, ("SOW", "NO_SOW"))
    # par (0,1): 3 validos
    assert res.cohen_pairwise[(0, 1)]["n"] == 3
    # par (0,2): 2 validos
    assert res.cohen_pairwise[(0, 2)]["n"] == 2
    # par (1,2): 2 validos
    assert res.cohen_pairwise[(1, 2)]["n"] == 2


def test_kappa_report_rechaza_columna_faltante():
    df = pd.DataFrame({"anotador_1": ["SOW"]})
    with pytest.raises(ValueError, match="Falta columna"):
        kappa_report(df, ("SOW", "NO_SOW"))


def test_kappa_report_determinista():
    df = _make_panel_3raters([
        ("SOW", "SOW", "NO_SOW"),
        ("NO_SOW", "NO_SOW", "SOW"),
    ])
    r1 = kappa_report(df, ("SOW", "NO_SOW"))
    r2 = kappa_report(df, ("SOW", "NO_SOW"))
    assert r1.fleiss_kappa == r2.fleiss_kappa
    for k in r1.cohen_pairwise:
        assert r1.cohen_pairwise[k] == r2.cohen_pairwise[k]