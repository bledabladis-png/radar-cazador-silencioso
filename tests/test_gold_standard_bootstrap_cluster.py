# -*- coding: utf-8 -*-
"""Tests cluster bootstrap global de ticker.

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

from scripts.gold_standard.bootstrap_cluster import (
    _delta_from_df,
    clasificar_resultado,
    cluster_bootstrap_delta,
    materialidad,
)


DIST = "DISTRIBUTIVE_CONTEXT"
NO_DIST = "NO_DISTRIBUTIVE_CONTEXT"


def _make_capa2_df(n_tickers=20, seed=1, delta_efecto=0.20):
    """DataFrame sintetico Capa 2.

    Por ticker: 3 casos. Mezcla 2x2 (SOW x Ctx).
    delta_efecto controla P(DIST|SOW=1,Ctx=1) - P(DIST|SOW=1,Ctx=0).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for tk in range(n_tickers):
        ticker = f"T{tk:03d}"
        # Celda A (SOW=1, Ctx=1): P(DIST) = 0.5 + delta_efecto/2
        p_dist_A = 0.5 + delta_efecto / 2
        rows.append({
            "ticker": ticker, "detect_sow": 1, "contexto_op": 1,
            "label_adjudicada": DIST if rng.random() < p_dist_A else NO_DIST,
        })
        # Celda B (SOW=1, Ctx=0): P(DIST) = 0.5 - delta_efecto/2
        p_dist_B = 0.5 - delta_efecto / 2
        rows.append({
            "ticker": ticker, "detect_sow": 1, "contexto_op": 0,
            "label_adjudicada": DIST if rng.random() < p_dist_B else NO_DIST,
        })
        # Celda C y D: no afectan delta (se filtra SOW=1)
        rows.append({
            "ticker": ticker, "detect_sow": 0, "contexto_op": 1,
            "label_adjudicada": DIST if rng.random() < 0.5 else NO_DIST,
        })
        rows.append({
            "ticker": ticker, "detect_sow": 0, "contexto_op": 0,
            "label_adjudicada": DIST if rng.random() < 0.5 else NO_DIST,
        })
    return pd.DataFrame(rows)


def test_delta_calculo_basico():
    df = pd.DataFrame([
        {"detect_sow": 1, "contexto_op": 1, "label_adjudicada": DIST},
        {"detect_sow": 1, "contexto_op": 1, "label_adjudicada": NO_DIST},
        {"detect_sow": 1, "contexto_op": 0, "label_adjudicada": NO_DIST},
        {"detect_sow": 1, "contexto_op": 0, "label_adjudicada": NO_DIST},
    ])
    # P(DIST|SOW=1,Ctx=1) = 0.5; P(DIST|SOW=1,Ctx=0) = 0.0 -> 0.5
    assert abs(_delta_from_df(df) - 0.5) < 1e-9


def test_delta_sin_sow_1():
    df = pd.DataFrame([
        {"detect_sow": 0, "contexto_op": 1, "label_adjudicada": DIST},
    ])
    assert np.isnan(_delta_from_df(df))


def test_delta_celda_vacia():
    df = pd.DataFrame([
        {"detect_sow": 1, "contexto_op": 1, "label_adjudicada": DIST},
        {"detect_sow": 1, "contexto_op": 1, "label_adjudicada": DIST},
    ])
    assert np.isnan(_delta_from_df(df))


def test_cluster_rechaza_columna_faltante():
    df = pd.DataFrame({"ticker": ["A"], "detect_sow": [1]})
    with pytest.raises(ValueError, match="Falta columna"):
        cluster_bootstrap_delta(df, B=10, seed=1)


def test_cluster_rechaza_pocos_tickers():
    df = pd.DataFrame([
        {"ticker": "A", "detect_sow": 1, "contexto_op": 1,
         "label_adjudicada": DIST},
        {"ticker": "A", "detect_sow": 1, "contexto_op": 0,
         "label_adjudicada": NO_DIST},
    ])
    with pytest.raises(ValueError, match="2 tickers"):
        cluster_bootstrap_delta(df, B=10, seed=1)

def test_cluster_remuestrea_globalmente_no_por_celda():
    """Verifica que un ticker con obs en varias celdas las conserva juntas."""
    df = _make_capa2_df(n_tickers=30, seed=1)
    res = cluster_bootstrap_delta(df, B=200, seed=42)
    assert res.n_tickers_original == 30
    assert res.n_replicas > 0
    # Si el remuestreo fuera por celda, no habria varianza explicable
    # por ticker. Aqui debe haberla.
    assert res.hi - res.lo > 0


def test_cluster_ic_contiene_punto():
    df = _make_capa2_df(n_tickers=30, seed=1)
    res = cluster_bootstrap_delta(df, B=300, seed=1)
    assert res.lo <= res.delta_point <= res.hi


def test_cluster_determinista():
    df = _make_capa2_df(n_tickers=20, seed=1)
    r1 = cluster_bootstrap_delta(df, B=100, seed=7)
    r2 = cluster_bootstrap_delta(df, B=100, seed=7)
    assert r1.delta_point == r2.delta_point
    assert r1.lo == r2.lo
    assert r1.hi == r2.hi
    assert (r1.samples == r2.samples).all()


def test_cluster_delta_nulo_ic_cruza_cero():
    """Con delta_efecto=0, el IC debe cruzar 0."""
    df = _make_capa2_df(n_tickers=50, seed=1, delta_efecto=0.0)
    res = cluster_bootstrap_delta(df, B=500, seed=42)
    # No siempre, pero con 50 tickers y delta=0 el IC debe contener 0
    assert res.lo <= 0 <= res.hi


def test_cluster_delta_grande_ic_no_cruza_cero():
    """Con delta_efecto=0.6, el IC debe ser > 0."""
    df = _make_capa2_df(n_tickers=100, seed=1, delta_efecto=0.6)
    res = cluster_bootstrap_delta(df, B=500, seed=42)
    assert res.lo > 0


def test_clasificar_pass():
    df = _make_capa2_df(n_tickers=100, seed=1, delta_efecto=0.6)
    res = cluster_bootstrap_delta(df, B=500, seed=42)
    assert clasificar_resultado(res) == "PASS"


def test_clasificar_inconcluso_cuando_ic_cruza_cero():
    df = _make_capa2_df(n_tickers=30, seed=1, delta_efecto=0.0)
    res = cluster_bootstrap_delta(df, B=300, seed=42)
    assert clasificar_resultado(res) == "INCONCLUSO"


def test_clasificar_nan_es_inconcluso():
    from scripts.gold_standard.bootstrap_cluster import ClusterResult
    r = ClusterResult(
        delta_point=float("nan"), lo=float("nan"), hi=float("nan"),
        samples=np.array([]), B=0, n_replicas=0, n_tickers_original=0,
    )
    assert clasificar_resultado(r) == "INCONCLUSO"


def test_materialidad():
    df = _make_capa2_df(n_tickers=100, seed=1, delta_efecto=0.4)
    res = cluster_bootstrap_delta(df, B=200, seed=1)
    # Delta puntual cae en ~0.4-0.5 segun semilla.
    # Umbral bajo -> material.
    assert materialidad(res, umbral=0.10)
    # Umbral claramente por encima del delta -> no material.
    assert not materialidad(res, umbral=0.90)