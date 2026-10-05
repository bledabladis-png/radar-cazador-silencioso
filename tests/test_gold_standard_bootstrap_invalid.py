# -*- coding: utf-8 -*-
"""Tests invalid_rate y preflight de soporte (P0.6, dictamen).

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
    ClusterResult,
    clasificar_resultado,
    cluster_bootstrap_delta,
)

DIST = "DISTRIBUTIVE_CONTEXT"
NO_DIST = "NO_DISTRIBUTIVE_CONTEXT"


def _make_capa2_df(n_tickers=50, seed=1, delta_efecto=0.20):
    rng = np.random.default_rng(seed)
    rows = []
    for tk in range(n_tickers):
        ticker = f"T{tk:03d}"
        p_dist_A = 0.5 + delta_efecto / 2
        p_dist_B = 0.5 - delta_efecto / 2
        rows.append({
            "ticker": ticker, "detect_sow": 1, "contexto_op": 1,
            "label_adjudicada": DIST if rng.random() < p_dist_A else NO_DIST,
        })
        rows.append({
            "ticker": ticker, "detect_sow": 1, "contexto_op": 0,
            "label_adjudicada": DIST if rng.random() < p_dist_B else NO_DIST,
        })
        rows.append({
            "ticker": ticker, "detect_sow": 0, "contexto_op": 1,
            "label_adjudicada": DIST if rng.random() < 0.5 else NO_DIST,
        })
        rows.append({
            "ticker": ticker, "detect_sow": 0, "contexto_op": 0,
            "label_adjudicada": DIST if rng.random() < 0.5 else NO_DIST,
        })
    return pd.DataFrame(rows)


def test_preflight_falla_si_pocos_tickers_celda():
    """Menos de K=20 tickers en una celda -> ValueError."""
    df = _make_capa2_df(n_tickers=10, seed=1)  # 10 < 20
    with pytest.raises(ValueError, match="Soporte insuficiente"):
        cluster_bootstrap_delta(df, B=50, seed=1)


def test_preflight_pasa_con_suficientes_tickers():
    df = _make_capa2_df(n_tickers=50, seed=1)
    res = cluster_bootstrap_delta(df, B=50, seed=1)
    assert isinstance(res, ClusterResult)


def test_invalid_rate_cero_en_caso_normal():
    df = _make_capa2_df(n_tickers=50, seed=1, delta_efecto=0.2)
    res = cluster_bootstrap_delta(df, B=100, seed=1)
    assert res.invalid_rate == 0.0
    assert res.n_boot_invalid == 0
    assert res.n_replicas == 100


def test_clasificar_bloqueado_si_invalid_rate_positivo():
    r = ClusterResult(
        delta_point=0.1, lo=0.05, hi=0.15, samples=np.array([0.1]),
        B=100, n_replicas=99, n_tickers_original=50,
        n_boot_invalid=1, invalid_rate=0.01,
    )
    assert clasificar_resultado(r) == "BLOQUEADO_INVALID"


def test_clasificar_pass_cuando_invalid_rate_cero():
    r = ClusterResult(
        delta_point=0.1, lo=0.05, hi=0.15, samples=np.array([0.1]),
        B=100, n_replicas=100, n_tickers_original=50,
        n_boot_invalid=0, invalid_rate=0.0,
    )
    assert clasificar_resultado(r) == "PASS"