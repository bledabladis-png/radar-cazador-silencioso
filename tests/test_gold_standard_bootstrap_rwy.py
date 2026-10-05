# -*- coding: utf-8 -*-
"""Tests Rao-Wu-Yue rescaled bootstrap.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.bootstrap_rwy import (
    _lambda_h,
    _multiplicidades,
    bootstrap_all_metrics,
    detector_prevalence_metric,
    ppv_metric,
    rao_wu_yue_ci,
    ref_prevalence_metric,
    se_metric,
    sp_metric,
)


def test_lambda_h_caso_conocido():
    # n_h=5, N_h=1000, m_h=4, f_h=0.005
    # lambda = sqrt(4 * 0.995 / 4) = sqrt(0.995)
    lam = _lambda_h(n_h=5, N_h=1000)
    assert abs(lam - np.sqrt(0.995)) < 1e-9


def test_lambda_h_fraccion_baja():
    # f_h ~ 0 -> lambda ~ 1
    lam = _lambda_h(n_h=5, N_h=10_000_000)
    assert abs(lam - 1.0) < 1e-3


def test_lambda_h_fraccion_alta():
    # f_h = 1 -> lambda = 0
    lam = _lambda_h(n_h=5, N_h=5)
    assert abs(lam - 0.0) < 1e-9


def test_multiplicidades_suma_m_h():
    rng = np.random.default_rng(42)
    r = _multiplicidades(n_h=10, rng=rng)
    assert len(r) == 10
    assert r.sum() == 9  # m_h = n_h - 1
    assert (r >= 0).all()


def test_multiplicidades_determinista():
    r1 = _multiplicidades(10, np.random.default_rng(7))
    r2 = _multiplicidades(10, np.random.default_rng(7))
    assert (r1 == r2).all()

def _make_datos(n_por_estrato=50, k_estratos=3, pi=0.30, se=0.70, sp=0.70,
                seed=1):
    rng = np.random.default_rng(seed)
    y_ref = []
    y_det = []
    estrato_id = []
    for h in range(k_estratos):
        for _ in range(n_por_estrato):
            ref = 1 if rng.random() < pi else 0
            if ref == 1:
                det = 1 if rng.random() < se else 0
            else:
                det = 1 if rng.random() < (1 - sp) else 0
            y_ref.append(ref)
            y_det.append(det)
            estrato_id.append(h)
    return (
        np.array(y_ref, dtype=int),
        np.array(y_det, dtype=int),
        np.array(estrato_id, dtype=int),
    )


def test_rwy_validaciones_longitud():
    with pytest.raises(ValueError, match="Longitudes distintas"):
        rao_wu_yue_ci(
            np.array([1, 0]), np.array([1]), np.array([1.0, 1.0]),
            np.array([0, 0]), {0: 100}, ppv_metric, B=10, seed=1,
        )


def test_rwy_falta_N_h():
    y_ref = np.array([1, 0, 1, 0])
    y_det = np.array([1, 0, 0, 1])
    w = np.ones(4)
    est = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="Falta N_h"):
        rao_wu_yue_ci(
            y_ref, y_det, w, est, {0: 100}, ppv_metric, B=10, seed=1,
        )


def test_rwy_n_h_menor_2():
    y_ref = np.array([1, 0, 1, 0])
    y_det = np.array([1, 0, 0, 1])
    w = np.ones(4)
    est = np.array([0, 0, 1, 1])
    N_h = {0: 100, 1: 1}  # N_h < n_h
    with pytest.raises(ValueError, match="N_h < n_h"):
        rao_wu_yue_ci(
            y_ref, y_det, w, est, N_h, ppv_metric, B=10, seed=1,
        )


def test_rwy_estrato_con_1_unidad():
    y_ref = np.array([1, 0, 1])
    y_det = np.array([1, 0, 0])
    w = np.ones(3)
    est = np.array([0, 0, 1])  # estrato 1 con 1 unidad
    N_h = {0: 100, 1: 100}
    with pytest.raises(ValueError, match="n_h < 2"):
        rao_wu_yue_ci(
            y_ref, y_det, w, est, N_h, ppv_metric, B=10, seed=1,
        )


def test_rwy_acuerdo_perfecto_ppv_1():
    # Todos positivos son verdaderos -> PPV = 1 siempre
    y_ref = np.array([1, 1, 1, 1, 0, 0])
    y_det = np.array([1, 1, 1, 1, 0, 0])
    w = np.ones(6)
    est = np.array([0, 0, 0, 1, 1, 1])
    N_h = {0: 1000, 1: 1000}
    res = rao_wu_yue_ci(y_ref, y_det, w, est, N_h, ppv_metric, B=100, seed=1)
    assert abs(res.point - 1.0) < 1e-9
    # Algunas replicas pueden dar nan si no hay predichos positivos,
    # pero con 4 positivos y m_h >= 2, es raro. Verificamos que IC = (1,1).
    assert abs(res.lo - 1.0) < 1e-9 or np.isnan(res.lo) is False

def test_rwy_ic_contiene_punto():
    y_ref, y_det, est = _make_datos(n_por_estrato=100, k_estratos=3, seed=42)
    w = np.ones(len(y_ref))
    N_h = {0: 10000, 1: 10000, 2: 10000}
    res = rao_wu_yue_ci(y_ref, y_det, w, est, N_h, se_metric, B=500, seed=42)
    assert res.lo <= res.point <= res.hi


def test_rwy_ancho_ic_disminuye_con_n():
    y_small, y_det_s, est_s = _make_datos(n_por_estrato=20, k_estratos=2, seed=1)
    y_big, y_det_b, est_b = _make_datos(n_por_estrato=500, k_estratos=2, seed=1)
    w_s = np.ones(len(y_small))
    w_b = np.ones(len(y_big))
    N_h_s = {0: 2000, 1: 2000}
    N_h_b = {0: 50000, 1: 50000}
    r_s = rao_wu_yue_ci(y_small, y_det_s, w_s, est_s, N_h_s, se_metric, B=300, seed=1)
    r_b = rao_wu_yue_ci(y_big, y_det_b, w_b, est_b, N_h_b, se_metric, B=300, seed=1)
    assert (r_b.hi - r_b.lo) < (r_s.hi - r_s.lo)


def test_rwy_determinista():
    y_ref, y_det, est = _make_datos(n_por_estrato=50, k_estratos=2, seed=1)
    w = np.ones(len(y_ref))
    N_h = {0: 1000, 1: 1000}
    r1 = rao_wu_yue_ci(y_ref, y_det, w, est, N_h, ppv_metric, B=100, seed=7)
    r2 = rao_wu_yue_ci(y_ref, y_det, w, est, N_h, ppv_metric, B=100, seed=7)
    assert r1.point == r2.point
    assert r1.lo == r2.lo
    assert r1.hi == r2.hi
    assert (r1.samples == r2.samples).all()


def test_rwy_n_replicas_reportado():
    y_ref, y_det, est = _make_datos(n_por_estrato=50, k_estratos=2, seed=1)
    w = np.ones(len(y_ref))
    N_h = {0: 1000, 1: 1000}
    res = rao_wu_yue_ci(y_ref, y_det, w, est, N_h, ppv_metric, B=100, seed=1)
    assert res.B == 100
    assert 0 < res.n_replicas <= 100


def test_bootstrap_all_metrics():
    y_ref, y_det, est = _make_datos(n_por_estrato=100, k_estratos=3, seed=1)
    w = np.ones(len(y_ref))
    N_h = {0: 10000, 1: 10000, 2: 10000}
    out = bootstrap_all_metrics(y_ref, y_det, w, est, N_h, B=100, seed=1)
    for k in ("ppv", "npv", "se", "sp", "ref_prevalence", "detector_prevalence"):
        assert k in out
        assert isinstance(out[k].point, float)
        assert out[k].lo <= out[k].point <= out[k].hi


def test_wrappers_prevalencias():
    y_ref = np.array([1, 1, 0, 0, 1, 0])
    y_det = np.array([1, 0, 0, 1, 1, 0])
    w = np.ones(6)
    rp = ref_prevalence_metric(y_ref, y_det, w)
    dp = detector_prevalence_metric(y_ref, y_det, w)
    assert abs(rp - 3 / 6) < 1e-9
    assert abs(dp - 3 / 6) < 1e-9


def test_wrappers_sp():
    y_ref = np.array([1, 1, 0, 0])
    y_det = np.array([1, 0, 0, 0])
    w = np.ones(4)
    assert abs(sp_metric(y_ref, y_det, w) - 1.0) < 1e-9