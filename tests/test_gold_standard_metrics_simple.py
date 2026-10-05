# -*- coding: utf-8 -*-
"""Tests metricas simples: Wilson, confusion, Se/Sp/PPV/NPV/F1.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.metrics_simple import (
    Confusion,
    confusion_from_labels,
    format_metrics,
    metrics_with_ci,
    simple_metrics,
    wilson_ci,
)


def test_wilson_ci_caso_conocido():
    # k=5, n=10 -> p=0.5, IC95 aproximadamente (0.237, 0.763)
    lo, hi = wilson_ci(5, 10)
    assert abs(lo - 0.2366) < 0.005
    assert abs(hi - 0.7634) < 0.005


def test_wilson_ci_extremos():
    lo, hi = wilson_ci(0, 10)
    assert abs(lo - 0.0) < 1e-6
    assert hi > 0 and hi < 0.35
    lo2, hi2 = wilson_ci(10, 10)
    assert abs(hi2 - 1.0) < 1e-6
    assert lo2 > 0.65 and lo2 < 1.0


def test_wilson_ci_vacio():
    lo, hi = wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_confusion_acuerdo_perfecto():
    y_ref = np.array([1, 1, 0, 0])
    y_det = np.array([1, 1, 0, 0])
    c = confusion_from_labels(y_ref, y_det)
    assert c.tp == 2 and c.tn == 2 and c.fp == 0 and c.fn == 0


def test_confusion_longitud_distinta():
    with pytest.raises(ValueError, match="distinta longitud"):
        confusion_from_labels(np.array([0, 1]), np.array([0]))

def _conf(tp, fp, tn, fn):
    return Confusion(tp=tp, fp=fp, tn=tn, fn=fn)


def test_simple_metrics_valores_conocidos():
    c = _conf(tp=80, fp=20, tn=160, fn=40)
    m = simple_metrics(c)
    assert abs(m["se"] - 80 / 120) < 1e-9
    assert abs(m["sp"] - 160 / 180) < 1e-9
    assert abs(m["ppv"] - 80 / 100) < 1e-9
    assert abs(m["npv"] - 160 / 200) < 1e-9


def test_simple_metrics_f1_armonica():
    c = _conf(tp=50, fp=50, tn=50, fn=50)
    m = simple_metrics(c)
    # se=0.5, ppv=0.5 -> f1=0.5
    assert abs(m["f1"] - 0.5) < 1e-9


def test_simple_metrics_division_por_cero():
    c = _conf(tp=0, fp=0, tn=0, fn=0)
    m = simple_metrics(c)
    assert np.isnan(m["se"])
    assert np.isnan(m["sp"])
    assert np.isnan(m["f1"])


def test_simple_metrics_denominador_cero_parcial():
    c = _conf(tp=0, fp=0, tn=50, fn=50)
    m = simple_metrics(c)
    assert m["se"] == 0.0
    assert m["sp"] == 1.0
    assert np.isnan(m["ppv"])


def test_metrics_with_ci_devuelve_intervalos():
    c = _conf(tp=80, fp=20, tn=160, fn=40)
    m = metrics_with_ci(c)
    for k in ("se_ci", "sp_ci", "ppv_ci", "npv_ci"):
        assert k in m
        lo, hi = m[k]
        assert 0.0 <= lo <= hi <= 1.0


def test_metrics_with_ci_contiene_puntual():
    c = _conf(tp=80, fp=20, tn=160, fn=40)
    m = metrics_with_ci(c)
    for k in ("se", "sp", "ppv", "npv"):
        lo, hi = m[f"{k}_ci"]
        v = m[k]
        assert lo - 1e-9 <= v <= hi + 1e-9


def test_format_metrics_basico():
    c = _conf(tp=80, fp=20, tn=160, fn=40)
    m = simple_metrics(c)
    s = format_metrics(m)
    assert "se=" in s and "sp=" in s and "f1=" in s


def test_determinista():
    c = _conf(tp=10, fp=5, tn=20, fn=3)
    assert simple_metrics(c) == simple_metrics(c)
    assert metrics_with_ci(c) == metrics_with_ci(c)