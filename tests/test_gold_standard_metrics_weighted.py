# -*- coding: utf-8 -*-
"""Tests metricas ponderadas (Hajek).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.metrics_weighted import (
    confusion_weighted,
    hajek_ratio,
    to_confusion_int,
    weighted_metrics_full,
)


def test_hajek_pesos_uniforme_igual_media():
    num = np.array([1, 1, 0, 0])
    den = np.array([1, 1, 1, 1])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    assert abs(hajek_ratio(num, den, w) - 0.5) < 1e-9


def test_hajek_pesos_desiguales():
    # num: 2 unos en los primeros; den: todos
    # w = [10, 10, 1, 1]
    # num_w = 20; den_w = 22 -> ratio = 20/22
    num = np.array([1, 1, 0, 0])
    den = np.array([1, 1, 1, 1])
    w = np.array([10.0, 10.0, 1.0, 1.0])
    assert abs(hajek_ratio(num, den, w) - 20.0 / 22.0) < 1e-9


def test_hajek_longitud_distinta():
    with pytest.raises(ValueError):
        hajek_ratio(
            np.array([1, 1]), np.array([1, 1, 1]), np.array([1.0, 1.0])
        )


def test_confusion_weighted_pesos_uniformes():
    y_ref = np.array([1, 1, 0, 0])
    y_det = np.array([1, 0, 0, 1])
    w = np.array([1.0, 1.0, 1.0, 1.0])
    wc = confusion_weighted(y_ref, y_det, w)
    assert abs(wc.tp - 1.0) < 1e-9
    assert abs(wc.fn - 1.0) < 1e-9
    assert abs(wc.tn - 1.0) < 1e-9
    assert abs(wc.fp - 1.0) < 1e-9
    assert abs(wc.sum_w - 4.0) < 1e-9

def test_weighted_metrics_valores_calculables():
    # tp=80, fp=20, tn=160, fn=40 con pesos unitarios
    y_ref = np.concatenate([np.ones(120, dtype=int), np.zeros(180, dtype=int)])
    y_det = np.concatenate([
        np.concatenate([np.ones(80, dtype=int), np.zeros(40, dtype=int)]),
        np.concatenate([np.ones(20, dtype=int), np.zeros(160, dtype=int)]),
    ])
    w = np.ones(len(y_ref))
    out = weighted_metrics_full(y_ref, y_det, w)
    assert abs(out["se"] - 80 / 120) < 1e-9
    assert abs(out["sp"] - 160 / 180) < 1e-9
    assert abs(out["ppv"] - 80 / 100) < 1e-9
    assert abs(out["npv"] - 160 / 200) < 1e-9
    assert abs(out["ref_prevalence"] - 120 / 300) < 1e-9
    assert abs(out["detector_prevalence"] - 100 / 300) < 1e-9


def test_weighted_prevalences_pesos_desiguales():
    # tp=0.0, fp=0.0, tn=0.0, fn=0.0
    y_ref = np.array([1, 1, 0, 0])
    y_det = np.array([0, 0, 0, 0])
    w = np.array([3.0, 7.0, 1.0, 1.0])
    out = weighted_metrics_full(y_ref, y_det, w)
    # ref_prevalence = (0 + 10) / 12
    assert abs(out["ref_prevalence"] - 10.0 / 12.0) < 1e-9


def test_weighted_division_por_cero():
    y_ref = np.array([1, 1, 0, 0])
    y_det = np.array([0, 0, 0, 0])
    w = np.ones(4)
    out = weighted_metrics_full(y_ref, y_det, w)
    assert out["se"] == 0.0
    assert out["sp"] == 1.0
    assert np.isnan(out["ppv"])


def test_to_confusion_int_redondea():
    y_ref = np.array([1, 1, 0, 0])
    y_det = np.array([1, 0, 0, 1])
    w = np.array([1.5, 2.5, 1.0, 1.0])
    wc = confusion_weighted(y_ref, y_det, w)
    c = to_confusion_int(wc)
    # Pesos: tp=1.5, fn=2.5, tn=1.0, fp=1.0
    # int(round(x)) usa banker's rounding de Python:
    #   round(1.5)=2, round(2.5)=2, round(1.0)=1
    assert c.tp == 2
    assert c.fn == 2
    assert c.tn == 1
    assert c.fp == 1


def test_weighted_metrics_sin_pesos_cambia_estimador():
    # Los pesos deben cambiar el resultado
    y_ref = np.array([1, 1, 0, 0])
    y_det = np.array([1, 0, 1, 0])
    w_uniform = np.ones(4)
    w_skewed = np.array([10.0, 1.0, 1.0, 1.0])
    u = weighted_metrics_full(y_ref, y_det, w_uniform)
    s = weighted_metrics_full(y_ref, y_det, w_skewed)
    assert abs(u["se"] - s["se"]) > 1e-3


def test_determinista():
    y_ref = np.array([1, 0, 1, 0, 1])
    y_det = np.array([1, 0, 0, 1, 1])
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert weighted_metrics_full(y_ref, y_det, w) == weighted_metrics_full(y_ref, y_det, w)