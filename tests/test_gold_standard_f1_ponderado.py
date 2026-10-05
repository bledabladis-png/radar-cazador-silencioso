# -*- coding: utf-8 -*-
"""Tests F1 ponderado directo (P1.1, dictamen).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.gold_standard.metrics_simple import simple_metrics
from scripts.gold_standard.metrics_weighted import (
    confusion_weighted,
    to_confusion_int,
    weighted_metrics,
)


def test_f1_ponderado_no_usa_redondeo():
    """Con pesos fraccionarios, F1_operacional != F1_redondeado."""
    y_ref = np.array([1, 1, 0, 0])
    y_det = np.array([1, 0, 1, 0])
    w = np.array([1.5, 2.5, 1.5, 2.5])
    wc = confusion_weighted(y_ref, y_det, w)
    f1_operacional = weighted_metrics(wc)["f1"]
    f1_redondeado = simple_metrics(to_confusion_int(wc))["f1"]
    assert not np.isclose(f1_operacional, f1_redondeado)


def test_f1_weighted_exacto_con_pesos_unitarios():
    y_ref = np.array([1, 1, 0, 0])
    y_det = np.array([1, 0, 0, 0])
    w = np.ones(4)
    wc = confusion_weighted(y_ref, y_det, w)
    m = weighted_metrics(wc)
    assert np.isclose(m["f1"], 2.0 / 3.0)


def test_weighted_metrics_usa_floats():
    """Los pesos de confusion son float, no int."""
    y_ref = np.array([1, 1, 0, 0])
    y_det = np.array([1, 0, 1, 0])
    w = np.array([1.7, 2.3, 1.7, 2.3])
    wc = confusion_weighted(y_ref, y_det, w)
    m = weighted_metrics(wc)
    assert isinstance(m["w_tp"], float)