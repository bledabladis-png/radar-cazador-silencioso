# -*- coding: utf-8 -*-
"""Metricas ponderadas por diseno (Hajek) para Muestra B.

Especificacion (protocolo v7, secciones 2.5, 3.7):
  - PPV, NPV, Se, Sp ponderadas por pesos de inclusion w_i.
  - reference_prevalence y detector_prevalence ponderadas.
  - Este modulo NO calcula pesos. Solo los aplica.
  - IC: delegado a bootstrap_rwy.py (Rao-Wu-Yue rescaled).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.gold_standard.metrics_simple import Confusion


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def hajek_ratio(
    num_mask: np.ndarray,
    den_mask: np.ndarray,
    w: np.ndarray,
) -> float:
    """Estimador Hajek: sum(w*num) / sum(w*den).

    num_mask y den_mask booleanos, misma longitud que w.
    """
    if not (len(num_mask) == len(den_mask) == len(w)):
        raise ValueError("Longitudes distintas")
    num = float((w * num_mask.astype(float)).sum())
    den = float((w * den_mask.astype(float)).sum())
    return _safe_div(num, den)

@dataclass
class WeightedConfusion:
    tp: float
    fp: float
    tn: float
    fn: float
    sum_w: float


def confusion_weighted(
    y_ref: np.ndarray,
    y_det: np.ndarray,
    w: np.ndarray,
    pos_label: int = 1,
) -> WeightedConfusion:
    """Matriz de confusion con pesos de diseno.

    tp = sum(w * [ref=1 & det=1])
    fp = sum(w * [ref=0 & det=1])
    tn = sum(w * [ref=0 & det=0])
    fn = sum(w * [ref=1 & det=0])
    sum_w = sum(w)  # util para prevalencias
    """
    if not (len(y_ref) == len(y_det) == len(w)):
        raise ValueError("Longitudes distintas")
    ref = (y_ref == pos_label)
    det = (y_det == pos_label)
    wf = w.astype(float)
    tp = float((wf * (ref & det).astype(float)).sum())
    fp = float((wf * (~ref & det).astype(float)).sum())
    tn = float((wf * (~ref & ~det).astype(float)).sum())
    fn = float((wf * (ref & ~det).astype(float)).sum())
    return WeightedConfusion(
        tp=tp, fp=fp, tn=tn, fn=fn, sum_w=float(wf.sum()),
    )

def weighted_metrics(wc: WeightedConfusion) -> dict:
    """Se, Sp, PPV, NPV ponderadas. Denominadores ponderados."""
    n_pos_ref = wc.tp + wc.fn
    n_neg_ref = wc.tn + wc.fp
    n_pos_pred = wc.tp + wc.fp
    n_neg_pred = wc.tn + wc.fn
    se = _safe_div(wc.tp, n_pos_ref)
    sp = _safe_div(wc.tn, n_neg_ref)
    ppv = _safe_div(wc.tp, n_pos_pred)
    npv = _safe_div(wc.tn, n_neg_pred)
    precision = ppv
    recall = se
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "se": se, "sp": sp, "ppv": ppv, "npv": npv,
        "precision": precision, "recall": recall, "f1": f1,
        "sum_w": wc.sum_w,
        "w_tp": wc.tp, "w_fp": wc.fp, "w_tn": wc.tn, "w_fn": wc.fn,
    }


def weighted_prevalences(wc: WeightedConfusion) -> dict:
    """Prevalencia ponderada del reference y del detector."""
    return {
        "ref_prevalence": _safe_div(wc.tp + wc.fn, wc.sum_w),
        "detector_prevalence": _safe_div(wc.tp + wc.fp, wc.sum_w),
    }


def weighted_metrics_full(
    y_ref: np.ndarray,
    y_det: np.ndarray,
    w: np.ndarray,
    pos_label: int = 1,
) -> dict:
    """Combina confusion, metricas y prevalencias en un dict."""
    wc = confusion_weighted(y_ref, y_det, w, pos_label=pos_label)
    out = weighted_metrics(wc)
    out.update(weighted_prevalences(wc))
    return out


def to_confusion_int(wc: WeightedConfusion) -> Confusion:
    """Convierte ponderada a entera (para reportes descriptivos)."""
    return Confusion(
        tp=int(round(wc.tp)),
        fp=int(round(wc.fp)),
        tn=int(round(wc.tn)),
        fn=int(round(wc.fn)),
    )