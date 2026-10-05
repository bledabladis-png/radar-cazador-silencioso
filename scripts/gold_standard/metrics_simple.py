# -*- coding: utf-8 -*-
"""Metricas simples: proporciones, Wilson CI, matriz de confusion.

Especificacion (protocolo v7, secciones 2.5, 3.7):
  - Proporciones simples en celda homogenea -> Wilson CI.
  - Metricas ponderadas por diseno -> metrics_weighted.py (RWY).
  - Este modulo NO calcula pesos. Solo cuenta.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Z_95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval para proporcion k/n.

    Devuelve (lo, hi) a nivel z (default 95%).
    Maneja n=0 -> (nan, nan).
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))) / denom
    return (float(center - half), float(center + half))

@dataclass
class Confusion:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def n_pos_ref(self) -> int:
        return self.tp + self.fn

    @property
    def n_neg_ref(self) -> int:
        return self.tn + self.fp

    @property
    def n_pos_pred(self) -> int:
        return self.tp + self.fp

    @property
    def n_neg_pred(self) -> int:
        return self.tn + self.fn


def confusion_from_labels(
    y_ref: np.ndarray,
    y_det: np.ndarray,
    pos_label: int = 1,
) -> Confusion:
    """Construye matriz de confusion. y_ref, y_det en {0, 1}."""
    if len(y_ref) != len(y_det):
        raise ValueError("y_ref y y_det con distinta longitud")
    tp = int(((y_ref == pos_label) & (y_det == pos_label)).sum())
    fp = int(((y_ref != pos_label) & (y_det == pos_label)).sum())
    tn = int(((y_ref != pos_label) & (y_det != pos_label)).sum())
    fn = int(((y_ref == pos_label) & (y_det != pos_label)).sum())
    return Confusion(tp=tp, fp=fp, tn=tn, fn=fn)


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else float("nan")


def simple_metrics(c: Confusion) -> dict:
    """Se, Sp, PPV, NPV, Precision, Recall, F1 (sin IC)."""
    se = _safe_div(c.tp, c.n_pos_ref)
    sp = _safe_div(c.tn, c.n_neg_ref)
    ppv = _safe_div(c.tp, c.n_pos_pred)
    npv = _safe_div(c.tn, c.n_neg_pred)
    precision = ppv
    recall = se
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "se": se, "sp": sp, "ppv": ppv, "npv": npv,
        "precision": precision, "recall": recall, "f1": f1,
        "n": c.n,
    }

def metrics_with_ci(c: Confusion, z: float = Z_95) -> dict:
    """Se, Sp, PPV, NPV, Precision, Recall con Wilson CI."""
    out = simple_metrics(c)

    def _ci(k: int, n: int) -> tuple[float, float]:
        return wilson_ci(k, n, z=z)

    out["se_ci"] = _ci(c.tp, c.n_pos_ref)
    out["sp_ci"] = _ci(c.tn, c.n_neg_ref)
    out["ppv_ci"] = _ci(c.tp, c.n_pos_pred)
    out["npv_ci"] = _ci(c.tn, c.n_neg_pred)
    out["precision_ci"] = out["ppv_ci"]
    out["recall_ci"] = out["se_ci"]
    # F1 CI: Wilson sobre proporcion no es correcto para F1.
    # Dejamos F1 sin IC (se reporta puntual).
    return out


def format_metrics(m: dict, precision_digits: int = 3) -> str:
    """Formato compacto para logs/reportes."""
    keys = ("se", "sp", "ppv", "npv", "f1")
    parts = []
    for k in keys:
        v = m.get(k, float("nan"))
        if np.isnan(v):
            parts.append(f"{k}=nan")
        else:
            parts.append(f"{k}={v:.{precision_digits}f}")
    return " ".join(parts)