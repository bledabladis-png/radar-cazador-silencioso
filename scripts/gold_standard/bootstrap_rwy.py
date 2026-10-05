# -*- coding: utf-8 -*-
"""Rao-Wu-Yue rescaled bootstrap para metricas ponderadas.

Especificacion (protocolo v7, seccion 2.5):
  m_h    = n_h - 1
  f_h    = n_h / N_h
  lambda_h = sqrt(m_h * (1 - f_h) / (n_h - 1))
  r_hi*  = multiplicidad de seleccion
  w_hi*  = [1 - lambda_h + lambda_h * (n_h/m_h) * r_hi*] * w_hi

  Requiere n_h >= 2 en todos los estratos.

  IC95% = percentiles 2.5 y 97.5 de las B replicas.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from scripts.gold_standard.constants import N_MIN_RAO_WU
from scripts.gold_standard.seeds import replica_seeds


def _lambda_h(n_h: int, N_h: int) -> float:
    """Factor de escala Rao-Wu-Yue."""
    m_h = n_h - 1
    f_h = n_h / N_h
    return float(np.sqrt(m_h * (1.0 - f_h) / (n_h - 1)))


def _multiplicidades(n_h: int, rng: np.random.Generator) -> np.ndarray:
    """Remuestrea m_h = n_h - 1 indices con reemplazo.

    Devuelve array (n_h,) con r_hi* para cada unidad original.
    """
    m_h = n_h - 1
    picks = rng.integers(0, n_h, size=m_h)
    r = np.zeros(n_h, dtype=int)
    for p in picks:
        r[p] += 1
    return r

@dataclass
class RWYResult:
    point: float
    lo: float
    hi: float
    samples: np.ndarray
    n_replicas: int
    B: int


MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def rao_wu_yue_ci(
    y_ref: np.ndarray,
    y_det: np.ndarray,
    w_orig: np.ndarray,
    estrato_id: np.ndarray,
    N_h_map: dict[int, int],
    metric_fn: MetricFn,
    B: int = 2000,
    seed: int = 0,
) -> RWYResult:
    """Bootstrap Rao-Wu-Yue sobre un diseno estratificado.

    y_ref, y_det, w_orig, estrato_id: arrays de misma longitud.
    N_h_map: dict {estrato_id: N_h}.
    metric_fn(y_ref, y_det, w) -> float.
    """
    n = len(y_ref)
    if not (len(y_det) == len(w_orig) == len(estrato_id) == n):
        raise ValueError("Longitudes distintas")
    if n == 0:
        raise ValueError("Sin datos")

    # Indexar por estrato
    estratos = np.unique(estrato_id)
    idx_por_estrato: dict[int, np.ndarray] = {}
    n_h_map: dict[int, int] = {}
    N_h_check: dict[int, int] = {}
    for h in estratos:
        idx = np.where(estrato_id == h)[0]
        n_h = len(idx)
        if h not in N_h_map:
            raise ValueError(f"Falta N_h para estrato {h}")
        N_h = int(N_h_map[h])
        if N_h < n_h:
            raise ValueError(f"N_h < n_h en estrato {h}")
        if n_h < N_MIN_RAO_WU:
            raise ValueError(f"n_h < {N_MIN_RAO_WU} en estrato {h}")
        idx_por_estrato[int(h)] = idx
        n_h_map[int(h)] = n_h
        N_h_check[int(h)] = N_h

    # Punto
    point = float(metric_fn(y_ref, y_det, w_orig))

    # Replicas. Seeds deterministas por replica (P1.3).
    seeds = replica_seeds(seed, B)
    samples = np.full(B, np.nan, dtype=float)
    for b in range(B):
        rng = np.random.default_rng(seeds[b])
        w_replica = np.zeros(n, dtype=float)
        for h in estratos:
            h_int = int(h)
            idx = idx_por_estrato[h_int]
            n_h = n_h_map[h_int]
            N_h = N_h_check[h_int]
            lam = _lambda_h(n_h, N_h)
            r = _multiplicidades(n_h, rng)
            m_h = n_h - 1
            factor = 1.0 - lam + lam * (n_h / m_h) * r
            w_replica[idx] = factor * w_orig[idx]
        samples[b] = float(metric_fn(y_ref, y_det, w_replica))

    valid = samples[~np.isnan(samples)]
    if len(valid) == 0:
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(valid, [2.5, 97.5])
    return RWYResult(
        point=point, lo=float(lo), hi=float(hi),
        samples=samples, n_replicas=len(valid), B=B,
    )

# Wrappers de conveniencia. Toman (y_ref, y_det, w) y devuelven la metrica.

def ppv_metric(y_ref: np.ndarray, y_det: np.ndarray, w: np.ndarray) -> float:
    from scripts.gold_standard.metrics_weighted import (
        confusion_weighted, weighted_metrics,
    )
    wc = confusion_weighted(y_ref, y_det, w)
    return weighted_metrics(wc)["ppv"]


def npv_metric(y_ref: np.ndarray, y_det: np.ndarray, w: np.ndarray) -> float:
    from scripts.gold_standard.metrics_weighted import (
        confusion_weighted, weighted_metrics,
    )
    wc = confusion_weighted(y_ref, y_det, w)
    return weighted_metrics(wc)["npv"]


def se_metric(y_ref: np.ndarray, y_det: np.ndarray, w: np.ndarray) -> float:
    from scripts.gold_standard.metrics_weighted import (
        confusion_weighted, weighted_metrics,
    )
    wc = confusion_weighted(y_ref, y_det, w)
    return weighted_metrics(wc)["se"]


def sp_metric(y_ref: np.ndarray, y_det: np.ndarray, w: np.ndarray) -> float:
    from scripts.gold_standard.metrics_weighted import (
        confusion_weighted, weighted_metrics,
    )
    wc = confusion_weighted(y_ref, y_det, w)
    return weighted_metrics(wc)["sp"]


def ref_prevalence_metric(
    y_ref: np.ndarray, y_det: np.ndarray, w: np.ndarray,
) -> float:
    from scripts.gold_standard.metrics_weighted import (
        confusion_weighted, weighted_prevalences,
    )
    wc = confusion_weighted(y_ref, y_det, w)
    return weighted_prevalences(wc)["ref_prevalence"]


def detector_prevalence_metric(
    y_ref: np.ndarray, y_det: np.ndarray, w: np.ndarray,
) -> float:
    from scripts.gold_standard.metrics_weighted import (
        confusion_weighted, weighted_prevalences,
    )
    wc = confusion_weighted(y_ref, y_det, w)
    return weighted_prevalences(wc)["detector_prevalence"]


METRICAS_DEFAULT = {
    "ppv": ppv_metric,
    "npv": npv_metric,
    "se": se_metric,
    "sp": sp_metric,
    "ref_prevalence": ref_prevalence_metric,
    "detector_prevalence": detector_prevalence_metric,
}


def bootstrap_all_metrics(
    y_ref: np.ndarray,
    y_det: np.ndarray,
    w_orig: np.ndarray,
    estrato_id: np.ndarray,
    N_h_map: dict[int, int],
    B: int = 2000,
    seed: int = 0,
) -> dict[str, RWYResult]:
    """Bootstrap RWY para todas las metricas de METRICAS_DEFAULT."""
    out = {}
    for nombre, fn in METRICAS_DEFAULT.items():
        # Cada metrica usa la misma semilla para comparabilidad.
        res = rao_wu_yue_ci(
            y_ref, y_det, w_orig, estrato_id, N_h_map,
            metric_fn=fn, B=B, seed=seed,
        )
        out[nombre] = res
    return out