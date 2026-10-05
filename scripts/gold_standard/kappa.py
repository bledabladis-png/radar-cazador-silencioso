# -*- coding: utf-8 -*-
"""Fleiss kappa y Cohen kappa para el panel de anotadores.

Especificacion (protocolo v7, secciones 3.7, 6):
  - Fleiss kappa principal sobre casos con 3 votos binarios validos.
  - technical_ineligible se trata como missing, NO como categoria.
  - Cohen kappa par a par en casos con 2 votos validos.
  - Umbral: kappa >= 0.60 (Landis-Koch: substantial agreement).

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KappaResult:
    fleiss_kappa: float
    fleiss_n: int
    cohen_pairwise: dict
    n_total: int
    n_3_validos: int


def encode_labels(
    series: pd.Series,
    valid_labels: tuple[str, ...],
) -> np.ndarray:
    """Convierte etiquetas str a codigos int. Missing -> -1."""
    label_map = {lab: i for i, lab in enumerate(valid_labels)}
    out = np.full(len(series), -1, dtype=int)
    for i, v in enumerate(series):
        if v is None or pd.isna(v):
            continue
        s = str(v).strip()
        if s in label_map:
            out[i] = label_map[s]
    return out

def fleiss_kappa(matrix: np.ndarray, n_categorias: int | None = None) -> float:
    """Fleiss kappa sobre matriz (n_items, n_raters) de codigos int.

    Todos los valores deben ser validos (>=0). Sin missing.
    Devuelve kappa. Si P_exp >= 1.0, devuelve 1.0 (acuerdo trivial).
    """
    if matrix.size == 0 or matrix.shape[1] < 2:
        return float("nan")
    n, m = matrix.shape
    if n_categorias is None:
        n_categorias = int(matrix.max()) + 1

    counts = np.zeros((n, n_categorias), dtype=int)
    for i in range(n):
        for j in range(n_categorias):
            counts[i, j] = int((matrix[i] == j).sum())

    # P_i = (1/(m*(m-1))) * (sum_j n_ij^2 - m)
    p_i = (counts.astype(float) ** 2).sum(axis=1) - m
    p_i = p_i / (m * (m - 1))
    p_obs = p_i.mean()

    # p_j = sum_i n_ij / (n*m)
    p_j = counts.sum(axis=0).astype(float) / (n * m)
    p_exp = float((p_j ** 2).sum())

    if p_exp >= 1.0:
        return 1.0
    return float((p_obs - p_exp) / (1.0 - p_exp))


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen kappa par a par. Sin missing. Misma longitud."""
    if len(a) != len(b):
        raise ValueError("a y b con distinta longitud")
    n = len(a)
    if n == 0:
        return float("nan")
    p_obs = float((a == b).mean())
    cats = np.unique(np.concatenate([a, b]))
    p_exp = 0.0
    for c in cats:
        pa = float((a == c).sum()) / n
        pb = float((b == c).sum()) / n
        p_exp += pa * pb
    if p_exp >= 1.0:
        return 1.0
    return float((p_obs - p_exp) / (1.0 - p_exp))

def kappa_report(
    df: pd.DataFrame,
    valid_labels: tuple[str, ...],
    label_cols: tuple[str, ...] = ("anotador_1", "anotador_2", "anotador_3"),
) -> KappaResult:
    """Calcula Fleiss (3 votos) y Cohen par a par (2 votos).

    df: DataFrame con columnas label_cols.
    valid_labels: etiquetas aceptadas (p.ej. ('SOW','NO_SOW')).
    """
    for c in label_cols:
        if c not in df.columns:
            raise ValueError(f"Falta columna {c}")

    encoded = np.full((len(df), len(label_cols)), -1, dtype=int)
    for j, c in enumerate(label_cols):
        encoded[:, j] = encode_labels(df[c], valid_labels)

    # Fleiss: solo filas con todos los 3 validos
    mask_3 = (encoded >= 0).all(axis=1)
    matrix_3 = encoded[mask_3]
    fleiss = fleiss_kappa(matrix_3, n_categorias=len(valid_labels))

    # Cohen pairwise: pares con ambos validos
    pares = [(0, 1), (0, 2), (1, 2)]
    cohen = {}
    for i, j in pares:
        m = (encoded[:, i] >= 0) & (encoded[:, j] >= 0)
        if m.sum() == 0:
            cohen[(i, j)] = {"kappa": float("nan"), "n": 0}
            continue
        k = cohen_kappa(encoded[m, i], encoded[m, j])
        cohen[(i, j)] = {"kappa": k, "n": int(m.sum())}

    return KappaResult(
        fleiss_kappa=fleiss,
        fleiss_n=int(mask_3.sum()),
        cohen_pairwise=cohen,
        n_total=len(df),
        n_3_validos=int(mask_3.sum()),
    )