# -*- coding: utf-8 -*-
"""Cluster bootstrap global de ticker para Capa 2.

Especificacion (protocolo v7, seccion 5.6):
  - Cluster = ticker. NO se remuestrea por celda.
  - Cada ticker arrastra TODAS sus observaciones (A/B/C/D juntas).
  - La estructura A/B/C/D se conserva como clasificacion.
  - Delta = P(DIST | SOW=1, Ctx=1) - P(DIST | SOW=1, Ctx=0).
  - IC95% = percentiles 2.5 y 97.5 de las B replicas.

Naturaleza: cluster bootstrap de dependencia intra-ticker,
no bootstrap design-based de encuesta.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ClusterResult:
    delta_point: float
    lo: float
    hi: float
    samples: np.ndarray
    B: int
    n_replicas: int               # replicas con Delta definido
    n_tickers_original: int
    n_boot_invalid: int = 0       # replicas NaN (celdas sin soporte)
    invalid_rate: float = 0.0     # n_boot_invalid / B


def _delta_from_df(
    df: pd.DataFrame,
    sow_col: str = "detect_sow",
    ctx_col: str = "contexto_op",
    dist_col: str = "label_adjudicada",
    dist_label: str = "DISTRIBUTIVE_CONTEXT",
) -> float:
    """Delta = P(DIST|SOW=1,Ctx=1) - P(DIST|SOW=1,Ctx=0)."""
    sub = df[df[sow_col] == 1]
    if len(sub) == 0:
        return float("nan")
    m1 = sub[sub[ctx_col] == 1]
    m0 = sub[sub[ctx_col] == 0]
    if len(m1) == 0 or len(m0) == 0:
        return float("nan")
    p1 = float((m1[dist_col] == dist_label).mean())
    p0 = float((m0[dist_col] == dist_label).mean())
    return p1 - p0

def cluster_bootstrap_delta(
    df: pd.DataFrame,
    B: int = 2000,
    seed: int = 0,
    ticker_col: str = "ticker",
    sow_col: str = "detect_sow",
    ctx_col: str = "contexto_op",
    dist_col: str = "label_adjudicada",
    dist_label: str = "DISTRIBUTIVE_CONTEXT",
) -> ClusterResult:
    """Cluster bootstrap global de ticker.

    df: DataFrame con columnas ticker, sow_col, ctx_col, dist_col.
    Cada replica remuestrea tickers con reemplazo (mismo tamano
    que el numero de tickers unicos) y arrastra TODAS las filas
    de los tickers seleccionados.
    """
    for c in (ticker_col, sow_col, ctx_col, dist_col):
        if c not in df.columns:
            raise ValueError(f"Falta columna {c}")
    if len(df) == 0:
        raise ValueError("DataFrame vacio")

    # Preflight de soporte (dictamen P0.6).
    # Cada una de las dos celdas de SOW=1 debe tener K tickers distintos.
    K_SOPORTE_MIN = 20
    sub_sow1 = df[df[sow_col] == 1]
    tickers_c1 = sub_sow1[sub_sow1[ctx_col] == 1][ticker_col].nunique()
    tickers_c0 = sub_sow1[sub_sow1[ctx_col] == 0][ticker_col].nunique()
    if tickers_c1 < K_SOPORTE_MIN or tickers_c0 < K_SOPORTE_MIN:
        raise ValueError(
            f"Soporte insuficiente: tickers SOW=1,Ctx=1 = {tickers_c1}, "
            f"SOW=1,Ctx=0 = {tickers_c0}. "
            f"Se requieren >= {K_SOPORTE_MIN} en cada celda."
        )

    tickers = df[ticker_col].unique()
    n_tickers = len(tickers)
    if n_tickers < 2:
        raise ValueError("Se requieren >= 2 tickers para cluster bootstrap")

    # Agrupar por ticker una vez (evita filtros repetidos)
    grupos: dict = {}
    for tk, sub in df.groupby(ticker_col, sort=False):
        grupos[tk] = sub.reset_index(drop=True)

    delta_point = _delta_from_df(df, sow_col, ctx_col, dist_col, dist_label)

    rng = np.random.default_rng(seed)
    samples = np.full(B, np.nan, dtype=float)
    for b in range(B):
        picks = rng.choice(tickers, size=n_tickers, replace=True)
        parts = [grupos[tk] for tk in picks]
        replica = pd.concat(parts, ignore_index=True)
        samples[b] = _delta_from_df(
            replica, sow_col, ctx_col, dist_col, dist_label,
        )

    invalid_mask = np.isnan(samples)
    n_invalid = int(invalid_mask.sum())
    valid = samples[~invalid_mask]
    invalid_rate = n_invalid / B if B > 0 else 0.0

    if len(valid) == 0:
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(valid, [2.5, 97.5])

    return ClusterResult(
        delta_point=delta_point,
        lo=float(lo), hi=float(hi),
        samples=samples, B=B, n_replicas=len(valid),
        n_tickers_original=n_tickers,
        n_boot_invalid=n_invalid,
        invalid_rate=invalid_rate,
    )

def clasificar_resultado(res: ClusterResult) -> str:
    """Clasifica segun protocolo v7, seccion 4.8.

    PASS: LCI > 0
    EVIDENCIA_EN_CONTRA: UCI < 0
    INCONCLUSO: IC cruza 0
    BLOQUEADO_INVALID: invalid_rate > 0 (dictamen P0.6)
    """
    if res.invalid_rate > 0:
        return "BLOQUEADO_INVALID"
    if np.isnan(res.lo) or np.isnan(res.hi):
        return "INCONCLUSO"
    if res.lo > 0:
        return "PASS"
    if res.hi < 0:
        return "EVIDENCIA_EN_CONTRA"
    return "INCONCLUSO"


def materialidad(res: ClusterResult, umbral: float = 0.10) -> bool:
    """Delta >= umbral (criterio separado, no gate)."""
    return res.delta_point >= umbral