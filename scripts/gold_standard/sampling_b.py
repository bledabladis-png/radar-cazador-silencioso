# -*- coding: utf-8 -*-
"""Muestra B — representativa del universo elegible.

Estratificada por sector x periodo (11 x 3 = 33 estratos).
Sin condicionamiento por detector. Sin min_gap.
Pesos conocidos w_h = N_h / n_h.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.gold_standard.constants import (
    N_MIN_ESTRATO,
    N_MIN_RAO_WU,
    SEED_GLOBAL,
)


def build_estratos(frame_meta: pd.DataFrame) -> pd.DataFrame:
    """Tabla de estratos con N_h inicial."""
    counts = (
        frame_meta.groupby(["sector", "periodo"], dropna=False)
        .size()
        .reset_index(name="N_h")
    )
    return counts


def collapse_estratos(
    estratos: pd.DataFrame,
    n_min: int = N_MIN_ESTRATO,
) -> pd.DataFrame:
    """Regla determinista de colapso. Solo por periodo (P1.5, dictamen).

    Reglas, en orden:
      1. Mantener sector x periodo si todas las celdas tienen
         N_h >= n_min.
      2. Si alguna celda < n_min, colapsar TODOS los periodos de ese
         sector a "global" (no se fusionan sectores distintos).
      3. Si tras el colapso un sector sigue con N_h < n_min, se
         reclasifica ese sector como "OTHER" (categoria predefinida).
         No se fusiona con otro sector.

    Nunca se fusionan sectores distintos.
    Regla determinista, congelada antes del muestreo.
    """
    df = estratos.copy()
    if df.empty:
        return df

    df["sector_eff"] = df["sector"]
    df["periodo_eff"] = df["periodo"]

    # Regla 1-2: por cada sector, si alguna celda < n_min, todo a global
    celda = df.groupby(["sector_eff", "periodo_eff"])["N_h"].sum()
    sectores_colapsar = set()
    for (s, _), n_h in celda.items():
        if n_h < n_min:
            sectores_colapsar.add(s)
    for s in sectores_colapsar:
        mask = df["sector_eff"] == s
        df.loc[mask, "periodo_eff"] = "global"

    # Regla 3: si un sector completo sigue < n_min tras colapsar, OTHER
    totales_sector = df.groupby(["sector_eff", "periodo_eff"])["N_h"].sum()
    sectores_a_other = set()
    for (s, _), n_h in totales_sector.items():
        if n_h < n_min:
            sectores_a_other.add(s)
    for s in sectores_a_other:
        mask = df["sector_eff"] == s
        df.loc[mask, "sector_eff"] = "OTHER"

    # Reagregar (puede haber colisiones tras el remap)
    out = (
        df.groupby(["sector_eff", "periodo_eff"], dropna=False)["N_h"]
        .sum()
        .reset_index()
        .rename(columns={"sector_eff": "sector", "periodo_eff": "periodo"})
    )
    return out



def proportional_allocation(
    estratos: pd.DataFrame,
    n_total: int,
) -> pd.DataFrame:
    """Reparto proporcional con minimo Rao-Wu-Yue."""
    df = estratos.copy()
    if df.empty or n_total <= 0:
        df["n_h"] = 0
        return df
    total = df["N_h"].sum()
    df["n_h"] = np.floor(df["N_h"] / total * n_total).astype(int)
    # garantizar n_h >= N_MIN_RAO_WU donde N_h >= N_MIN_RAO_WU
    df["n_h"] = df.apply(
        lambda r: max(r["n_h"], N_MIN_RAO_WU)
        if r["N_h"] >= N_MIN_RAO_WU
        else r["n_h"],
        axis=1,
    )
    # ajustar para sumar exactamente n_total
    diff = n_total - df["n_h"].sum()
    if diff != 0:
        order = df["N_h"].sort_values(ascending=False).index
        i = 0
        while diff != 0 and i < len(order) * 10:
            idx = order[i % len(order)]
            if diff > 0 and df.at[idx, "n_h"] < df.at[idx, "N_h"]:
                df.at[idx, "n_h"] += 1
                diff -= 1
            elif diff < 0 and df.at[idx, "n_h"] > N_MIN_RAO_WU:
                df.at[idx, "n_h"] -= 1
                diff += 1
            i += 1
    # no superar N_h
    df["n_h"] = df[["n_h", "N_h"]].min(axis=1)
    return df


def sample_b(
    frame_meta: pd.DataFrame,
    n_B: int,
    estratos_finales: pd.DataFrame,
    seed: int = SEED_GLOBAL,
) -> pd.DataFrame:
    """Muestreo aleatorio dentro de cada estrato.

    frame_meta: DataFrame con ticker, t, sector, periodo.
    n_B: tamano total de la muestra.
    estratos_finales: salida de collapse_estratos con columnas
                      sector, periodo, N_h.
    """
    estratos_asign = proportional_allocation(estratos_finales, n_B)
    rng = np.random.default_rng(seed)

    parts = []
    for _, row in estratos_asign.iterrows():
        if row["n_h"] <= 0:
            continue
        sub = frame_meta[
            (frame_meta["sector"] == row["sector"])
            & (frame_meta["periodo"] == row["periodo"])
        ]
        if sub.empty:
            continue
        n_h = int(min(row["n_h"], len(sub)))
        idx = rng.choice(sub.index.to_numpy(), size=n_h, replace=False)
        part = sub.loc[idx].copy()
        part["estrato_sector"] = row["sector"]
        part["estrato_periodo"] = row["periodo"]
        part["N_h"] = int(row["N_h"])
        part["n_h"] = n_h
        part["w_h"] = row["N_h"] / n_h
        parts.append(part)

    if not parts:
        raise ValueError("Muestra B vacia: ningun estrato seleccionable")
    out = pd.concat(parts, axis=0).reset_index(drop=True)
    out["muestra"] = "B"
    return out