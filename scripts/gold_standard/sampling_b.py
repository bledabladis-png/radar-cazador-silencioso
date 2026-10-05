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
    """Regla determinista de colapso.

    1. Colapsar por sector (fusionar el sector menos poblado con el mas
       proximo en tamano total).
    2. Si aun hay celdas < n_min, colapsar por periodo.
    3. Regla reproducible.
    """
    df = estratos.copy()

    # Totales por sector
    sect_tot = df.groupby("sector")["N_h"].sum().sort_values()
    sector_remap = {s: s for s in sect_tot.index}

    while True:
        df["sector_eff"] = df["sector"].map(sector_remap)
        cell = df.groupby(["sector_eff", "periodo"])["N_h"].sum()
        small = cell[cell < n_min]
        if small.empty:
            break
        # elegir el sector efectivo mas pequeno globalmente y fusionarlo
        sect_small = (
            df.groupby("sector_eff")["N_h"].sum().sort_values()
        )
        if len(sect_small) <= 1:
            break
        smallest = sect_small.index[0]
        # fusionar con el proximo
        rest = sect_small.drop(smallest)
        if rest.empty:
            break
        target = rest.index[0]
        for s, t in sector_remap.items():
            if t == smallest:
                sector_remap[s] = target
        # evitar bucle infinito
        if sector_remap.get(smallest) == smallest:
            break

    df["sector_eff"] = df["sector"].map(sector_remap)

    # Colapso por periodo si aun hay celdas pequenas
    df["periodo_eff"] = df["periodo"]
    cell2 = df.groupby(["sector_eff", "periodo_eff"])["N_h"].sum()
    small2 = cell2[cell2 < n_min]
    if not small2.empty:
        # colapsar todos los periodos a "global"
        df["periodo_eff"] = "global"

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