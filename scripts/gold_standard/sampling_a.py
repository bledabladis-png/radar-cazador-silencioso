# -*- coding: utf-8 -*-
"""Muestra A — enriquecida 200+200, min_gap=120.

Muestra A es descriptiva/enriquecida. NO produce estimaciones
poblacionales (Se, Sp, F1, PPV, NPV). Esas se estiman sobre Muestra B.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.gold_standard.constants import (
    N_POS_A,
    N_NEG_A,
    MIN_GAP_A_SESSIONS,
    SEED_GLOBAL,
    PERIODOS,
)


def assign_periodo(year: int) -> str:
    """Mapea un anio a uno de los 3 periodos del estudio."""
    for y0, y1 in PERIODOS:
        if y0 <= year <= y1:
            return f"{y0}-{y1}"
    raise ValueError(f"Anio fuera de rango: {year}")


def build_metadata(
    episodes: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Anade sector y periodo a los episodios del frame.

    Tickers sin mapping sectorial reciben 'UNKNOWN'.
    NO se descartan. El universo se conserva completo.
    """
    df = episodes.copy()
    df["sector"] = df["ticker"].map(sector_map).fillna("UNKNOWN")
    df["year"] = pd.to_datetime(df["t"]).dt.year
    df["periodo"] = df["year"].apply(assign_periodo)
    return df


def apply_min_gap(
    episodes: pd.DataFrame,
    min_gap_sessions: int = MIN_GAP_A_SESSIONS,
    seed: int = SEED_GLOBAL,
) -> pd.DataFrame:
    """Aplica min_gap por ticker con aceptacion aleatoria determinista.

    Unidad: SESIONES (posicion en el calendario del ticker), no dias
    naturales. Dictamen P0.4.

    120 sesiones pueden corresponder a ~168 dias naturales con fines
    de semana y festivos. La condicion se aplica sobre la distancia
    ordinal entre sesiones del mismo ticker.
    """
    rng = np.random.default_rng(seed)
    accepted_indices = []
    for tk, group in episodes.groupby("ticker", sort=False):
        g = group.sort_values("t")
        idx_original = g.index.to_numpy()
        n = len(g)
        order = rng.permutation(n)
        accepted_pos = []
        for pos in order:
            if all(abs(int(pos) - int(p)) >= min_gap_sessions
                   for p in accepted_pos):
                accepted_pos.append(int(pos))
        for pos in accepted_pos:
            accepted_indices.append(idx_original[pos])
    return episodes.loc[sorted(accepted_indices)].copy()


def _stratified_quota(
    df: pd.DataFrame,
    n_total: int,
    strata_cols: tuple[str, ...] = ("sector", "periodo"),
) -> pd.Series:
    """Reparto proporcional por estrato. Ajusta para sumar n_total."""
    counts = df.groupby(list(strata_cols), dropna=False).size()
    if counts.sum() == 0:
        return pd.Series(dtype=int)
    quota = (counts / counts.sum() * n_total).astype(int)
    diff = n_total - quota.sum()
    if diff > 0:
        order = counts.sort_values(ascending=False).index
        for i in range(diff):
            quota[order[i % len(order)]] += 1
    elif diff < 0:
        order = quota.sort_values(ascending=False).index
        for i in range(-diff):
            if quota[order[i % len(order)]] > 0:
                quota[order[i % len(order)]] -= 1
    quota = quota.combine(counts, min)
    deficit = n_total - quota.sum()
    if deficit > 0:
        spare = counts - quota
        order = spare.sort_values(ascending=False).index
        for i in range(deficit):
            quota[order[i % len(order)]] += 1
    return quota


def _sample_stratified(
    df: pd.DataFrame,
    n_total: int,
    rng: np.random.Generator,
    strata_cols: tuple[str, ...] = ("sector", "periodo"),
) -> pd.DataFrame:
    """Muestreo aleatorio dentro de cada estrato segun cuota."""
    if df.empty or n_total <= 0:
        return df.iloc[0:0].copy()
    quota = _stratified_quota(df, n_total, strata_cols)
    selected = []
    for key, n_h in quota.items():
        if n_h <= 0:
            continue
        sub = df
        for col, val in zip(strata_cols, key):
            sub = sub[sub[col] == val]
        if len(sub) <= n_h:
            selected.append(sub)
        else:
            idx = rng.choice(sub.index.to_numpy(), size=n_h, replace=False)
            selected.append(sub.loc[idx])
    if not selected:
        return df.iloc[0:0].copy()
    return pd.concat(selected, axis=0)


def sample_a(
    episodes_meta: pd.DataFrame,
    detector_flags: pd.Series,
    n_pos: int = N_POS_A,
    n_neg: int = N_NEG_A,
    min_gap_sessions: int = MIN_GAP_A_SESSIONS,
    seed: int = SEED_GLOBAL,
    excluir: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Muestra A enriquecida 200+200.

    excluir: DataFrame con columnas ticker, t. Esos (ticker, t) se
    descartan ANTES de muestrear. Sirve para garantizar A ∩ B = ∅
    cuando B se muestrea primero (dictamen P0.3).
    """
    df = episodes_meta.copy()
    if excluir is not None and len(excluir) > 0:
        clave_excluir = set(zip(excluir["ticker"], excluir["t"]))
        mask = [
            (tk, t) not in clave_excluir
            for tk, t in zip(df["ticker"], df["t"])
        ]
        df = df[mask].copy()

    df = df.set_index(["ticker", "t"])
    df["detect_sow"] = detector_flags.reindex(df.index).fillna(0).astype(int)
    df = df.reset_index()

    df_gap = apply_min_gap(df, min_gap_sessions, seed)
    rng = np.random.default_rng(seed)

    pos = df_gap[df_gap["detect_sow"] == 1]
    neg = df_gap[df_gap["detect_sow"] == 0]

    if len(pos) < n_pos:
        raise ValueError(
            f"Muestra A: solo {len(pos)} positivos tras min_gap, "
            f"se requieren {n_pos}"
        )
    if len(neg) < n_neg:
        raise ValueError(
            f"Muestra A: solo {len(neg)} negativos tras min_gap, "
            f"se requieren {n_neg}"
        )

    pos_sel = _sample_stratified(pos, n_pos, rng)
    neg_sel = _sample_stratified(neg, n_neg, rng)
    out = pd.concat([pos_sel, neg_sel], axis=0).reset_index(drop=True)
    out["muestra"] = "A"
    return out