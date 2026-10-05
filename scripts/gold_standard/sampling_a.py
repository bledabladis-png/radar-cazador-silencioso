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
    calendar_map: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Anade sector, periodo y pos_sesion a los episodios del frame.

    Tickers sin mapping sectorial reciben 'UNKNOWN'.
    NO se descartan. El universo se conserva completo.

    calendar_map: dict {ticker: {fecha: pos_ordinal}}.
        Si se pasa, se usa para computar pos_sesion (posicion ordinal
        dentro del calendario completo del ticker). Necesario para que
        apply_min_gap mida SESIONES reales, no posiciones dentro del
        subconjunto filtrado.
    """
    df = episodes.copy()
    df["sector"] = df["ticker"].map(sector_map).fillna("UNKNOWN")
    df["year"] = pd.to_datetime(df["t"]).dt.year
    df["periodo"] = df["year"].apply(assign_periodo)
    if calendar_map is not None:
        df["pos_sesion"] = [
            calendar_map.get(tk, {}).get(t, -1)
            for tk, t in zip(df["ticker"], df["t"])
        ]
    return df


def build_calendar_map(dataset: pd.DataFrame) -> dict[str, dict]:
    """dict {ticker: {fecha: pos_ordinal}} sobre las filas validas.

    Usa build_ticker_df para consistencia con el contrato del detector.
    """
    from indicators.wyckoff import build_ticker_df
    out = {}
    tickers = sorted(dataset.columns.get_level_values(1).unique())
    for tk in tickers:
        try:
            tdf = build_ticker_df(dataset, tk)
        except KeyError:
            continue
        out[tk] = {t: i for i, t in enumerate(tdf.index)}
    return out


def apply_min_gap(
    episodes: pd.DataFrame,
    min_gap_sessions: int = MIN_GAP_A_SESSIONS,
    seed: int = SEED_GLOBAL,
    pos_col: str | None = "pos_sesion",
) -> pd.DataFrame:
    """Aplica min_gap por ticker con aceptacion aleatoria determinista.

    Unidad: SESIONES (posicion ordinal en el calendario completo del
    ticker). Dictamen P0.4 + hallazgo 2026-10-05.

    Usa columna pos_col si existe (posicion en el calendario completo).
    Si no, fallback a la posicion dentro del subconjunto (solo valido
    cuando episodes NO esta filtrado por clase).

    Hallazgo 2026-10-05: sin pos_col, min_gap=5 sobre 40 positivos
    filtrados de 1200 filas deja solo 7 en vez de ~40, porque la
    posicion se mide dentro del subconjunto de 14 elementos, no del
    calendario completo.
    """
    rng = np.random.default_rng(seed)
    accepted_indices = []
    for tk, group in episodes.groupby("ticker", sort=False):
        g = group.sort_values("t")
        idx_original = g.index.to_numpy()
        n = len(g)
        if pos_col is not None and pos_col in g.columns:
            positions = g[pos_col].to_numpy()
        else:
            positions = np.arange(n)
        order = rng.permutation(n)
        accepted_pos = []
        for i in order:
            p_cur = int(positions[i])
            if all(
                abs(p_cur - int(positions[j])) >= min_gap_sessions
                for j in accepted_pos
            ):
                accepted_pos.append(i)
        for i in accepted_pos:
            accepted_indices.append(idx_original[i])
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
    strict: bool = True,
) -> pd.DataFrame:
    """Muestra A enriquecida 200+200.

    excluir: DataFrame con columnas ticker, t. Esos (ticker, t) se
    descartan ANTES de muestrear. Sirve para garantizar A ∩ B = ∅
    cuando B se muestrea primero (dictamen P0.3).

    strict: si True (default), exige n_pos/n_neg disponibles o falla.
            Si False (preflight), usa min(pedido, disponible) y reporta
            cuantos hay realmente. No falla por falta de positivos.
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

    # min_gap POR CLASE, no sobre el conjunto.
    # Hallazgo 2026-10-05 (preflight): aplicar min_gap al conjunto
    # elimina 88% de positivos porque los negativos dominan y desplazan
    # a los positivos en el greedy aleatorio. Medido:
    #   min_gap por clase: 2559 -> 298 positivos
    #   min_gap conjunto:  2559 -> 12 positivos
    pos = apply_min_gap(
        df[df["detect_sow"] == 1], min_gap_sessions, seed,
    )
    neg = apply_min_gap(
        df[df["detect_sow"] == 0], min_gap_sessions, seed + 1,
    )
    rng = np.random.default_rng(seed)

    if strict:
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
    else:
        n_pos = min(n_pos, len(pos))
        n_neg = min(n_neg, len(neg))

    pos_sel = _sample_stratified(pos, n_pos, rng)
    neg_sel = _sample_stratified(neg, n_neg, rng)
    out = pd.concat([pos_sel, neg_sel], axis=0).reset_index(drop=True)
    out["muestra"] = "A"
    return out