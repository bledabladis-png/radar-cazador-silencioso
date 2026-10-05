# -*- coding: utf-8 -*-
"""Muestreo B v8: 4 celdas + SRSWOR + Hamilton + censos.

Protocolo v8, secciones 2, 4, 5.

Celdas:
    C1 = (D=1, ctx=1)
    C2 = (D=1, ctx=0)
    C3 = (D=0, ctx=1)
    C4 = (D=0, ctx=0)

Diseno: SRSWOR dentro de cada estrato.
n_h >= 5 o censo. Reparto por Hamilton hasta K=200 por celda.

BORRADOR — pendiente firma auditor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_MIN_H = 5
K_CELDA = 200
CELDAS = ("C1", "C2", "C3", "C4")


def compute_ctx_for_meta(
    meta: pd.DataFrame,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Anade columna 'ctx' a meta calculando ctx por ticker.

    meta: DataFrame con ticker, t.
    dataset: MultiIndex (field, ticker).
    """
    from indicators.wyckoff import build_ticker_df
    from scripts.gold_standard.ctx import compute_ctx

    df = meta.copy()
    cache = {}
    for tk in df["ticker"].unique():
        try:
            tdf = build_ticker_df(dataset, tk)
        except KeyError:
            continue
        cache[tk] = compute_ctx(tdf)

    ctx_vals = []
    for tk, t in zip(df["ticker"], df["t"]):
        serie = cache.get(tk)
        if serie is None:
            ctx_vals.append(False)
            continue
        try:
            ctx_vals.append(bool(serie.loc[t]))
        except KeyError:
            ctx_vals.append(False)
    df["ctx"] = ctx_vals
    return df


def asignar_celdas(
    meta: pd.DataFrame,
    detect_col: str = "detect_sow",
    ctx_col: str = "ctx",
) -> pd.DataFrame:
    """Anade columna 'celda' en {C1, C2, C3, C4}."""
    for c in (detect_col, ctx_col):
        if c not in meta.columns:
            raise ValueError(f"Falta columna {c}")
    df = meta.copy()
    d = df[detect_col].astype(int)
    c = df[ctx_col].astype(bool)
    celda = np.where(
        d == 1,
        np.where(c, "C1", "C2"),
        np.where(c, "C3", "C4"),
    )
    df["celda"] = celda
    return df


def build_estratos_v8(meta_celda: pd.DataFrame) -> pd.DataFrame:
    """Tabla de estratos con N_h por (celda, sector, periodo)."""
    counts = (
        meta_celda.groupby(["celda", "sector", "periodo"], dropna=False)
        .size()
        .reset_index(name="N_h")
    )
    return counts

def verificar_factibilidad_minimos(
    estratos: pd.DataFrame,
    K: int = K_CELDA,
) -> dict:
    """Paso 0: verificar que suma n_h_min <= K en cada celda.

    n_h_min = 5 si N_h >= 5, sino N_h (censo).
    """
    for celda in CELDAS:
        sub = estratos[estratos["celda"] == celda]
        if len(sub) == 0:
            continue
        n_min = np.where(sub["N_h"] >= N_MIN_H, N_MIN_H, sub["N_h"])
        n_min = np.minimum(n_min, sub["N_h"].values)
        total_min = int(n_min.sum())
        if total_min > K:
            return {
                "ok": False,
                "celda": celda,
                "suma_minimos": total_min,
                "K": K,
            }
    return {"ok": True, "K": K}


def _hamilton(cuotas: np.ndarray, total_target: int) -> np.ndarray:
    """Largest remainder / Hamilton. Devuelve enteros que suman total_target."""
    cuotas = np.asarray(cuotas, dtype=float)
    base = np.floor(cuotas).astype(int)
    rem = total_target - int(base.sum())
    if rem <= 0:
        # Si base excede target, recortar desde el mayor resto
        if rem == 0:
            return base
        # rem < 0: quitar a los de menor resto
        restos = cuotas - base
        order = np.argsort(restos)
        i = 0
        while rem < 0 and i < len(base):
            if base[order[i]] > 0:
                base[order[i]] -= 1
                rem += 1
            i += 1
        return base
    restos = cuotas - base
    order = np.argsort(-restos)
    for i in order[:rem]:
        base[i] += 1
    return base


def asignar_n_h(estratos: pd.DataFrame, K: int = K_CELDA) -> pd.DataFrame:
    """Aplica seccion 5: n_h >= 5 o censo, Hamilton hasta K por celda."""
    df = estratos.copy()
    df["n_h"] = 0
    for celda in CELDAS:
        mask = df["celda"] == celda
        sub = df[mask]
        if len(sub) == 0:
            continue
        N_h = sub["N_h"].to_numpy()
        # Minimos
        n_min = np.where(N_h >= N_MIN_H, N_MIN_H, N_h)
        n_min = np.minimum(n_min, N_h)
        R = K - int(n_min.sum())
        if R < 0:
            raise ValueError(f"Celda {celda}: minimos ({n_min.sum()}) exceden K={K}")
        if R == 0:
            df.loc[mask, "n_h"] = n_min
            continue
        # Remanente por Hamilton
        total_N = N_h.sum()
        if total_N == 0:
            continue
        proporcional = R * N_h / total_N
        extra = _hamilton(proporcional, R)
        df.loc[mask, "n_h"] = n_min + extra
    # Nunca exceder N_h
    df["n_h"] = df[["n_h", "N_h"]].min(axis=1)
    df["n_h"] = df["n_h"].astype(int)
    df["N_h"] = df["N_h"].astype(int)
    return df

def sample_b_v8(
    meta_celda: pd.DataFrame,
    estratos_con_n: pd.DataFrame,
    seed: int = 20261006,
) -> pd.DataFrame:
    """SRSWOR por estrato con n_h asignado.

    meta_celda: DataFrame con ticker, t, celda, sector, periodo.
    estratos_con_n: DataFrame con celda, sector, periodo, N_h, n_h.

    Devuelve las unidades seleccionadas con columnas adicionales:
        N_h, n_h, w_h = N_h/n_h.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for _, row in estratos_con_n.iterrows():
        celda = row["celda"]
        sector = row["sector"]
        periodo = row["periodo"]
        N_h = int(row["N_h"])
        n_h = int(row["n_h"])
        if n_h <= 0:
            continue
        mask = (
            (meta_celda["celda"] == celda)
            & (meta_celda["sector"] == sector)
            & (meta_celda["periodo"] == periodo)
        )
        sub = meta_celda[mask]
        if len(sub) == 0:
            continue
        take = min(n_h, len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        part = sub.loc[idx].copy()
        part["N_h"] = N_h
        part["n_h"] = n_h
        part["w_h"] = N_h / n_h
        parts.append(part)
    if not parts:
        raise ValueError("Muestra B v8 vacia: ningun estrato seleccionable")
    out = pd.concat(parts, axis=0).reset_index(drop=True)
    return out


def totales_por_celda(estratos_con_n: pd.DataFrame) -> dict:
    """Suma n_h y N_h por celda."""
    out = {}
    for celda in CELDAS:
        sub = estratos_con_n[estratos_con_n["celda"] == celda]
        out[celda] = {
            "n_h_total": int(sub["n_h"].sum()),
            "N_h_total": int(sub["N_h"].sum()),
            "n_estratos": int(len(sub)),
        }
    return out