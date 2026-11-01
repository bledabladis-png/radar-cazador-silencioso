# -*- coding: utf-8 -*-
"""FASE-A-IAE-PER-TICKER (2026-11-01): agregacion por ticker del 13F.

El pipeline contractual produce un delta por posicion
(filing_manager_cik, canonical_security, discretion_type). Este
modulo agrega ese delta por ticker del catalogo radar y anade
metricas de consenso entre gestores.

Filtros de anomalia (sin hardcode, sin lista de tickers):

  - pct_increase >= 0.99 -> ANOMALOUS_ONE_SIDED
    (todos los gestores aumentan. Caso tipico: IPO donde el universo
    entero declara posicion nueva al mismo tiempo.)

  - pct_increase <= 0.01 -> ANOMALOUS_ONE_SIDED
    (todos los gestores reducen. Caso tipico: spin-off donde el
    ticker viejo pierde las acciones que migran al nuevo ticker.)

  - pct_new >= 0.50 -> MOSTLY_NEW_POSITIONS
    (mas de la mitad de gestores declaran posicion nueva. Posible
    IPO o rebalanceo masivo.)

  - pct_exit >= 0.50 -> MOSTLY_EXITS
    (mas de la mitad de gestores salen. Posible spin-off o
    delisting.)

  pct_increase = n_increase / (n_increase + n_decrease). Si no hay
  ni increase ni decrease (todos new o todos exit), pct_increase es
  NaN, y el filtro de ratio no aplica; pero pct_new o pct_exit
  disparan y caen en su categoria.

Estos filtros capturan el evento sin conocer el evento. Si manana
hay otro IPO o spin-off, cae solo por la misma regla.

Metricas por ticker:
  n_managers      gestores con posicion (BOTH + NEW + EXIT)
  n_increase      gestores cuyo delta_shares > 0
  n_decrease      gestores cuyo delta_shares < 0
  n_new           gestores con shares_prev=0, shares_curr>0
  n_exit          gestores con shares_prev>0, shares_curr=0
  sum_delta       suma de delta_shares en acciones
  pct_increase    n_increase / (n_increase + n_decrease)
  pct_new         n_new / n_managers
  pct_exit        n_exit / n_managers
  filter_reason   None si pasa; razon si se excluye
"""
from __future__ import annotations

import pandas as pd


OBSERVABLE_STATUSES = ("BOTH", "NEW", "EXIT")

# Filtros de anomalia. Ratios (no lista de tickers, no hardcode).
PCT_INCREASE_HIGH = 0.99
PCT_INCREASE_LOW = 0.01
PCT_NEW_HIGH = 0.50
PCT_EXIT_HIGH = 0.50

FILTER_ANOMALOUS = "ANOMALOUS_ONE_SIDED"
FILTER_MOSTLY_NEW = "MOSTLY_NEW_POSITIONS"
FILTER_MOSTLY_EXIT = "MOSTLY_EXITS"

PER_TICKER_COLUMNS = (
    "ticker", "n_managers", "n_increase", "n_decrease", "n_new", "n_exit",
    "sum_delta", "pct_increase", "pct_new", "pct_exit", "filter_reason",
)


def _ticker_from_canonical(canonical):
    """Extrae ticker de 'equity:X'. Devuelve None si no aplica."""
    if canonical is None or not isinstance(canonical, str):
        return None
    prefix = "equity:"
    if canonical.startswith(prefix):
        t = canonical[len(prefix):].strip()
        return t or None
    return None


def _agg_per_manager(g):
    """Agrega un DataFrame (ya filtrado a un manager+ticker) a un dict.

    n_increase / n_decrease excluyen NEW y EXIT. Un gestor NEW
    (prev=0, curr>0) cumple numericamente curr > prev, pero no es
    "aumento" en el sentido de posicion existente. Se cuenta como
    n_new. Simetrico para EXIT. Asi pct_increase = increase /
    (increase + decrease) mide consenso entre gestores con posicion
    en AMBOS trimestres, no entre todos.
    """
    shares_prev = float(g["sshprnamt_previous"].sum())
    shares_curr = float(g["sshprnamt_current"].sum())
    delta = shares_curr - shares_prev
    has_prev = g["sshprnamt_previous"] > 0
    has_curr = g["sshprnamt_current"] > 0
    n_increase = int(((g["sshprnamt_current"] > g["sshprnamt_previous"]) & has_prev).sum())
    n_decrease = int(((g["sshprnamt_current"] < g["sshprnamt_previous"]) & has_curr).sum())
    n_new = int((~has_prev & has_curr).sum())
    n_exit = int((has_prev & ~has_curr).sum())
    return pd.Series({
        "n_increase": n_increase,
        "n_decrease": n_decrease,
        "n_new": n_new,
        "n_exit": n_exit,
        "sum_delta": delta,
    })


def aggregate_by_ticker(delta_df):
    """Agrega delta_shares por ticker.

    Args:
        delta_df: salida de compute_delta_shares. Debe tener columnas
            filing_manager_cik, canonical_security, sshprnamt_current,
            sshprnamt_previous, delta_shares, match_status.

    Returns:
        DataFrame con PER_TICKER_COLUMNS, ordenado por sum_delta
        descendente. Incluye los tickers con filter_reason != None
        (el consumidor los separa).
    """
    empty = pd.DataFrame(columns=list(PER_TICKER_COLUMNS))
    if delta_df is None or delta_df.empty:
        return empty

    df = delta_df[delta_df["match_status"].isin(OBSERVABLE_STATUSES)].copy()
    if df.empty:
        return empty

    df["ticker"] = df["canonical_security"].apply(_ticker_from_canonical)
    df = df.dropna(subset=["ticker"])
    if df.empty:
        return empty

    # Primero colapsar por (manager, ticker): un manager puede tener
    # varias filas (SOLE/DFND/OTR) para el mismo canonical_security.
    # Sumar antes de decidir "aumenta/reduce" evita sesgo por discretion.
    per_mgr = (
        df.groupby(["ticker", "filing_manager_cik"], as_index=False, dropna=False)
          .agg(sshprnamt_current=("sshprnamt_current", "sum"),
               sshprnamt_previous=("sshprnamt_previous", "sum"))
    )
    per_mgr["sshprnamt_current"] = pd.to_numeric(per_mgr["sshprnamt_current"], errors="coerce").fillna(0.0)
    per_mgr["sshprnamt_previous"] = pd.to_numeric(per_mgr["sshprnamt_previous"], errors="coerce").fillna(0.0)

    # Agregar por ticker
    agg = per_mgr.groupby("ticker").apply(_agg_per_manager, include_groups=False).reset_index()
    agg["n_managers"] = agg["n_increase"] + agg["n_decrease"] + (agg["n_new"] + agg["n_exit"]) - agg["n_increase"] - agg["n_decrease"]
    # Recalcular correctamente: n_managers = filas en per_mgr para ese ticker
    counts = per_mgr.groupby("ticker").size().rename("n_managers_count").reset_index()
    agg = agg.merge(counts, on="ticker", how="left")
    agg["n_managers"] = agg["n_managers_count"].astype(int)
    agg = agg.drop(columns=["n_managers_count"])

    denom_incdec = agg["n_increase"] + agg["n_decrease"]
    agg["pct_increase"] = agg["n_increase"] / denom_incdec.where(denom_incdec > 0, pd.NA)
    agg["pct_new"] = agg["n_new"] / agg["n_managers"]
    agg["pct_exit"] = agg["n_exit"] / agg["n_managers"]

    # Filtro de anomalias (ratio, sin lista de tickers).
    # PRIORIDAD: new/exit primero (eventos de universo como IPO o
    # spin-off). Si disparan, los filtros de consenso extremo no
    # sobreescriben. Si no disparan, se aplica consenso extremo.
    #
    # Verificado 2026-11-01 con datos reales (2026Q1->Q2):
    #   SPCX (IPO 12-jun-2026): pct_new=0.999 -> MOSTLY_NEW.
    #   HON (spin-off SOLS): pct_exit=0.923 -> MOSTLY_EXIT.
    #   DD (spin-off Qnity): pct_exit=0.927 -> MOSTLY_EXIT.
    # Los tres caen sin lista de tickers, solo por ratios.
    agg["filter_reason"] = None

    mask_new = agg["pct_new"] >= PCT_NEW_HIGH
    agg.loc[mask_new, "filter_reason"] = FILTER_MOSTLY_NEW

    mask_exit = (agg["pct_exit"] >= PCT_EXIT_HIGH) & agg["filter_reason"].isna()
    agg.loc[mask_exit, "filter_reason"] = FILTER_MOSTLY_EXIT

    mask_high = (agg["pct_increase"] >= PCT_INCREASE_HIGH) & agg["filter_reason"].isna()
    agg.loc[mask_high, "filter_reason"] = FILTER_ANOMALOUS

    mask_low = (
        (agg["pct_increase"] <= PCT_INCREASE_LOW)
        & agg["pct_increase"].notna()
        & agg["filter_reason"].isna()
    )
    agg.loc[mask_low, "filter_reason"] = FILTER_ANOMALOUS

    agg = agg.sort_values("sum_delta", ascending=False).reset_index(drop=True)
    return agg[list(PER_TICKER_COLUMNS)]


def top_accumulating(per_ticker_df, n=20):
    """Top N acumulando, ordenado por CONSENSO (n_increase), no por suma bruta.

    Filtra: filter_reason isna (paso los filtros) y sum_delta > 0.
    Ordena por (n_increase desc, sum_delta desc) y devuelve n.
    """
    if per_ticker_df is None or per_ticker_df.empty:
        return pd.DataFrame(columns=list(PER_TICKER_COLUMNS))
    df = per_ticker_df[
        per_ticker_df["filter_reason"].isna() & (per_ticker_df["sum_delta"] > 0)
    ]
    if df.empty:
        return df
    return df.sort_values(
        ["n_increase", "sum_delta"], ascending=[False, False]
    ).head(n).reset_index(drop=True)


def top_distributing(per_ticker_df, n=20):
    """Top N distribuyendo, ordenado por CONSENSO (n_decrease), no por suma bruta."""
    if per_ticker_df is None or per_ticker_df.empty:
        return pd.DataFrame(columns=list(PER_TICKER_COLUMNS))
    df = per_ticker_df[
        per_ticker_df["filter_reason"].isna() & (per_ticker_df["sum_delta"] < 0)
    ]
    if df.empty:
        return df
    return df.sort_values(
        ["n_decrease"], ascending=[False]
    ).head(n).reset_index(drop=True)


def excluded_anomalies(per_ticker_df):
    """Tickers excluidos con razon, ordenados por magnitud de sum_delta."""
    if per_ticker_df is None or per_ticker_df.empty:
        return pd.DataFrame(columns=list(PER_TICKER_COLUMNS))
    df = per_ticker_df[per_ticker_df["filter_reason"].notna()].copy()
    return df.reindex(
        df["sum_delta"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
