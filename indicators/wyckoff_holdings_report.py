"""Tablas de tickers por fase Wyckoff para el reporte diario.

Sustituye a los lideres por sector basados en fase ETF. Deteccion
directa sobre los holdings (top-20 por peso de cada sector/indice).

Salida: 6 DataFrames (USA sectores ACC/DIST, USA indices ACC/DIST,
EU indices ACC/DIST), cada uno con el esquema definido en
docs/auditoria/wyckoff/63_plan_sistema_holdings_2026-10-24.md.
"""
from __future__ import annotations

import pandas as pd

from indicators.wyckoff_orchestrator import classify_phase
from src.utils import get_col, _observation_date_from_df


TOP_N_HOLDINGS = 20
MIN_BARS_THEORETICAL = 200
MIN_BARS_VALID = 460

ACC_PHASES = frozenset({"ACCUMULATION", "ACCUMULATION_FORMING", "MARKUP"})
DIST_PHASES = frozenset({"DISTRIBUTION", "DISTRIBUTION_FORMING", "MARKDOWN"})

PHASE_ORDER_ACC = {"ACCUMULATION": 0, "ACCUMULATION_FORMING": 1, "MARKUP": 2}
PHASE_ORDER_DIST = {"DISTRIBUTION": 0, "DISTRIBUTION_FORMING": 1, "MARKDOWN": 2}

USA_SECTOR_GROUP_TYPE = "sector_usa"
USA_INDEX_GROUP_TYPE = "index_usa"
EU_INDEX_GROUP_TYPE = "index_eu"

USA_INDEX_ETFS = ("SPY", "QQQ", "DIA")
EU_INDEX_ETFS = ("FEZ", "DAXEX", "ISF.L", "LYXI")

OUTPUT_COLUMNS = [
    "date", "group", "group_type", "ticker", "name", "weight",
    "phase", "event_phase", "phase_since", "close", "basis",
    "coverage_weight_pct",
]


def compute_wyckoff_holdings_report(
    df_stocks,
    usa_sectors_df,
    usa_indices_df,
    eu_indices_df,
):
    """Genera las 6 tablas: (usa_sector|usa_index|eu_index)_(acc|dist)."""
    usa_sector_acc, usa_sector_dist = _build_group_tables(
        df_stocks, usa_sectors_df, USA_SECTOR_GROUP_TYPE, has_name=False,
    )
    usa_index_acc, usa_index_dist = _build_group_tables(
        df_stocks, usa_indices_df, USA_INDEX_GROUP_TYPE, has_name=True,
    )
    eu_index_acc, eu_index_dist = _build_group_tables(
        df_stocks, eu_indices_df, EU_INDEX_GROUP_TYPE, has_name=True,
    )
    return {
        "usa_sector_acc": usa_sector_acc,
        "usa_sector_dist": usa_sector_dist,
        "usa_index_acc": usa_index_acc,
        "usa_index_dist": usa_index_dist,
        "eu_index_acc": eu_index_acc,
        "eu_index_dist": eu_index_dist,
    }


def _empty_table():
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _build_group_tables(df_stocks, holdings_df, group_type, has_name):
    if holdings_df is None or holdings_df.empty:
        return _empty_table(), _empty_table()

    obs_date = _observation_date_from_df(df_stocks)
    acc_rows = []
    dist_rows = []

    for group_key, group in holdings_df.groupby("etf"):
        top = group.nlargest(TOP_N_HOLDINGS, "weight")
        coverage_pct = float(top["weight"].sum())
        for _, hrow in top.iterrows():
            row = _evaluate_holding(
                df_stocks, group_key, hrow, group_type, has_name,
                obs_date, coverage_pct,
            )
            if row is None:
                continue
            phase = row["phase"]
            if phase in ACC_PHASES:
                acc_rows.append(row)
            elif phase in DIST_PHASES:
                dist_rows.append(row)

    return _sort_rows(acc_rows, PHASE_ORDER_ACC), _sort_rows(dist_rows, PHASE_ORDER_DIST)


def _evaluate_holding(df_stocks, group_key, hrow, group_type, has_name, obs_date, coverage_pct):
    ticker = hrow["ticker"]
    try:
        close_series = get_col(df_stocks, ticker, "Close").dropna()
    except (KeyError, ValueError, TypeError):
        return None
    n_bars = len(close_series)
    if n_bars < MIN_BARS_THEORETICAL:
        return None

    try:
        fp = classify_phase(df_stocks, ticker)
    except (KeyError, ValueError, TypeError, IndexError):
        return None

    if fp.phase is None or fp.phase == "INSUFFICIENT_DATA":
        return None

    if n_bars >= MIN_BARS_VALID:
        basis = fp.current_phase_basis or ""
    else:
        basis = "warm_up_incompleto (n={}): {}".format(
            n_bars, fp.current_phase_basis or ""
        )

    name = hrow["name"] if has_name and "name" in hrow.index else ticker

    return {
        "date": obs_date,
        "group": group_key,
        "group_type": group_type,
        "ticker": ticker,
        "name": name,
        "weight": float(hrow["weight"]),
        "phase": fp.phase,
        "event_phase": fp.event_phase or "",
        "phase_since": fp.phase_since if fp.phase_since is not None else "",
        "close": float(close_series.iloc[-1]),
        "basis": basis,
        "coverage_weight_pct": coverage_pct,
    }


def _sort_rows(rows, phase_order):
    if not rows:
        return _empty_table()
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df["_phase_order"] = df["phase"].map(phase_order).fillna(99).astype(int)
    df = df.sort_values(
        ["group", "_phase_order", "weight"],
        ascending=[True, True, False],
    )
    return df.drop(columns=["_phase_order"]).reset_index(drop=True)