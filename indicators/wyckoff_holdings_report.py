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
    "coverage_weight_pct", "n_bars", "evaluable",
]


def compute_wyckoff_holdings_report(
    df_stocks,
    usa_sectors_df,
    usa_indices_df,
    eu_indices_df,
    snapshot=None,
):
    """Genera las 6 tablas: (usa_sector|usa_index|eu_index)_(acc|dist).

    Si snapshot se pasa, lo usa. Si no, lo calcula. El snapshot es la
    fuente unica de holdings evaluados (Fase 4).
    """
    if snapshot is None:
        snapshot = compute_wyckoff_snapshot(
            df_stocks, usa_sectors_df, usa_indices_df, eu_indices_df,
        )
    return _tables_from_snapshot(snapshot)


def _tables_from_snapshot(snapshot):
    """Proyecta el snapshot en las 6 tablas por fase y grupo."""
    if snapshot is None or snapshot.empty:
        return {
            "usa_sector_acc": _empty_table(), "usa_sector_dist": _empty_table(),
            "usa_index_acc": _empty_table(), "usa_index_dist": _empty_table(),
            "eu_index_acc": _empty_table(), "eu_index_dist": _empty_table(),
        }

    out = {}
    for gt, prefix in (
        (USA_SECTOR_GROUP_TYPE, "usa_sector"),
        (USA_INDEX_GROUP_TYPE, "usa_index"),
        (EU_INDEX_GROUP_TYPE, "eu_index"),
    ):
        sub = snapshot[snapshot["group_type"] == gt]
        out[f"{prefix}_acc"] = _sort_rows(
            sub[sub["phase"].isin(ACC_PHASES)].to_dict("records"),
            PHASE_ORDER_ACC,
        )
        out[f"{prefix}_dist"] = _sort_rows(
            sub[sub["phase"].isin(DIST_PHASES)].to_dict("records"),
            PHASE_ORDER_DIST,
        )
    return out


def compute_wyckoff_snapshot(df_stocks, usa_sectors_df, usa_indices_df, eu_indices_df):
    """Snapshot unico de holdings evaluados (Fase 4).

    Devuelve DataFrame long con TODOS los tickers evaluados (top-20 por
    grupo, n_bars >= MIN_BARS_THEORETICAL). Columnas = OUTPUT_COLUMNS.

    Consumidores:
    - compute_wyckoff_holdings_report: filtra por fase para los 6 CSVs.
    - sector_wyckoff_distribution: agrega pct_* por sector (top-20).
    - sector_regime_matrix: deriva structure_positive (via el agregado).

    Una sola pasada de classify_phase. Test de coherencia en
    tests/test_wyckoff_holdings_coherence.py.
    """
    rows = []
    for holdings_df, group_type, has_name in (
        (usa_sectors_df, USA_SECTOR_GROUP_TYPE, False),
        (usa_indices_df, USA_INDEX_GROUP_TYPE, True),
        (eu_indices_df, EU_INDEX_GROUP_TYPE, True),
    ):
        if holdings_df is None or holdings_df.empty:
            continue
        obs_date = _observation_date_from_df(df_stocks)
        for group_key, group in holdings_df.groupby("etf"):
            top = group.nlargest(TOP_N_HOLDINGS, "weight")
            coverage_pct = float(top["weight"].sum())
            for _, hrow in top.iterrows():
                row = _evaluate_holding(
                    df_stocks, group_key, hrow, group_type, has_name,
                    obs_date, coverage_pct,
                )
                if row is not None:
                    rows.append(row)
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _empty_table():
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _phase_since_iso(value):
    """Formatea phase_since como ISO date (YYYY-MM-DD) o '' si vacio.

    Contrato doc 63 §4: phase_since es fecha, no timestamp."""
    if value is None:
        return ""
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return ""


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
        evaluable = True
    else:
        basis = "warm_up_incompleto (n={}): {}".format(
            n_bars, fp.current_phase_basis or ""
        )
        evaluable = False

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
        "phase_since": _phase_since_iso(fp.phase_since),
        "close": float(close_series.iloc[-1]),
        "basis": basis,
        "coverage_weight_pct": coverage_pct,
        "n_bars": int(n_bars),
        "evaluable": bool(evaluable),
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