"""Fase del pipeline: tablas de tickers por fase Wyckoff.

Carga los holdings de sectores USA + indices USA/EU, aplica el modulo
indicators/wyckoff_holdings_report.py y persiste los 6 CSVs via el
wrapper output_writer (persist/skip con receipt).

Contrato: ver docs/auditoria/wyckoff/63_plan_sistema_holdings_2026-10-24.md.
"""
from __future__ import annotations

import os

import pandas as pd

from indicators.wyckoff_holdings_report import (
    compute_wyckoff_holdings_report,
    USA_INDEX_ETFS,
    EU_INDEX_ETFS,
)
from src.pipeline.output_writer import persist, skip


DEFAULT_INDEX_HOLDINGS_PATH = "data/index_holdings.csv"

_OUTPUT_NAME_MAP = (
    ("usa_sector_acc", "wyckoff_usa_sector_accumulation"),
    ("usa_sector_dist", "wyckoff_usa_sector_distribution"),
    ("usa_index_acc", "wyckoff_usa_index_accumulation"),
    ("usa_index_dist", "wyckoff_usa_index_distribution"),
    ("eu_index_acc", "wyckoff_eu_index_accumulation"),
    ("eu_index_dist", "wyckoff_eu_index_distribution"),
)


def _load_index_holdings(path=None):
    """Carga data/index_holdings.csv. Devuelve None si no existe o esta vacio."""
    p = path or DEFAULT_INDEX_HOLDINGS_PATH
    if not os.path.exists(p):
        return None
    try:
        df = pd.read_csv(p)
    except (OSError, pd.errors.ParserError, ValueError):
        return None
    if df is None or df.empty or "etf" not in df.columns:
        return None
    return df


def _split_indices(index_holdings_df):
    """Devuelve (usa_idx_df, eu_idx_df) filtrados por ETF de interes."""
    if index_holdings_df is None or index_holdings_df.empty:
        return None, None
    usa = index_holdings_df[index_holdings_df["etf"].isin(USA_INDEX_ETFS)].copy()
    eu = index_holdings_df[index_holdings_df["etf"].isin(EU_INDEX_ETFS)].copy()
    return (usa if not usa.empty else None,
            eu if not eu.empty else None)


def compute_wyckoff_holdings(df_stocks, holdings_df, index_holdings_path=None):
    """Calcula y persiste los 6 CSVs de tablas Wyckoff por holdings.

    Args:
        df_stocks: DataFrame MultiIndex OHLCV (stock_prices.parquet).
        holdings_df: DataFrame con columnas etf/ticker/weight de
            data/etf_holdings.csv.
        index_holdings_path: path alternativo a data/index_holdings.csv
            (tests).

    Returns:
        dict con las 6 tablas (mismas claves que compute_wyckoff_holdings_report)
        o None si df_stocks es insuficiente.
    """
    if df_stocks is None or df_stocks.empty:
        for _, name in _OUTPUT_NAME_MAP:
            skip(name, "INPUT_UNAVAILABLE")
        return None
    if holdings_df is None or holdings_df.empty:
        for _, name in _OUTPUT_NAME_MAP:
            skip(name, "INPUT_UNAVAILABLE")
        return None

    index_holdings = _load_index_holdings(index_holdings_path)
    usa_idx_df, eu_idx_df = _split_indices(index_holdings)

    try:
        tables = compute_wyckoff_holdings_report(
            df_stocks, holdings_df, usa_idx_df, eu_idx_df,
        )
    except (KeyError, ValueError, TypeError, IndexError, AttributeError,
            OSError, RuntimeError) as e:
        print(f"  Wyckoff holdings omitido: {e}")
        for _, name in _OUTPUT_NAME_MAP:
            skip(name, "COMPUTE_ERROR")
        return None

    for key, name in _OUTPUT_NAME_MAP:
        df = tables.get(key)
        if df is None or df.empty:
            skip(name, "EMPTY_DF")
            continue
        try:
            date_val = pd.to_datetime(df["date"]).max()
        except (KeyError, ValueError, TypeError):
            date_val = None
        persist(
            name, df, date_value=date_val,
            sort_by=["date", "group", "phase", "ticker"],
        )
    print("  Tablas Wyckoff por holdings calculadas (6 CSVs).")
    return tables