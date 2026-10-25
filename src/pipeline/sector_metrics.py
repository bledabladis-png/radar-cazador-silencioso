# -*- coding: utf-8 -*-
"""Fase 6b del pipeline: metricas sectoriales derivadas.

C4.3: retirados sector_leader_divergence y leader_representativeness.
Quedan 3 writers: sector_wyckoff_distribution, rs_internal,
sector_concentration.

Extraido de run.py (refactor C2, fase C2-7b).
"""

import pandas as pd

from src.pipeline.output_writer import persist, skip
from indicators.sector_wyckoff_distribution import compute_sector_wyckoff_distribution
from indicators.rs_internal import compute_rs_internal
from indicators.sector_concentration import compute_sector_concentration


def _compute_wyckoff(df_stocks, holdings_df, snapshot=None):
    try:
        if df_stocks is not None and not df_stocks.empty:
            sector_wyckoff_distribution_df = compute_sector_wyckoff_distribution(
                df_stocks, holdings_df, snapshot=snapshot,
            )
            if not sector_wyckoff_distribution_df.empty:
                date_val = sector_wyckoff_distribution_df["date"].max()
                persist("sector_wyckoff_distribution", sector_wyckoff_distribution_df, date_value=date_val)
                print("  Distribucion Wyckoff sectorial calculada.")
            else:
                skip("sector_wyckoff_distribution", "EMPTY_DF")
        else:
            sector_wyckoff_distribution_df = None
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError, pd.errors.ParserError) as e:
        print(f"  Distribucion Wyckoff sectorial omitida: {e}")
        sector_wyckoff_distribution_df = None
    return sector_wyckoff_distribution_df


def _compute_rs_internal(df_stocks, holdings_df, df_market, temporal_meta=None):
    try:
        if df_stocks is not None and not df_stocks.empty:
            rs_internal_df = compute_rs_internal(df_stocks, holdings_df, df_market, benchmark="SPY", temporal_meta=temporal_meta)
            if not rs_internal_df.empty:
                date_val = rs_internal_df["date"].max()
                persist("rs_internal", rs_internal_df, date_value=date_val)
                print("  RS Interno y Absoluto calculado.")
            else:
                skip("rs_internal", "EMPTY_DF")
        else:
            rs_internal_df = None
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError, pd.errors.ParserError) as e:
        print(f"  RS Interno y Absoluto omitido: {e}")
        rs_internal_df = None
    return rs_internal_df


def _compute_concentration(df_stocks, holdings_df, full_metrics_df, reference_date=None, sc_path=None):
    """C4.3: leader_df eliminado. compute_sector_concentration nunca lo uso."""
    try:
        if df_stocks is not None and not df_stocks.empty:
            sector_concentration_df = compute_sector_concentration(
                df_stocks, holdings_df, full_metrics_df, reference_date=reference_date
            )
            if not sector_concentration_df.empty:
                sector_concentration_df = sector_concentration_df.dropna(subset=["date"])
                sector_concentration_df = sector_concentration_df.drop_duplicates(subset=["date", "sector"], keep="last")
                date_val = sector_concentration_df["date"].max()
                persist("sector_concentration", sector_concentration_df,
                        date_value=date_val,
                        path_override=sc_path if sc_path is not None else None)
                print("  Sector Concentration calculado.")
            else:
                skip("sector_concentration", "EMPTY_DF")
        else:
            sector_concentration_df = None
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError, pd.errors.ParserError) as e:
        print(f"  Sector Concentration omitido: {e}")
        sector_concentration_df = None
    return sector_concentration_df


def compute_sector_metrics(df_stocks, holdings_df, full_metrics_df, df_market, effective_meta=None, temporal_meta=None, wyckoff_snapshot=None):
    """Calcula las metricas sectoriales derivadas de df_stocks.

    Returns:
        dict con keys:
            sector_wyckoff_distribution_df, rs_internal_df,
            sector_concentration_df
    """
    _ref_date = None
    if effective_meta is not None:
        _ref_date = effective_meta.get("date")
    if _ref_date is None and df_stocks is not None and not df_stocks.empty:
        _ref_date = df_stocks.index[-1]
    return {
        "sector_wyckoff_distribution_df": _compute_wyckoff(df_stocks, holdings_df, snapshot=wyckoff_snapshot),
        "rs_internal_df": _compute_rs_internal(df_stocks, holdings_df, df_market, temporal_meta=temporal_meta),
        "sector_concentration_df": _compute_concentration(df_stocks, holdings_df, full_metrics_df, reference_date=_ref_date),
    }
