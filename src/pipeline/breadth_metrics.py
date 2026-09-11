# -*- coding: utf-8 -*-
"""Fase 6c del pipeline: Sector Breadth & Health + Momentum de amplitud.

Extraido de run.py (refactor C2, fase C2-7c).
"""

from pathlib import Path

import pandas as pd

from src.utils import append_dedup
from indicators.sector_breadth import compute_sector_breadth
from indicators.sector_breadth_momentum import compute_sector_breadth_momentum


def _compute_momentum_amplitud(df_stocks):
    try:
        if df_stocks is not None and not df_stocks.empty:
            sector_breadth_momentum_df = compute_sector_breadth_momentum(
                'outputs/history/sector_breadth.csv'
            )
            sbm_path = Path('outputs/history/sector_breadth_momentum.csv')
            sbm_path.parent.mkdir(parents=True, exist_ok=True)
            if not sector_breadth_momentum_df.empty:
                if sbm_path.exists():
                    hist_sbm = pd.read_csv(sbm_path)
                    sector_breadth_momentum_df = append_dedup(hist_sbm, sector_breadth_momentum_df, ["date","sector"])
                sector_breadth_momentum_df.to_csv(sbm_path, index=False)
                print("  Momentum de amplitud sectorial calculado.")
        else:
            sector_breadth_momentum_df = None
    except Exception as e:
        print(f"  Momentum de amplitud sectorial omitido: {e}")
        sector_breadth_momentum_df = None
    return sector_breadth_momentum_df


def _compute_sector_breadth_health(df_stocks, df_market, holdings_df):
    try:
        if df_stocks is not None and not df_stocks.empty:
            sector_breadth_df = compute_sector_breadth(df_market, df_stocks, holdings_df)
            sb_path = Path('outputs/history/sector_breadth.csv')
            sb_path.parent.mkdir(parents=True, exist_ok=True)
            if not sector_breadth_df.empty:
                if sb_path.exists():
                    hist_sb = pd.read_csv(sb_path)
                    sector_breadth_df = append_dedup(hist_sb, sector_breadth_df, ["date","sector"])
                sector_breadth_df.to_csv(sb_path, index=False)
                print("  Sector Breadth & Health calculado.")
        else:
            sector_breadth_df = None
    except Exception as e:
        print(f"  Sector Breadth & Health omitido: {e}")
        sector_breadth_df = None
    return sector_breadth_df


def compute_breadth_metrics(df_stocks, df_market, holdings_df):
    """Calcula Sector Breadth & Health + Momentum de amplitud.

    Returns:
        dict con keys:
            sector_breadth_momentum_df, sector_breadth_df
    """
    return {
        'sector_breadth_momentum_df': _compute_momentum_amplitud(df_stocks),
        'sector_breadth_df': _compute_sector_breadth_health(df_stocks, df_market, holdings_df),
    }
