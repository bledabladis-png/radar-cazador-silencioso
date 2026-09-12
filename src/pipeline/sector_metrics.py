# -*- coding: utf-8 -*-
"""Fase 6b del pipeline: metricas sectoriales derivadas.

Divergencia sector-lideres + Distribucion Wyckoff + RS Interno +
Sector Concentration + Representatividad del lider.

Extraido de run.py (refactor C2, fase C2-7b).
"""

from pathlib import Path

import pandas as pd

from src.utils import append_dedup
from indicators.sector_leader_divergence import compute_sector_leader_divergence
from indicators.sector_wyckoff_distribution import compute_sector_wyckoff_distribution
from indicators.rs_internal import compute_rs_internal
from indicators.sector_concentration import compute_sector_concentration
from indicators.leader_representativeness import compute_leader_representativeness


def _compute_divergencia(df_stocks, holdings_df, leader_df, df_market):
    try:
        if df_stocks is not None and not df_stocks.empty and leader_df is not None and not leader_df.empty:
            sector_leader_divergence_df = compute_sector_leader_divergence(
                df_stocks, holdings_df, leader_df, df_market
            )
            sld_path = Path('outputs/history/sector_leader_divergence.csv')
            sld_path.parent.mkdir(parents=True, exist_ok=True)
            if not sector_leader_divergence_df.empty:
                if sld_path.exists():
                    hist_sld = pd.read_csv(sld_path)
                    sector_leader_divergence_df = append_dedup(hist_sld, sector_leader_divergence_df, ["date","sector"])
                sector_leader_divergence_df.to_csv(sld_path, index=False, encoding='utf-8')
                print("  Divergencia sector-lideres calculada.")
        else:
            sector_leader_divergence_df = None
    except Exception as e:
        print(f"  Divergencia sector-lideres omitida: {e}")
        sector_leader_divergence_df = None
    return sector_leader_divergence_df


def _compute_wyckoff(df_stocks, holdings_df):
    try:
        if df_stocks is not None and not df_stocks.empty:
            sector_wyckoff_distribution_df = compute_sector_wyckoff_distribution(df_stocks, holdings_df)
            wyckoff_path = Path('outputs/history/sector_wyckoff_distribution.csv')
            wyckoff_path.parent.mkdir(parents=True, exist_ok=True)
            if not sector_wyckoff_distribution_df.empty:
                if wyckoff_path.exists():
                    hist_wy = pd.read_csv(wyckoff_path)
                    sector_wyckoff_distribution_df = append_dedup(hist_wy, sector_wyckoff_distribution_df, ["date","sector"])
                sector_wyckoff_distribution_df.to_csv(wyckoff_path, index=False, encoding='utf-8')
                print("  Distribucion Wyckoff sectorial calculada.")
        else:
            sector_wyckoff_distribution_df = None
    except Exception as e:
        print(f"  Distribucion Wyckoff sectorial omitida: {e}")
        sector_wyckoff_distribution_df = None
    return sector_wyckoff_distribution_df


def _compute_rs_internal(df_stocks, holdings_df, df_market):
    try:
        if df_stocks is not None and not df_stocks.empty:
            rs_internal_df = compute_rs_internal(df_stocks, holdings_df, df_market, benchmark='SPY')
            rs_path = Path('outputs/history/rs_internal.csv')
            rs_path.parent.mkdir(parents=True, exist_ok=True)
            if not rs_internal_df.empty:
                if rs_path.exists():
                    hist_rs = pd.read_csv(rs_path)
                    rs_internal_df = append_dedup(hist_rs, rs_internal_df, ["date","sector"])
                rs_internal_df.to_csv(rs_path, index=False, encoding='utf-8')
                print("  RS Interno y Absoluto calculado.")
        else:
            rs_internal_df = None
    except Exception as e:
        print(f"  RS Interno y Absoluto omitido: {e}")
        rs_internal_df = None
    return rs_internal_df


def _compute_concentration(df_stocks, holdings_df, leader_df, full_metrics_df):
    try:
        if df_stocks is not None and not df_stocks.empty and leader_df is not None and not leader_df.empty:
            sector_concentration_df = compute_sector_concentration(df_stocks, holdings_df, full_metrics_df)
            sc_path = Path('outputs/history/sector_concentration.csv')
            sc_path.parent.mkdir(parents=True, exist_ok=True)
            if not sector_concentration_df.empty:
                if sc_path.exists():
                    hist_sc = pd.read_csv(sc_path)
                    hist_sc = hist_sc.dropna(subset=['date'])
                    sector_concentration_df = append_dedup(hist_sc, sector_concentration_df, ["date","sector"])
                sector_concentration_df = sector_concentration_df.dropna(subset=['date'])
                sector_concentration_df = sector_concentration_df.drop_duplicates(subset=['date','sector'], keep='last')
                sector_concentration_df.to_csv(sc_path, index=False)
                print("  Sector Concentration calculado.")
        else:
            sector_concentration_df = None
    except Exception as e:
        print(f"  Sector Concentration omitido: {e}")
        sector_concentration_df = None
    return sector_concentration_df


def _compute_representativeness(leader_df, reference_date=None):
    try:
        if leader_df is not None and not leader_df.empty:
            leader_representativeness_df = compute_leader_representativeness(
                leader_df, 'outputs/history/sector_concentration.csv',
                reference_date=reference_date
            )
            lr_path = Path('outputs/history/leader_representativeness.csv')
            lr_path.parent.mkdir(parents=True, exist_ok=True)
            if not leader_representativeness_df.empty:
                if lr_path.exists():
                    hist_lr = pd.read_csv(lr_path)
                    leader_representativeness_df = append_dedup(hist_lr, leader_representativeness_df, ["date","sector","ticker"])
                leader_representativeness_df.to_csv(lr_path, index=False)
                print("  Representatividad del lider calculada.")
        else:
            leader_representativeness_df = None
    except Exception as e:
        print(f"  Representatividad del lider omitida: {e}")
        leader_representativeness_df = None
    return leader_representativeness_df


def compute_sector_metrics(df_stocks, holdings_df, leader_df, full_metrics_df, df_market):
    """Calcula las metricas sectoriales derivadas de df_stocks.

    Returns:
        dict con keys:
            sector_leader_divergence_df, sector_wyckoff_distribution_df,
            rs_internal_df, sector_concentration_df,
            leader_representativeness_df
    """
    return {
        'sector_leader_divergence_df': _compute_divergencia(df_stocks, holdings_df, leader_df, df_market),
        'sector_wyckoff_distribution_df': _compute_wyckoff(df_stocks, holdings_df),
        'rs_internal_df': _compute_rs_internal(df_stocks, holdings_df, df_market),
        'sector_concentration_df': _compute_concentration(df_stocks, holdings_df, leader_df, full_metrics_df),
        'leader_representativeness_df': _compute_representativeness(
            leader_df, reference_date=df_stocks.index[-1]),
    }
