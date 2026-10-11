# -*- coding: utf-8 -*-
"""Fase 6b del pipeline: metricas sectoriales derivadas.

Divergencia sector-lideres + Distribucion Wyckoff + RS Interno +
Sector Concentration + Representatividad del lider.

Extraido de run.py (refactor C2, fase C2-7b).
"""

import pandas as pd

from src.pipeline.output_writer import persist, skip
from indicators.sector_leader_divergence import compute_sector_leader_divergence
from indicators.sector_wyckoff_distribution import compute_sector_wyckoff_distribution
from indicators.rs_internal import compute_rs_internal
from indicators.sector_concentration import compute_sector_concentration
from indicators.leader_representativeness import compute_leader_representativeness


def _compute_divergencia(df_stocks, holdings_df, leader_df, df_market, temporal_meta=None):
    try:
        if df_stocks is not None and not df_stocks.empty and leader_df is not None and not leader_df.empty:
            sector_leader_divergence_df = compute_sector_leader_divergence(
                df_stocks, holdings_df, leader_df, df_market,
                temporal_meta=temporal_meta,
            )
            if not sector_leader_divergence_df.empty:
                date_val = sector_leader_divergence_df['date'].max()
                persist('sector_leader_divergence', sector_leader_divergence_df, date_value=date_val)
                print("  Divergencia sector-lideres calculada.")
            else:
                skip('sector_leader_divergence', 'EMPTY_DF')
        else:
            sector_leader_divergence_df = None
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError, pd.errors.ParserError) as e:
        print(f"  Divergencia sector-lideres omitida: {e}")
        sector_leader_divergence_df = None
    return sector_leader_divergence_df


def _compute_wyckoff(df_stocks, holdings_df):
    try:
        if df_stocks is not None and not df_stocks.empty:
            sector_wyckoff_distribution_df = compute_sector_wyckoff_distribution(df_stocks, holdings_df)
            if not sector_wyckoff_distribution_df.empty:
                date_val = sector_wyckoff_distribution_df['date'].max()
                persist('sector_wyckoff_distribution', sector_wyckoff_distribution_df, date_value=date_val)
                print("  Distribucion Wyckoff sectorial calculada.")
            else:
                skip('sector_wyckoff_distribution', 'EMPTY_DF')
        else:
            sector_wyckoff_distribution_df = None
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError, pd.errors.ParserError) as e:
        print(f"  Distribucion Wyckoff sectorial omitida: {e}")
        sector_wyckoff_distribution_df = None
    return sector_wyckoff_distribution_df


def _compute_rs_internal(df_stocks, holdings_df, df_market, temporal_meta=None):
    try:
        if df_stocks is not None and not df_stocks.empty:
            rs_internal_df = compute_rs_internal(df_stocks, holdings_df, df_market, benchmark='SPY', temporal_meta=temporal_meta)
            if not rs_internal_df.empty:
                date_val = rs_internal_df['date'].max()
                persist('rs_internal', rs_internal_df, date_value=date_val)
                print("  RS Interno y Absoluto calculado.")
            else:
                skip('rs_internal', 'EMPTY_DF')
        else:
            rs_internal_df = None
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError, pd.errors.ParserError) as e:
        print(f"  RS Interno y Absoluto omitido: {e}")
        rs_internal_df = None
    return rs_internal_df


def _compute_concentration(df_stocks, holdings_df, leader_df, full_metrics_df, reference_date=None, sc_path=None):
    """FU-008-b (2026-09-13): el writer NO depende de leader_df.

    compute_sector_concentration opera solo con df_stocks, holdings_df y full_metrics_df.
    Antes exigia leader_df no vacio, lo que bloqueaba el writer en runs sin sectores
    favorables (estado comun), dejando el CSV congelado con datos obsoletos.
    """
    try:
        if df_stocks is not None and not df_stocks.empty:
            sector_concentration_df = compute_sector_concentration(
                df_stocks, holdings_df, full_metrics_df, reference_date=reference_date
            )
            if not sector_concentration_df.empty:
                sector_concentration_df = sector_concentration_df.dropna(subset=['date'])
                sector_concentration_df = sector_concentration_df.drop_duplicates(subset=['date','sector'], keep='last')
                date_val = sector_concentration_df['date'].max()
                persist('sector_concentration', sector_concentration_df,
                        date_value=date_val,
                        path_override=sc_path if sc_path is not None else None)
                print("  Sector Concentration calculado.")
            else:
                skip('sector_concentration', 'EMPTY_DF')
        else:
            sector_concentration_df = None
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError, pd.errors.ParserError) as e:
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
            if not leader_representativeness_df.empty:
                date_val = leader_representativeness_df['date'].max()
                persist('leader_representativeness', leader_representativeness_df, date_value=date_val)
                print("  Representatividad del lider calculada.")
            else:
                skip('leader_representativeness', 'EMPTY_DF')
        else:
            leader_representativeness_df = None
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError, pd.errors.ParserError) as e:
        print(f"  Representatividad del lider omitida: {e}")
        leader_representativeness_df = None
    return leader_representativeness_df


def compute_sector_metrics(df_stocks, holdings_df, leader_df, full_metrics_df, df_market, effective_meta=None, temporal_meta=None):
    """Calcula las metricas sectoriales derivadas de df_stocks.

    Returns:
        dict con keys:
            sector_leader_divergence_df, sector_wyckoff_distribution_df,
            rs_internal_df, sector_concentration_df,
            leader_representativeness_df
    """
    _ref_date = None
    if effective_meta is not None:
        _ref_date = effective_meta.get('date')
    if _ref_date is None and df_stocks is not None and not df_stocks.empty:
        _ref_date = df_stocks.index[-1]
    return {
        'sector_leader_divergence_df': _compute_divergencia(df_stocks, holdings_df, leader_df, df_market, temporal_meta=temporal_meta),
        'sector_wyckoff_distribution_df': _compute_wyckoff(df_stocks, holdings_df),
        'rs_internal_df': _compute_rs_internal(df_stocks, holdings_df, df_market, temporal_meta=temporal_meta),
        'sector_concentration_df': _compute_concentration(df_stocks, holdings_df, leader_df, full_metrics_df, reference_date=_ref_date),
        'leader_representativeness_df': _compute_representativeness(
            leader_df, reference_date=_ref_date),
    }
