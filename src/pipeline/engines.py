# -*- coding: utf-8 -*-
"""Fase 8a del pipeline: tactical/structural engines + persistence sectorial.

Extraido de run.py (refactor C2, fase C2-8a).
"""

import pandas as pd

from src.utils import get_col, _observation_date_from_df, writer_observation_date, get_effective_meta
from indicators.persistence import compute_persistence
from src.pipeline.output_writer import persist, skip
from config.tickers import MARKET_TICKERS


# A5-11 (2026-09-28): SECTOR_ETFS unificado con MARKET_TICKERS['sectors']
# (fuente unica declarada). Antes era constante local duplicada.
SECTOR_ETFS = MARKET_TICKERS['sectors']


def _compute_tactical_structural(df_market, temporal_meta=None):
    tactical_scores = {}
    structural_scores = {}
    try:
        from regimes.tactical_engine import compute_tactical_score
        from regimes.structural_engine import compute_structural_score
        for sector_etf in SECTOR_ETFS:
            try:
                tactical_scores[sector_etf] = compute_tactical_score(df_market, sector_etf, temporal_meta=temporal_meta)
                structural_scores[sector_etf] = compute_structural_score(df_market, sector_etf, temporal_meta=temporal_meta)
            except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError, pd.errors.ParserError) as e:
                print(f"  [WARN] tactical/structural engine: {e}")
                tactical_scores[sector_etf] = 0.0
                structural_scores[sector_etf] = 0.0
        print(f"    Tactical/Structural engines calculados para {len(tactical_scores)} sectores.")
    except Exception as e:
        print(f"    Tactical/Structural engines omitidos: {e}")
    return tactical_scores, structural_scores


def _compute_persistence_and_save(df_market, temporal_meta=None):
    sector_persistence = {}
    try:
        for sector_etf in SECTOR_ETFS:
            try:
                close_sector = get_col(df_market, sector_etf, 'Close')
                close_spy = get_col(df_market, '^GSPC', 'Close')
                rs = close_sector / close_spy
                rs20 = rs.pct_change(20, fill_method=None)
                pers = compute_persistence(rs20, threshold=0.0, lookback=12)
                sector_persistence[sector_etf] = pers
            except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, RuntimeError, pd.errors.ParserError) as e:
                print(f"  [WARN] persistence: {e}")
                sector_persistence[sector_etf] = None
        print(f"    Persistence calculada para {len(sector_persistence)} sectores.")
    except Exception as e:
        print(f"    Persistence omitida: {e}")
        sector_persistence = {s: None for s in SECTOR_ETFS}

    # Guardar CSV historico de persistencia sectorial
    try:
        persist_rows = []
        date_val = writer_observation_date(
            temporal_meta, ['EQUITY_EOD'],
            lambda: _observation_date_from_df(df_market))
        _eff_meta = get_effective_meta(temporal_meta or {}, ['EQUITY_EOD'])
        for sec, val in sector_persistence.items():
            persist_rows.append({
                'date': date_val,
                'effective_date': _eff_meta.get('effective_date'),
                'expected_date': _eff_meta.get('expected_date'),
                'coverage': _eff_meta.get('coverage'),
                'sector': sec,
                'persistence': val,
            })
        persist_df = pd.DataFrame(persist_rows)
        persist('sector_persistence', persist_df)
        print("  Sector Persistence CSV guardado.")
    except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as e:
        print(f"  Sector Persistence CSV omitido: {e}")
        skip('sector_persistence', 'NO_DATA')

    return sector_persistence


def compute_engines(df_market, sector_results, sector_flow_rank, otros_flow_rank, temporal_meta=None):
    """Ejecuta tactical/structural + persistence.

    Returns:
        dict con keys:
            tactical_scores, structural_scores, sector_persistence
    """
    tactical_scores, structural_scores = _compute_tactical_structural(df_market, temporal_meta=temporal_meta)
    sector_persistence = _compute_persistence_and_save(df_market, temporal_meta=temporal_meta)
    return {
        'tactical_scores': tactical_scores,
        'structural_scores': structural_scores,
        'sector_persistence': sector_persistence,
    }
