# -*- coding: utf-8 -*-
"""
Distribucion de fases Wyckoff por sector v2.0

Consume classify_phase de indicators/wyckoff_orchestrator.py (cascada
nueva: BC/SC -> AR -> Range -> SOW/SOS + fase derivada por vigencia).

Sustituye classify_wyckoff_phase v1 (que detectaba 0/327 DISTRIBUTION).

Fases validas: ACCUMULATION, MARKUP, RANGE, DISTRIBUTION, MARKDOWN, CONFLICT.
INSUFFICIENT_DATA se cuenta como n_insufficient (no es una fase).

No alimenta motores, scores, pesos ni State Machine.
"""
import pandas as pd
import numpy as np
from src.utils import get_col, _observation_date_from_df
from indicators.wyckoff_orchestrator import classify_phase

from config.tickers import MARKET_TICKERS

SECTORS = MARKET_TICKERS['sectors']
VALID_PHASES = [
    'ACCUMULATION', 'ACCUMULATION_FORMING', 'MARKUP',
    'RANGE',
    'DISTRIBUTION', 'DISTRIBUTION_FORMING', 'MARKDOWN',
    'CONFLICT',
]


def _safe_classify(df_stocks, ticker):
    """Wrapper: devuelve string de fase o None si no evaluable.

    Separado para permitir mocking en tests y centralizar el manejo de
    excepciones del orquestador.
    """
    try:
        result = classify_phase(df_stocks, ticker)
    except (KeyError, ValueError, TypeError, IndexError, AttributeError):
        return None
    return result.phase


def compute_sector_wyckoff_distribution(df_stocks, holdings_df):
    rows = []
    for sector_etf, group in holdings_df.groupby('etf'):
        if sector_etf not in SECTORS:
            continue
        tickers = group['ticker'].tolist()
        phase_counts = {phase: 0 for phase in VALID_PHASES}
        n_valid = 0
        n_insufficient = 0

        for ticker in tickers:
            try:
                close = get_col(df_stocks, ticker, 'Close').dropna()
            except KeyError:
                n_insufficient += 1
                continue
            if len(close) < 60:
                n_insufficient += 1
                continue

            phase = _safe_classify(df_stocks, ticker)
            if phase is None or phase == "INSUFFICIENT_DATA":
                n_insufficient += 1
                continue
            if phase in VALID_PHASES:
                phase_counts[phase] += 1
                n_valid += 1
            else:
                n_insufficient += 1

        n_total = len(tickers)
        coverage = (n_valid / n_total * 100) if n_total else np.nan

        row = {
            'date': _observation_date_from_df(df_stocks),
            'sector': sector_etf,
            'n_total': n_total,
            'n_valid_wyckoff': n_valid,
            'n_insufficient_wyckoff': n_insufficient,
            'coverage_wyckoff': coverage,
        }
        if n_valid >= 5:
            for phase in VALID_PHASES:
                row[f'count_{phase.lower()}'] = phase_counts[phase]
                row[f'pct_{phase.lower()}'] = phase_counts[phase] / n_valid * 100
        else:
            for phase in VALID_PHASES:
                row[f'count_{phase.lower()}'] = phase_counts[phase]
                row[f'pct_{phase.lower()}'] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)