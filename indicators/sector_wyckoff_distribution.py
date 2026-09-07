# -*- coding: utf-8 -*-
"""
Distribución de fases Wyckoff por sector v1.0
Consume classify_wyckoff_phase de indicators/wyckoff.py.
No alimenta motores, scores, pesos ni State Machine.
"""
import pandas as pd
import numpy as np
from src.utils import get_col
from indicators.wyckoff import classify_wyckoff_phase

SECTORS = ['XLK','XLF','XLV','XLE','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
VALID_PHASES = ['ACCUMULATION','MARKUP','RANGE','DISTRIBUTION','MARKDOWN']

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

            try:
                phase = classify_wyckoff_phase(df_stocks, ticker)
            except Exception:
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
            'date': pd.Timestamp.now().normalize(),
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
