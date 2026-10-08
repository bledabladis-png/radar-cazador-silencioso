# -*- coding: utf-8 -*-
"""Regenera sector_wyckoff_distribution.csv con v1.8 core.

2026-10-08: migracion legacy -> v1.8. Reconstruye el historico
completo aplicando v1.8 con as_of=fecha sobre el parquet actual.

NO productivo. Se ejecuta una vez.
El CSV anterior (legacy) queda en:
- git history (commit 089f819a).
- outputs/audit/wyckoff_legacy_REAL_20261008.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from config.tickers import MARKET_TICKERS
from indicators.wyckoff_v1 import build_ticker_df, classify_wyckoff_phase
from src.utils import get_col

SECTORS = MARKET_TICKERS['sectors']
VALID_PHASES = ['ACCUMULATION','MARKUP','RANGE','DISTRIBUTION','MARKDOWN']

LEGACY_CSV = Path("outputs/audit/wyckoff_legacy_REAL_20261008.csv")
OUT_CSV = Path("outputs/history/sector_wyckoff_distribution.csv")
STOCKS_PARQUET = Path("data/stock_prices.parquet")
HOLDINGS_CSV = Path("data/etf_holdings.csv")


def classify_sector_at(fecha, tickers, df_stocks):
    phase_counts = {phase: 0 for phase in VALID_PHASES}
    n_valid = 0
    n_insufficient = 0
    for ticker in tickers:
        try:
            close = get_col(df_stocks, ticker, 'Close').dropna()
        except (KeyError, TypeError):
            n_insufficient += 1
            continue
        if len(close) < 60:
            n_insufficient += 1
            continue
        try:
            ticker_df = build_ticker_df(df_stocks, ticker)
            phase = classify_wyckoff_phase(ticker_df, ticker, as_of=fecha)
        except (KeyError, ValueError, TypeError, IndexError, AttributeError):
            n_insufficient += 1
            continue
        if phase in VALID_PHASES:
            phase_counts[phase] += 1
            n_valid += 1
        else:
            n_insufficient += 1
    return phase_counts, n_valid, n_insufficient


def main():
    print(f"Cargando {STOCKS_PARQUET}...")
    df_stocks = pd.read_parquet(STOCKS_PARQUET)
    print(f"  shape: {df_stocks.shape}")
    print(f"Cargando {HOLDINGS_CSV}...")
    holdings_df = pd.read_csv(HOLDINGS_CSV)
    print(f"  sectores: {holdings_df['etf'].nunique()}")
    print(f"Cargando fechas de {LEGACY_CSV}...")
    legacy = pd.read_csv(LEGACY_CSV)
    fechas = sorted(legacy['date'].unique())
    print(f"  fechas unicas: {len(fechas)}")
    print()

    rows = []
    for i, fecha in enumerate(fechas, 1):
        print(f"[{i}/{len(fechas)}] {fecha}...")
        fecha_ts = pd.Timestamp(fecha)
        for sector_etf, group in holdings_df.groupby('etf'):
            if sector_etf not in SECTORS:
                continue
            tickers = group['ticker'].tolist()
            n_total = len(tickers)
            phase_counts, n_valid, n_insufficient = classify_sector_at(
                fecha_ts, tickers, df_stocks
            )
            coverage = (n_valid / n_total * 100) if n_total else np.nan
            row = {
                'date': fecha,
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

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(['date', 'sector']).reset_index(drop=True)
    out_df.to_csv(OUT_CSV, index=False, lineterminator='\n')
    print()
    print(f"OK -> {OUT_CSV} ({len(out_df)} filas)")


if __name__ == "__main__":
    main()