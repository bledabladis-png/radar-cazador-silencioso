"""Saneamiento puntual: USA en festivos NYSE en stock_prices.parquet.

Contexto (2026-09-24): el merge de lotes Yahoo mixtos UK+USA hacia que
_fill_holes_respecting_sessions propagara valores de tickers UK (LSE
abierto en festivos USA) a tickers USA en festivos NYSE. El fix en
stock_data_loader.py corrige runs futuros; este script limpia el
parquet historico.

Operacion:
  1. Backup: data/stock_prices.parquet.pre_cleanup_YYYYMMDD
  2. Para cada (fecha, ticker) donde:
       - is_market_day(fecha) == False
       - fecha.weekday() < 5  (no fin de semana)
       - get_market(ticker) == 'US_EQUITY'
     -> NaN en todas las columnas del ticker (Close, High, Low, Volume, Open)
  3. Reporta celdas limpiadas.

Uso:
    py scripts/cleanup_stock_prices_nyse_holidays.py --dry-run
    py scripts/cleanup_stock_prices_nyse_holidays.py --apply
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.market_calendar import is_market_day
from src.instrument_registry import get_market

PARQUET = ROOT / "data" / "stock_prices.parquet"
TS_FMT = "%Y%m%d_%H%M%S"


def _is_us_holiday_weekday(d):
    """True si d es dia laborable NO NYSE (festivo) y no fin de semana."""
    if d.weekday() >= 5:
        return False
    return not is_market_day(d)


def _flatten(col):
    return col[1] if isinstance(col, tuple) else col


def _field(col):
    return col[0] if isinstance(col, tuple) else "Close"


def scan(df):
    """Devuelve lista de (fecha, ticker, field) a limpiar."""
    hits = []
    idx = pd.to_datetime(df.index)
    for i, fecha in enumerate(idx):
        if not _is_us_holiday_weekday(fecha.date()):
            continue
        for col in df.columns:
            ticker = _flatten(col)
            if get_market(str(ticker)) != "US_EQUITY":
                continue
            val = df.iloc[i][col]
            if pd.notna(val):
                hits.append((fecha.date(), ticker, _field(col)))
    return hits


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not PARQUET.exists():
        print("[FAIL] no existe: " + str(PARQUET))
        return 1

    df = pd.read_parquet(PARQUET)
    df.index = pd.to_datetime(df.index)
    print("Parquet: " + str(PARQUET))
    print("Shape: " + str(df.shape))
    print("Rango: " + str(df.index[0].date()) + " -> " + str(df.index[-1].date()))

    hits = scan(df)
    print("\nCeldas a limpiar: " + str(len(hits)))

    by_date = {}
    by_ticker = {}
    for f, t, fl in hits:
        by_date[f] = by_date.get(f, 0) + 1
        by_ticker[t] = by_ticker.get(t, 0) + 1

    print("\nPor fecha:")
    for f in sorted(by_date):
        print("  " + str(f) + " (" + f.strftime("%A") + "): " + str(by_date[f]) + " celdas")
    print("\nPor ticker:")
    for t in sorted(by_ticker):
        print("  " + t + ": " + str(by_ticker[t]) + " celdas")

    if args.dry_run:
        print("\n[DRY-RUN] no se ha modificado el parquet.")
        return 0

    # --- apply ---
    ts = datetime.now().strftime(TS_FMT)
    backup = PARQUET.with_name("stock_prices.parquet.pre_cleanup_" + ts)
    shutil.copy2(PARQUET, backup)
    print("\nBackup: " + str(backup))

    n_changed = 0
    for f, t, _ in hits:
        # set NaN en todas las filas del ticker para la fecha
        for col in df.columns:
            if _flatten(col) == t:
                df.at[pd.Timestamp(f), col] = float("nan")
                n_changed += 1

    df.to_parquet(PARQUET, compression="snappy")
    print("[OK] " + str(n_changed) + " celdas limpiadas y parquet reescrito.")
    return 0


if __name__ == "__main__":
    sys.exit(main())