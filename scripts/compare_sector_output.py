"""Compara output de sector_wyckoff_distribution viejo (classify_wyckoff_phase)
vs nuevo (classify_phase + phase/event_phase)."""
import sys
sys.path.insert(0, ".")
import pandas as pd
from collections import Counter
from indicators.wyckoff_v1 import build_ticker_df, classify_wyckoff_phase
from indicators.wyckoff_orchestrator import classify_phase
from src.utils import get_col

df = pd.read_parquet("data/stock_prices.parquet")
etf = pd.read_csv("data/etf_holdings.csv")

# Para cada ETF: contar fases de sus top-20 holdings con AMBOS sistemas
for sector, group in etf.groupby("etf"):
    tickers = group.head(20)["ticker"].tolist()
    old_phases = Counter()
    new_phases = Counter()
    new_event_phases = Counter()
    n_valid = 0

    for tk in tickers:
        try:
            close = get_col(df, tk, "Close").dropna()
            if len(close) < 300:
                continue
        except (KeyError, ValueError):
            continue
        try:
            tdf = build_ticker_df(df, tk)
            old_phases[classify_wyckoff_phase(tdf, tk)] += 1
            r = classify_phase(df, tk)
            new_phases[r.phase] += 1
            new_event_phases[r.event_phase] += 1
            n_valid += 1
        except Exception:
            continue

    print(f"\n{sector} (n={n_valid}):")
    print(f"  OLD:      {dict(old_phases)}")
    print(f"  NEW:      {dict(new_phases)}")
    print(f"  EVENTS:   {dict(new_event_phases)}")