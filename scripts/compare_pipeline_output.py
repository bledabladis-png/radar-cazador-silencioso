"""Corre compute_sector_wyckoff_distribution con datos reales y compara con CSV actual."""
import sys
sys.path.insert(0, ".")
import pandas as pd
from indicators.sector_wyckoff_distribution import (
    compute_sector_wyckoff_distribution, VALID_PHASES,
)

df_stocks = pd.read_parquet("data/stock_prices.parquet")
holdings = pd.read_csv("data/etf_holdings.csv")

print(f"df_stocks: {df_stocks.shape}")
print(f"holdings: {holdings.shape}, ETFs: {holdings.etf.nunique()}")

result = compute_sector_wyckoff_distribution(df_stocks, holdings)
print(f"\nResultado: {result.shape}")
print(f"Columnas: {result.columns.tolist()}")
print()

# Ver la tabla por sector
for _, row in result.iterrows():
    sector = row['sector']
    n = int(row['n_valid_wyckoff'])
    print(f"\n{sector}: n_valid={n}, cobertura={row['coverage_wyckoff']:.0f}%")
    if n >= 5:
        for phase in VALID_PHASES:
            cnt = row[f'count_{phase.lower()}']
            pct = row[f'pct_{phase.lower()}']
            if cnt > 0:
                print(f"  {phase:15s} {int(cnt):3d} ({pct:5.1f}%)")

# Comparar con CSV actual
print("\n\n=== CSV actual en produccion ===")
old = pd.read_csv("outputs/history/sector_wyckoff_distribution.csv")
old_latest = old[old.date == old.date.max()]
print(f"Fecha: {old_latest.date.iloc[0]}")
for _, row in old_latest.iterrows():
    sector = row['sector']
    n = int(row['n_valid_wyckoff'])
    print(f"\n{sector}: n_valid={n}, cobertura={row['coverage_wyckoff']:.0f}%")
    for phase in ["accumulation","markup","range","distribution","markdown"]:
        cnt = row[f'count_{phase}']
        pct = row[f'pct_{phase}']
        if cnt > 0:
            print(f"  {phase:15s} {int(cnt):3d} ({pct:5.1f}%)")