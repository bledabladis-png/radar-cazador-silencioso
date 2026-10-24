"""Muestra el output real del pipeline con TODAS las fases (incluye FORMING)."""
import sys
sys.path.insert(0, ".")
import pandas as pd
from indicators.sector_wyckoff_distribution import (
    compute_sector_wyckoff_distribution, VALID_PHASES,
)

df_stocks = pd.read_parquet("data/stock_prices.parquet")
holdings = pd.read_csv("data/etf_holdings.csv")

print(f"VALID_PHASES: {VALID_PHASES}\n")

result = compute_sector_wyckoff_distribution(df_stocks, holdings)

# Mostrar TODAS las fases con count > 0, no un subset hardcoded
for _, row in result.iterrows():
    sector = row['sector']
    n = int(row['n_valid_wyckoff'])
    print(f"\n{sector}: n_valid={n}, cob={row['coverage_wyckoff']:.0f}%")
    for phase in VALID_PHASES:
        col = f'count_{phase.lower()}'
        pcol = f'pct_{phase.lower()}'
        if col in row.index and row[col] > 0:
            print(f"  {phase:25s} {int(row[col]):3d}  ({row[pcol]:5.1f}%)")

# Resumen agregado
print("\n=== Agregado por fase (suma de todos los sectores) ===")
agg = {}
for phase in VALID_PHASES:
    col = f'count_{phase.lower()}'
    if col in result.columns:
        agg[phase] = int(result[col].sum())
for phase, total in sorted(agg.items(), key=lambda x: -x[1]):
    if total > 0:
        print(f"  {phase:25s} {total:4d}")
print(f"  TOTAL: {sum(agg.values())}")