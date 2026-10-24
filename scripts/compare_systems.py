import sys
import time
import numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, ".")
import pandas as pd
from indicators.wyckoff_v1 import build_ticker_df, classify_wyckoff_phase
from indicators.wyckoff_orchestrator import classify_phase

df = pd.read_parquet("data/stock_prices.parquet")
tickers = sorted(df.columns.get_level_values(1).unique().tolist())
print(f"stock_prices.parquet: {len(tickers)} tickers")

OLD_VALID = {"ACCUMULATION", "MARKUP", "RANGE", "DISTRIBUTION", "MARKDOWN"}

matrix = defaultdict(Counter)  # old -> new
errored_old = []
errored_new = []
per_ticker = {}

t0 = time.time()
for i, tk in enumerate(tickers):
    try:
        close = df[("Close", tk)].dropna()
        if len(close) < 300:
            continue
    except Exception:
        continue

    try:
        tdf = build_ticker_df(df, tk)
        old_phase = classify_wyckoff_phase(tdf, tk)
        if old_phase not in OLD_VALID:
            old_phase = f"INVALID:{old_phase}"
    except Exception as e:
        old_phase = f"ERR:{type(e).__name__}"
        errored_old.append((tk, str(e)))

    try:
        new_res = classify_phase(df, tk)
        new_phase = new_res.phase
    except Exception as e:
        new_phase = f"ERR:{type(e).__name__}"
        errored_new.append((tk, str(e)))

    matrix[old_phase][new_phase] += 1
    per_ticker[tk] = (old_phase, new_phase)

elapsed = time.time() - t0
n_total = sum(sum(c.values()) for c in matrix.values())
print(f"Procesados en {elapsed:.1f}s. Auditados: {n_total}")
print(f"Errores old: {len(errored_old)} | Errores new: {len(errored_new)}")

# Matriz de confusion
old_phases = sorted(matrix.keys())
new_phases = sorted({np for c in matrix.values() for np in c.keys()})
print("\n=== Matriz de confusion (filas=old, columnas=new) ===")
header = "old\\new".ljust(15) + "".join(p.ljust(22) for p in new_phases) + "TOTAL"
print(header)
for op in old_phases:
    row = op.ljust(15)
    row_total = 0
    for np in new_phases:
        v = matrix[op][np]
        row_total += v
        row += (str(v) if v else ".").ljust(22)
    row += str(row_total)
    print(row)
print()
col_totals = {np: sum(matrix[op][np] for op in old_phases) for np in new_phases}
print("TOTAL".ljust(15) + "".join(str(col_totals[np]).ljust(22) for np in new_phases) + str(n_total))

# Distribuciones marginales
print("\n=== Viejo (classify_wyckoff_phase) ===")
old_total = Counter()
for op in matrix:
    old_total[op] = sum(matrix[op].values())
for k, v in old_total.most_common():
    print(f"  {k:30s} {v:4d}  ({100*v/n_total:5.1f}%)")

print("\n=== Nuevo (classify_phase) ===")
for k, v in Counter(col_totals).most_common():
    print(f"  {k:30s} {v:4d}  ({100*v/n_total:5.1f}%)")

# Casos discrepantes: viejos DISTRIBUTION vs nuevos no DISTRIBUTION, etc
print("\n=== Discrepancias DISTRIBUTION (old) vs DISTRIBUTION (new) ===")
discrepancias = []
for tk, (o, n) in per_ticker.items():
    if o == "DISTRIBUTION" and n != "DISTRIBUTION":
        discrepancias.append((tk, o, n))
print(f"old=DISTRIBUTION, new!=DISTRIBUTION: {len(discrepancias)}")
for tk, o, n in discrepancias[:15]:
    print(f"  {tk}: old={o} new={n}")

print("\n=== Discrepancias ACCUMULATION (old) vs ACCUMULATION (new) ===")
discrepancias = []
for tk, (o, n) in per_ticker.items():
    if o == "ACCUMULATION" and n != "ACCUMULATION":
        discrepancias.append((tk, o, n))
print(f"old=ACCUMULATION, new!=ACCUMULATION: {len(discrepancias)}")
for tk, o, n in discrepancias[:15]:
    print(f"  {tk}: old={o} new={n}")

# MARKUP/MARKDOWN: que les pasa en el nuevo
print("\n=== MARKUP/MARKDOWN (old) mapeados en nuevo ===")
for old_phase in ["MARKUP", "MARKDOWN"]:
    dist = Counter()
    for tk, (o, n) in per_ticker.items():
        if o == old_phase:
            dist[n] += 1
    print(f"  {old_phase}: {dict(dist)}")