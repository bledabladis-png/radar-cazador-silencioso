"""Validacion holdout de la regla de vigencia (pre-registro v1)."""
import sys
import time
sys.path.insert(0, ".")
import pandas as pd
from indicators.wyckoff_distribution import classify_distribution_at
from indicators.wyckoff_accumulation import classify_accumulation_at
from indicators.wyckoff_orchestrator import _is_dist_current, _is_acc_current
from src.utils import get_col

HOLDOUT = "data/stock_prices_reserved_cohort.parquet"
df = pd.read_parquet(HOLDOUT)
tickers = sorted(df.columns.get_level_values(1).unique().tolist())
print(f"HOLDOUT: {HOLDOUT}")
print(f"Tickers: {len(tickers)}")
print(f"Rango: {df.index.min().date()} -> {df.index.max().date()}")

# Contadores
n_eval = 0
dist_confirmed_orig = 0
dist_invalidated = 0
acc_confirmed_orig = 0
acc_invalidated = 0

# Diagnostico detallado
dist_details = []
acc_details = []

t0 = time.time()
for i, tk in enumerate(tickers):
    try:
        close = get_col(df, tk, "Close").dropna()
        if len(close) < 300:
            continue
        as_of = close.index[-1]
        current_price = float(close.loc[as_of])
    except Exception:
        continue

    try:
        dist = classify_distribution_at(df, tk, as_of)
        acc = classify_accumulation_at(df, tk, as_of)
    except Exception:
        continue

    n_eval += 1

    if dist.classification == "DISTRIBUTION_CONFIRMED":
        dist_confirmed_orig += 1
        is_current = _is_dist_current(dist, current_price)
        if not is_current:
            dist_invalidated += 1
        dist_details.append((tk, is_current, current_price,
                             dist.range_resistance, dist.range_tol_abs))

    if acc.classification == "ACCUMULATION_CONFIRMED":
        acc_confirmed_orig += 1
        is_current = _is_acc_current(acc, current_price)
        if not is_current:
            acc_invalidated += 1
        acc_details.append((tk, is_current, current_price,
                            acc.range_support, acc.range_tol_abs))

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(tickers)} ({time.time()-t0:.1f}s)")

elapsed = time.time() - t0
print(f"\nEvaluados: {n_eval} ({elapsed:.1f}s)")

# Metricas
print("\n=== DIST ===")
print(f"  Confirmadas originalmente: {dist_confirmed_orig}")
print(f"  Validadas vigentes:        {dist_confirmed_orig - dist_invalidated}")
print(f"  Invadiladas (precio fuera): {dist_invalidated}")
ratio_dist = dist_invalidated / dist_confirmed_orig if dist_confirmed_orig else 0.0
print(f"  Ratio invalidacion: {ratio_dist:.1%}")

print("\n=== ACC ===")
print(f"  Confirmadas originalmente: {acc_confirmed_orig}")
print(f"  Validadas vigentes:        {acc_confirmed_orig - acc_invalidated}")
print(f"  Invadiladas (precio fuera): {acc_invalidated}")
ratio_acc = acc_invalidated / acc_confirmed_orig if acc_confirmed_orig else 0.0
print(f"  Ratio invalidacion: {ratio_acc:.1%}")

# Aplicar criterios pre-registrados
print("\n" + "="*60)
print("=== CRITERIOS PRE-REGISTRADOS ===")
print("="*60)

# C1: ratio invalidacion DIST en [0.20, 0.80]
c1 = 0.20 <= ratio_dist <= 0.80
print(f"C1 (dist invalidados en [20%, 80%]): {ratio_dist:.1%} -> {'PASA' if c1 else 'FALLA'}")

# C2: |ratio_dist - ratio_acc| <= 0.40
c2_diff = abs(ratio_dist - ratio_acc)
c2 = c2_diff <= 0.40
print(f"C2 (|dist - acc| <= 40pp): |{ratio_dist:.1%} - {ratio_acc:.1%}| = {c2_diff:.1%} -> {'PASA' if c2 else 'FALLA'}")

print("\n" + "="*60)
if c1 and c2:
    print("VEREDICTO: REGLA ACEPTADA")
    print("="*60)
    print("La regla se aplica al pipeline. No se ajusta.")
else:
    print("VEREDICTO: REGLA DESCARTADA")
    print("="*60)
    print("El sistema queda como estaba (CONFLICT cuando ambas confirmadas).")
    print("No se ajusta la regla. Se documenta y se cierra.")

# Muestras para diagnostico
print("\n=== Muestras de invalidaciones DIST (top 10) ===")
for tk, is_cur, price, res, tol in sorted(dist_details, key=lambda x: (x[1], -(x[2] or 0)))[:10]:
    if not is_cur:
        lim = (res + tol) if (res is not None and tol is not None) else None
        exceso = (price - lim) / lim * 100 if lim else 0
        print(f"  {tk}: price={price:.2f} resistance+tol={lim:.2f} exceso={exceso:+.1f}%")

print("\n=== Muestras de invalidaciones ACC (top 10) ===")
for tk, is_cur, price, sup, tol in sorted(acc_details, key=lambda x: (x[1], x[2] or 0))[:10]:
    if not is_cur:
        lim = (sup - tol) if (sup is not None and tol is not None) else None
        deficit = (lim - price) / lim * 100 if lim else 0
        print(f"  {tk}: price={price:.2f} support-tol={lim:.2f} deficit={deficit:+.1f}%")