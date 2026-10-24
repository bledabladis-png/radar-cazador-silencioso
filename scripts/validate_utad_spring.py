"""Validacion de pre-registro 61: UTAD_PENDING / SPRING_PENDING."""
import sys
import time
from collections import Counter
sys.path.insert(0, ".")
import pandas as pd
from indicators.wyckoff_distribution import classify_distribution_at
from indicators.wyckoff_accumulation import classify_accumulation_at
from indicators.wyckoff_orchestrator import classify_phase_at
from src.utils import get_col

df = pd.read_parquet("data/stock_prices.parquet")
tickers = sorted(df.columns.get_level_values(1).unique().tolist())
print(f"stock_prices.parquet: {len(tickers)} tickers")

dist_classes = Counter()
acc_classes = Counter()
final_phases = Counter()
utad_pending = []
spring_pending = []

t0 = time.time()
for i, tk in enumerate(tickers):
    try:
        close = get_col(df, tk, "Close").dropna()
        if len(close) < 300:
            continue
        as_of = close.index[-1]
        dist = classify_distribution_at(df, tk, as_of)
        acc = classify_accumulation_at(df, tk, as_of)
        final = classify_phase_at(df, tk, as_of)
    except Exception:
        continue

    dist_classes[dist.classification] += 1
    acc_classes[acc.classification] += 1
    final_phases[final.phase] += 1

    if dist.classification == "UTAD_PENDING":
        utad_pending.append(tk)
    if acc.classification == "SPRING_PENDING":
        spring_pending.append(tk)

elapsed = time.time() - t0
print(f"\nProcesados en {elapsed:.1f}s")

print("\n=== dist_classification ===")
for k, v in dist_classes.most_common():
    print(f"  {k:40s} {v:4d}")

print("\n=== acc_classification ===")
for k, v in acc_classes.most_common():
    print(f"  {k:40s} {v:4d}")

print("\n=== phase (final) ===")
for k, v in final_phases.most_common():
    print(f"  {k:20s} {v:4d}")

print(f"\nUTAD_PENDING: {len(utad_pending)} -> {utad_pending}")
print(f"SPRING_PENDING: {len(spring_pending)} -> {spring_pending}")

# --- Criterios pre-registrados ---
print("\n" + "="*60)
print("=== CRITERIOS PRE-REGISTRADOS (61) ===")
print("="*60)

# A1: no inflacion de DISTRIBUTION confirmadas
n_dist_before = 16  # baseline de la corrida anterior
n_dist_now = final_phases.get("DISTRIBUTION", 0)
a1 = n_dist_now == n_dist_before
print(f"A1 (DISTRIBUTION = {n_dist_before}): {n_dist_now} -> {'PASA' if a1 else 'FALLA'}")

# A2: coherencia - ningun UTAD_PENDING con SOW confirmed, ningun SPRING_PENDING con SOS confirmed
a2_violations = []
for tk in utad_pending:
    close = get_col(df, tk, "Close").dropna()
    dist = classify_distribution_at(df, tk, close.index[-1])
    if dist.sow_status == "CONFIRMED":
        a2_violations.append((tk, "UTAD_PENDING_with_SOW_CONFIRMED"))
for tk in spring_pending:
    close = get_col(df, tk, "Close").dropna()
    acc = classify_accumulation_at(df, tk, close.index[-1])
    if acc.sow_status == "CONFIRMED":
        a2_violations.append((tk, "SPRING_PENDING_with_SOS_CONFIRMED"))
a2 = len(a2_violations) == 0
print(f"A2 (coherencia): {len(a2_violations)} violaciones -> {'PASA' if a2 else 'FALLA'}")
if a2_violations:
    for v in a2_violations[:5]:
        print(f"    {v}")

# A3: simetria
if len(spring_pending) == 0 and len(utad_pending) == 0:
    a3 = True
    print("A3 (simetria): ambos 0 -> PASA")
elif len(spring_pending) == 0 or len(utad_pending) == 0:
    a3 = False
    print("A3 (simetria): uno es 0 -> FALLA")
else:
    ratio = max(len(utad_pending), len(spring_pending)) / min(len(utad_pending), len(spring_pending))
    a3 = ratio <= 5.0
    print(f"A3 (simetria): ratio {ratio:.2f}x -> {'PASA' if a3 else 'FALLA'}")

print("\n" + "="*60)
if a1 and a2 and a3:
    print("VEREDICTO: INCORPORAR UTAD_PENDING / SPRING_PENDING")
else:
    print("VEREDICTO: DESCARTAR (revertir a RANGE_ACTIVE_NO_BREAK)")
print("="*60)