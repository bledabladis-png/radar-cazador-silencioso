"""Diagnostico H2 (causas de RANGE) + verificacion manual H3."""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

from indicators import wyckoff_v1 as w1

df = pd.read_parquet("data/stock_prices.parquet")
tickers = sorted(set(c[1] for c in df.columns if c[0] == "Close"))

# --- Parte 1: causas de RANGE ---
print("=" * 100)
print("H2 - ANALISIS DE CAUSAS DE RANGE")
print("=" * 100)

T_STRONG = 0.30
S_STRONG = 0.30
S_BASE_LOW = -0.20
S_BASE_HIGH = 0.20
C_MIN = 0.30
T_WEAK = 0.30
PREC_WEAK = -0.20
PREC_STRONG = 0.30

reasons = {
    "no_markup_t": 0,
    "no_markup_struct": 0,
    "no_accum_band": 0,
    "no_accum_compression": 0,
    "no_accum_prec": 0,
    "no_accum_trend_weak": 0,
    "near_miss_markup_struct": 0,
    "near_miss_markup_t": 0,
    "near_miss_accum_prec": 0,
    "near_miss_accum_band": 0,
    "near_miss_accum_compression": 0,
}
near_examples = []

total_range = 0
for tk in tickers:
    try:
        tdf = w1.build_ticker_df(df, tk)
        if tdf.empty:
            continue
        phase = w1.classify_wyckoff_phase(tdf, tk)
        if phase != "RANGE":
            continue
        total_range += 1
        combined, struct_s, tact_s, t_norm, c_norm, v_norm, e_norm = w1.wyckoff_score(tdf, tk)
        struct_c = struct_s.dropna()
        t_c = t_norm.dropna()
        c_c = c_norm.dropna()
        if struct_c.empty or t_c.empty or c_c.empty:
            continue
        s = float(struct_c.iloc[-1])
        t = float(t_c.iloc[-1])
        c = float(c_c.iloc[-1])
        prec = w1._compute_precedent(struct_c, t_c, window=w1.PRECEDENT_WINDOW)
        if prec is None:
            continue
        s_min = prec["struct_min"]

        # MARKUP: t > 0.30 AND s > 0.30
        markup_t_ok = t > T_STRONG
        markup_s_ok = s > S_STRONG
        if not markup_t_ok:
            reasons["no_markup_t"] += 1
            if t > T_STRONG - 0.10:
                reasons["near_miss_markup_t"] += 1
        if not markup_s_ok:
            reasons["no_markup_struct"] += 1
            if s > S_STRONG - 0.10:
                reasons["near_miss_markup_struct"] += 1

        # ACCUMULATION: trend_weak AND in_base_band AND compression AND prec_weak
        trend_weak = (t < 0) or (abs(t) < T_WEAK)
        in_base_band = S_BASE_LOW <= s <= S_BASE_HIGH
        compression = c > C_MIN
        prec_weak = s_min < PREC_WEAK

        if not trend_weak:
            reasons["no_accum_trend_weak"] += 1
        if not in_base_band:
            reasons["no_accum_band"] += 1
            # near miss: fuera de banda por poco
            if S_BASE_LOW - 0.05 <= s <= S_BASE_HIGH + 0.05:
                reasons["near_miss_accum_band"] += 1
        if not compression:
            reasons["no_accum_compression"] += 1
            if c > C_MIN - 0.05:
                reasons["near_miss_accum_compression"] += 1
        if not prec_weak:
            reasons["no_accum_prec"] += 1
            if s_min < PREC_WEAK + 0.05:
                reasons["near_miss_accum_prec"] += 1
    except Exception:
        pass

print(f"\nTickers en RANGE: {total_range}")
print("\nCausas de no-MARKUP (por fila, puede sumar >total):")
print(f"  t_norm <= 0.30:                 {reasons['no_markup_t']}")
print(f"    near-miss t (0.20-0.30):      {reasons['near_miss_markup_t']}")
print(f"  struct <= 0.30:                 {reasons['no_markup_struct']}")
print(f"    near-miss struct (0.20-0.30): {reasons['near_miss_markup_struct']}")
print("\nCausas de no-ACCUMULATION:")
print(f"  trend no weak:                  {reasons['no_accum_trend_weak']}")
print(f"  fuera de banda de base:         {reasons['no_accum_band']}")
print(f"    near-miss banda:              {reasons['near_miss_accum_band']}")
print(f"  sin compresion c<=0.30:         {reasons['no_accum_compression']}")
print(f"    near-miss compresion:         {reasons['near_miss_accum_compression']}")
print(f"  sin precedente debil:           {reasons['no_accum_prec']}")
print(f"    near-miss precedente:         {reasons['near_miss_accum_prec']}")

# --- Parte 2: verificacion manual ---
print("\n" + "=" * 100)
print("H3 - VERIFICACION MANUAL")
print("=" * 100)

from indicators import wyckoff as legacy

for tk in ["CHTR", "PM", "MBG.DE", "MCD", "AZO"]:
    print(f"\n{'=' * 100}")
    print(f"TICKER: {tk}")
    print('=' * 100)
    try:
        tdf = w1.build_ticker_df(df, tk)
        if tdf.empty:
            print("  sin datos")
            continue
        close = tdf["Close"]
        ma50 = close.rolling(50, min_periods=50).mean()
        ma200 = close.rolling(200, min_periods=200).mean()
        trend = ma50 / (ma200 + 1e-9) - 1
        last_close = float(close.iloc[-1])
        last_ma50 = float(ma50.dropna().iloc[-1]) if not ma50.dropna().empty else float("nan")
        last_ma200 = float(ma200.dropna().iloc[-1]) if not ma200.dropna().empty else float("nan")
        last_trend = float(trend.dropna().iloc[-1]) if not trend.dropna().empty else float("nan")

        print(f"  close:      {last_close:.2f}")
        print(f"  MA50:       {last_ma50:.2f}")
        print(f"  MA200:      {last_ma200:.2f}")
        print(f"  trend:      {last_trend:+.4f}  (MA50/MA200-1)")
        print(f"  t_norm:     {float(np.tanh(last_trend/w1.WYCKOFF_T_NORM_K)):+.4f}")

        combined, struct_s, tact_s, t_norm, c_norm, v_norm, e_norm = w1.wyckoff_score(tdf, tk)
        s = float(struct_s.dropna().iloc[-1])
        t = float(t_norm.dropna().iloc[-1])
        c = float(c_norm.dropna().iloc[-1])
        ct = float(tact_s.dropna().iloc[-1])
        tt = float(combined.dropna().iloc[-1])
        print(f"  struct:     {s:+.4f}")
        print(f"  tact:       {ct:+.4f}")
        print(f"  combined:   {tt:+.4f}")
        print(f"  c_norm:     {c:+.4f}")

        prec = w1._compute_precedent(struct_s.dropna(), t_norm.dropna(), window=w1.PRECEDENT_WINDOW)
        if prec:
            print(f"  precedente: s_min={prec['struct_min']:+.3f} s_max={prec['struct_max']:+.3f} s_mean={prec['struct_mean']:+.3f}")

        phase = w1.classify_wyckoff_phase(tdf, tk)
        print(f"  FASE v1.3:  {phase}")

        try:
            tdf_leg = legacy.build_ticker_df(df, tk)
            phase_leg = legacy.classify_wyckoff_phase(tdf_leg, tk)
            comb_leg, *_ = legacy.wyckoff_score(tdf_leg, tk)
            s_leg = float(comb_leg.dropna().iloc[-1])
            print(f"  FASE legacy:{phase_leg}  (score {s_leg:+.4f})")
        except Exception as e:
            print(f"  FASE legacy: ERROR {e}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")

print("\n" + "=" * 100)
print("FIN DEL DIAGNOSTICO")
print("=" * 100)
