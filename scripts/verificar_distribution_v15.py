"""Verificacion manual de los 10 DISTRIBUTION de v1.5.

Para cada ticker: OHLCV crudo, MA50, MA200, trend, t_norm, struct, c_norm,
precedente, fase. Confirmar contra el precio observable.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, ".")
from indicators import wyckoff_v1 as w1
from indicators import wyckoff as legacy

df = pd.read_parquet("data/stock_prices.parquet")

tickers = ["ACS.MC", "ANA.MC", "BA", "BKNG", "O", "PCG", "SLB", "TEF.MC", "VMRK", "WFC"]

for tk in tickers:
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
        print(f"  trend:      {last_trend:+.4f}")
        print(f"  t_norm:     {float(np.tanh(last_trend/w1.WYCKOFF_T_NORM_K)):+.4f}")

        combined, struct_s, tact_s, t_norm, c_norm, v_norm, e_norm = w1.wyckoff_score(tdf, tk)
        s = float(struct_s.dropna().iloc[-1])
        t = float(t_norm.dropna().iloc[-1])
        c = float(c_norm.dropna().iloc[-1])
        ct = float(tact_s.dropna().iloc[-1])
        print(f"  struct:     {s:+.4f}")
        print(f"  tact:       {ct:+.4f}")
        print(f"  c_norm:     {c:+.4f}")

        prec = w1._compute_precedent(struct_s.dropna(), t_norm.dropna(), window=w1.PRECEDENT_WINDOW)
        if prec:
            print(f"  precedente: s_min={prec['struct_min']:+.3f} s_max={prec['struct_max']:+.3f} s_mean={prec['struct_mean']:+.3f}")

        phase = w1.classify_wyckoff_phase(tdf, tk)
        print(f"  FASE v1.5:  {phase}")

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
print("FIN")
print("=" * 100)
